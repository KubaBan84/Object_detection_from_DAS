"""All plotting helpers (time on the vertical axis, space horizontal - as in the brief).

Scripts select a non-interactive backend themselves (``matplotlib.use("Agg")``) before
importing this module, so the helpers also work inside notebooks.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator


def hms_formatter(t0: datetime | None, fmt: str = "%H:%M:%S") -> FuncFormatter:
    """Tick formatter: seconds relative to ``t0`` -> wall-clock time (or plain seconds)."""
    if t0 is None:
        return FuncFormatter(lambda s, _pos: f"{s:g} s")
    return FuncFormatter(lambda s, _pos: (t0 + timedelta(seconds=float(s))).strftime(fmt))


def plot_waterfall(img: np.ndarray, t0: datetime | None, row_dt: float, col_dx: float,
                   x0_m: float = 0.0, ax: Axes | None = None, title: str = "",
                   cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None,
                   colorbar: bool = True, cbar_label: str = "", t_offset_s: float = 0.0,
                   n_time_ticks: int = 7):
    """Display a (time, space) image with physical axes, time running downwards.

    ``t_offset_s`` is the time of the first row relative to ``t0``. Returns (ax, image).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 8))
    n_rows, n_cols = img.shape
    extent = (x0_m, x0_m + n_cols * col_dx, t_offset_s + n_rows * row_dt, t_offset_s)
    im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                   interpolation="nearest")
    ax.yaxis.set_major_locator(MaxNLocator(n_time_ticks))
    ax.yaxis.set_major_formatter(hms_formatter(t0))
    ax.set_xlabel("distance along fibre [m]")
    ax.set_ylabel("time")
    if title:
        ax.set_title(title)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        if cbar_label:
            cb.set_label(cbar_label)
    return ax, im


def plot_overview_panels(panels: list[dict], row_dt: float, col_dx: float, ncols: int,
                         vmin: float, vmax: float, cmap: str = "viridis",
                         cbar_label: str = "", suptitle: str = "") -> Figure:
    """Grid of waterfall panels sharing one colour scale (overview of a long recording).

    Each panel dict holds ``img`` (time, space), ``t0`` (datetime of the first row),
    ``title`` and optional ``highlights``: list of (start_s, end_s, label, colour) spans
    (relative to the panel's ``t0``) drawn as framed bands.
    """
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 6.0 * nrows), squeeze=False,
                             layout="constrained")
    im = None
    for ax, panel in zip(axes.ravel(), panels):
        _, im = plot_waterfall(panel["img"], panel["t0"], row_dt, col_dx, ax=ax, cmap=cmap,
                               vmin=vmin, vmax=vmax, colorbar=False, n_time_ticks=5)
        ax.set_title(panel.get("title", ""), fontsize=9)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
        width = panel["img"].shape[1] * col_dx
        for start, end, label, colour in panel.get("highlights", []):
            ax.add_patch(plt.Rectangle((0, start), width, end - start, fill=False,
                                       edgecolor=colour, linewidth=2.5))
            ax.text(width / 2, start + (end - start) / 2, label, color=colour, fontsize=9,
                    ha="center", va="center", rotation=90, fontweight="bold")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
        cb.set_label(cbar_label)
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def plot_stages(stages: dict[str, np.ndarray], row_dt: float, col_dx: float | dict,
                t0: datetime | None = None, title: str = "", cmaps: dict | None = None,
                symmetric: tuple[str, ...] = (), panel_width: float = 2.6) -> Figure:
    """One row of waterfall panels, one per processing step (before -> after, removed parts).

    Every panel has its own colour scale (percentiles 1-99.5; symmetric around 0 for the
    names in ``symmetric``, e.g. zero-mean filter outputs). ``col_dx`` may be a dict
    {name: dx} when panels have different column spacing (e.g. after upsampling).
    """
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(panel_width * n, 8), sharey=True,
                             layout="constrained", squeeze=False)
    for ax, (name, img) in zip(axes[0], stages.items()):
        dx = col_dx[name] if isinstance(col_dx, dict) else col_dx
        cmap = (cmaps or {}).get(name, "RdBu_r" if name in symmetric else "viridis")
        if name in symmetric:
            lim = float(np.percentile(np.abs(img), 99.5)) or 1.0
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = (float(v) for v in np.percentile(img, [1, 99.5]))
        plot_waterfall(img, t0, row_dt, dx, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                       colorbar=False, title=name.replace("_", " "))
        ax.set_xlabel("x [m]")
        if ax is not axes[0, 0]:
            ax.set_ylabel("")
    if title:
        fig.suptitle(title)
    return fig


def plot_segments(ax: Axes, segments, row_dt: float, col_dx: float, colours="red",
                  lw: float = 1.0, alpha: float = 0.9) -> None:
    """Draw Stage-6 pixel segments (``detection.LineSegment``) in physical coordinates.

    ``colours``: one colour for all segments or a list with one colour per segment (e.g.
    by group). Pixel centres are at ((c + 0.5) col_dx, (r + 0.5) row_dt), as in
    :func:`plot_waterfall`.
    """
    for k, s in enumerate(segments):
        col = colours if isinstance(colours, str) else colours[k]
        ax.plot([(s.c0 + 0.5) * col_dx, (s.c1 + 0.5) * col_dx],
                [(s.r0 + 0.5) * row_dt, (s.r1 + 0.5) * row_dt], color=col, lw=lw, alpha=alpha)


def plot_detections(img, objects, row_dt, col_dx, t0, ax=None, path=None, title: str = "",
                    vmin: float | None = None, vmax: float | None = None, cmap: str = "viridis",
                    label_fontsize: float = 7.0, extend: bool = False, label: bool = True,
                    colours=None):
    """Waterfall with the detected objects: red lines + velocity labels "xx.xx km/h".

    As in the figure of the brief. With ``extend`` the averaged line is drawn over the
    whole cable, otherwise over the detected extent. ``colours`` (one per object)
    replaces the red of the lines and labels. Returns the axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 9), layout="constrained")
    plot_waterfall(img, t0, row_dt, col_dx, ax=ax, colorbar=False, vmin=vmin, vmax=vmax,
                   cmap=cmap, title=title)
    width, height = img.shape[1] * col_dx, img.shape[0] * row_dt
    for k, o in enumerate(objects):
        col = "red" if colours is None else colours[k]
        xs = np.array([0.0, width]) if extend else np.array([o.x_start_m, o.x_end_m])
        ax.plot(xs, o.t_at(xs), color=col, lw=1.4)
        if label:
            xm = 0.5 * (o.x_start_m + o.x_end_m)
            ax.text(xm, float(o.t_at(xm)), f"{abs(o.velocity_kmh):.2f} km/h", color=col,
                    fontsize=label_fontsize, ha="center", va="bottom", fontweight="bold",
                    clip_on=True,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    if path is not None:
        save_figure(ax.figure, path)
    return ax


def save_figure(fig: Figure, path: Path | str, dpi: int = 120) -> Path:
    """Save (creating parent directories) and close the figure. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
