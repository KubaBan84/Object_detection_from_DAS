"""All plotting helpers (time on the vertical axis, space horizontal - as in the brief).

Scripts select a non-interactive backend themselves (``matplotlib.use("Agg")``) before
importing this module, so the helpers also work inside notebooks.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


SOURCE_COLOURS = {"strong": "red", "weak": "cyan", "hough_global": "yellow"}


def plot_segments(ax: Axes, segments, row_dt: float, col_dx: float, colours=None,
                  lw: float = 1.0, alpha: float = 0.9) -> None:
    """Draw Stage-6 pixel segments (``detection.LineSegment``) in physical coordinates.

    ``colours``: None -> by source (strong red, HoughLinesP cyan, global Hough yellow);
    a single colour; or a list with one colour per segment (e.g. by group).
    Pixel centres are at ((c + 0.5) col_dx, (r + 0.5) row_dt), as in :func:`plot_waterfall`.
    """
    for k, s in enumerate(segments):
        if colours is None:
            col = SOURCE_COLOURS.get(s.source, "white")
        elif isinstance(colours, str):
            col = colours
        else:
            col = colours[k]
        ax.plot([(s.c0 + 0.5) * col_dx, (s.c1 + 0.5) * col_dx],
                [(s.r0 + 0.5) * row_dt, (s.r1 + 0.5) * row_dt], color=col, lw=lw, alpha=alpha,
                ls="--" if s.band == "lf" else "-")


SPEED_CLASS_COLOURS = {"constant": "deepskyblue", "varying": "orange", "": "silver"}


def plot_detections(img, objects, row_dt, col_dx, t0, ax=None, path=None, title: str = "",
                    vmin: float | None = None, vmax: float | None = None, cmap: str = "viridis",
                    label_fontsize: float = 7.0, extend: bool = False, tracks: bool = False,
                    lines: bool = True, label: bool = True, colours=None):
    """Waterfall with the detected objects: red lines + velocity labels [km/h].

    As in the figure of the brief: strong (thick) objects solid, weak (thin) ones dashed;
    the label is the reported velocity (track fit of Stage 8 when available, else the
    Stage-7 line). With ``extend`` the averaged line is drawn over the whole cable.
    With ``tracks`` the measured (curved) trajectory of Stage 9 is drawn too, coloured by
    speed class (constant blue, varying orange). ``colours`` (one per object, e.g. by
    cluster in Stage 10) replaces the red of the lines and labels. Returns the axes.
    """
    from .velocity import reported_velocity_kmh

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 9), layout="constrained")
    plot_waterfall(img, t0, row_dt, col_dx, ax=ax, colorbar=False, vmin=vmin, vmax=vmax,
                   cmap=cmap, title=title)
    width, height = img.shape[1] * col_dx, img.shape[0] * row_dt
    for k, o in enumerate(objects):
        col = "red" if colours is None else colours[k]
        if lines:
            xs = np.array([0.0, width]) if extend else np.array([o.x_start_m, o.x_end_m])
            ax.plot(xs, o.t_at(xs), color=col, lw=1.4 if o.kind == "strong" else 1.0,
                    ls="-" if o.kind == "strong" else "--")
        has_track = tracks and o.track is not None and len(o.track) > 1
        if has_track:
            ax.plot(o.track[:, 0], o.track[:, 1], color=SPEED_CLASS_COLOURS.get(o.speed_class, "w"),
                    lw=1.3, marker=".", ms=2.5)
        if label:
            if has_track:
                k = len(o.track) // 2
                xm, tm = o.track[k]
            else:
                xm = 0.5 * (o.x_start_m + o.x_end_m)
                tm = float(o.t_at(xm))
            ax.text(xm, tm, f"{abs(reported_velocity_kmh(o)):.2f} km/h", color=col,
                    fontsize=label_fontsize, ha="center", va="bottom", fontweight="bold",
                    clip_on=True)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    if path is not None:
        save_figure(ax.figure, path)
    return ax


def plot_velocity_profiles(objects, t0: datetime | None = None, ax: Axes | None = None,
                           colours=None, title: str = "", path=None):
    """Combined v(t) plot of one segment: |v| of every tracked object versus time.

    Each curve is the local velocity profile of Stage 9 (sliding-window regression),
    drawn over the time the object is on the cable and labelled with its id. Colours:
    by speed class by default (constant blue, varying orange), or one colour per object
    (``colours``, e.g. by cluster in Stage 10). Dashed horizontal = constant-velocity fit.
    """
    from .velocity import reported_velocity_kmh

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for k, o in enumerate(objects):
        if o.profile is None or len(o.profile) == 0:
            continue
        col = colours[k] if colours is not None else SPEED_CLASS_COLOURS.get(o.speed_class, "gray")
        t, v = o.profile[:, 0], np.abs(o.profile[:, 2])
        ax.plot(t, v, color=col, lw=1.6 if o.kind == "strong" else 1.0,
                ls="-" if o.kind == "strong" else "--")
        ax.plot([o.track[:, 1].min(), o.track[:, 1].max()], [abs(reported_velocity_kmh(o))] * 2,
                color=col, lw=0.6, ls=":")
        ax.text(t[-1], v[-1], f" {o.obj_id}", color=col, fontsize=7, va="center")
    ax.xaxis.set_major_formatter(hms_formatter(t0))
    ax.set_xlabel("time")
    ax.set_ylabel("|v| [km/h]")
    ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=c, label=n) for n, c in SPEED_CLASS_COLOURS.items() if n]
    handles += [plt.Line2D([], [], color="k", ls="-", label="strong"),
                plt.Line2D([], [], color="k", ls="--", label="weak"),
                plt.Line2D([], [], color="k", ls=":", lw=0.6, label="constant-velocity fit")]
    ax.legend(handles=handles, fontsize=8, ncol=5, loc="upper right")
    if title:
        ax.set_title(title)
    if path is not None:
        save_figure(ax.figure, path)
    return ax


def plot_profile_gallery(objects, t0: datetime | None = None, ncols: int = 6,
                         references: dict | None = None, title: str = "") -> Figure:
    """Small multiples: v(t) of every tracked object with its constant and quadratic fits.

    Each panel: local profile (dots), constant-velocity fit (dotted) and the velocity of
    the quadratic x(t) fit (line from v_start to v_end), title with id, kind, class and
    acceleration. ``references`` {obj_id: (t, v)} adds a ground-truth profile (black).
    """
    from .velocity import reported_velocity_kmh

    objs = [o for o in objects if o.profile is not None and len(o.profile) > 0]
    nrows = max(int(np.ceil(len(objs) / ncols)), 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.1 * nrows), squeeze=False,
                             layout="constrained")
    for ax, o in zip(axes.ravel(), objs):
        col = SPEED_CLASS_COLOURS.get(o.speed_class, "gray")
        t, v = o.profile[:, 0], np.abs(o.profile[:, 2])
        ax.plot(t, v, ".", color=col, ms=3)
        t0_, t1_ = o.track[:, 1].min(), o.track[:, 1].max()
        ax.plot([t0_, t1_], [abs(reported_velocity_kmh(o))] * 2, ":", color="gray", lw=1)
        ax.plot([t0_, t1_], [abs(o.v_start_kmh), abs(o.v_end_kmh)], "-", color=col, lw=1.2)
        if references and o.obj_id in references:
            tr, vr = references[o.obj_id]
            ax.plot(tr, np.abs(vr), "-", color="k", lw=0.8)
        ax.set_title(f"#{o.obj_id} {o.kind[0]} {o.speed_class} a={o.accel_ms2:+.2f}", fontsize=8)
        ax.xaxis.set_major_formatter(hms_formatter(t0, "%M:%S") if t0 else hms_formatter(None))
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=7)
        lo, hi = np.nanmin(v), np.nanmax(v)
        pad = max(5.0, 0.15 * (hi - lo))
        ax.set_ylim(max(lo - pad, 0), hi + pad)
    for ax in axes.ravel()[len(objs):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("|v| [km/h]", fontsize=8)
    if title:
        fig.suptitle(title)
    return fig


CLUSTER_COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                   "tab:brown", "tab:pink", "tab:olive"]


def cluster_colour(label: int) -> str:
    return CLUSTER_COLOURS[int(label) % len(CLUSTER_COLOURS)]


def plot_clusters(X2d, labels, names, ax: Axes | None = None, groups=None,
                  explained=None, annotate=None, title: str = "", path=None) -> Axes:
    """2-D PCA scatter of the objects coloured by cluster.

    ``groups`` (optional, one string per object, e.g. the segment or the synthetic type)
    selects the marker; ``annotate`` (optional strings) labels the points; ``explained``
    (2,) variance ratios go into the axis labels.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
    labels = np.asarray(labels)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    gnames = sorted(set(groups)) if groups is not None else [None]
    for c, name in enumerate(names):
        for gi, g in enumerate(gnames):
            sel = labels == c
            if g is not None:
                sel &= np.array([x == g for x in groups])
            if sel.any():
                ax.scatter(X2d[sel, 0], X2d[sel, 1], color=cluster_colour(c),
                           marker=markers[gi % len(markers)], s=36, edgecolor="k", lw=0.4,
                           alpha=0.85)
    if annotate is not None:
        for (x, y), s in zip(X2d, annotate):
            ax.annotate(s, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    handles = [plt.Line2D([], [], color=cluster_colour(c), marker="o", ls="",
                          label=f"{c}: {n} ({int(np.sum(labels == c))})")
               for c, n in enumerate(names)]
    if groups is not None:
        handles += [plt.Line2D([], [], color="gray", marker=markers[gi % len(markers)], ls="",
                               label=str(g)) for gi, g in enumerate(gnames)]
    ax.legend(handles=handles, fontsize=7, loc="best")
    ev = explained if explained is not None else (np.nan, np.nan)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.0f} % of variance)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.0f} % of variance)")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    if path is not None:
        save_figure(ax.figure, path)
    return ax


def plot_k_selection(scores: dict, chosen: int, title: str = "", path=None,
                     ax: Axes | None = None) -> Figure:
    """Silhouette and k-means compactness (elbow) versus the number of clusters."""
    ks = sorted(scores)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    else:
        fig = ax.figure
    ax.plot(ks, [scores[k]["silhouette"] for k in ks], "o-", color="C0")
    ax.axvline(chosen, color="gray", ls=":")
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("mean silhouette", color="C0")
    ax2 = ax.twinx()
    ax2.plot(ks, [scores[k]["compactness"] for k in ks], "s--", color="C1")
    ax2.set_ylabel("compactness (elbow)", color="C1")
    ax.set_title(title or f"choice of k (chosen k = {chosen})", fontsize=10)
    if path is not None:
        save_figure(fig, path)
    return fig


def plot_cluster_features(X, labels, keys, names, title: str = "", path=None) -> Figure:
    """Box plots of the clustering features (physical units) per cluster."""
    labels = np.asarray(labels)
    n = len(keys)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows),
                             layout="constrained", squeeze=False)
    for j, key in enumerate(keys):
        ax = axes.flat[j]
        data = [X[(labels == c) & np.isfinite(X[:, j]), j] for c in range(len(names))]
        pos = [c for c, d in enumerate(data) if d.size]
        if pos:
            bp = ax.boxplot([data[c] for c in pos], positions=pos, widths=0.6,
                            patch_artist=True, showfliers=True)
            for patch, c in zip(bp["boxes"], pos):
                patch.set_facecolor(cluster_colour(c))
                patch.set_alpha(0.6)
        ax.set_title(key, fontsize=9)
        ax.set_xticks(range(len(names)), [str(c) for c in range(len(names))], fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title or "features per cluster: " + "; ".join(f"{c} = {nm}" for c, nm
                                                               in enumerate(names)), fontsize=9)
    if path is not None:
        save_figure(fig, path)
    return fig


def plot_cluster_gallery(thumbs, labels, names, captions, row_dt: float, dx: float,
                         per_cluster: int = 8, title: str = "", path=None) -> Figure:
    """Gallery of straightened line images ("shape of the lines"), one row per cluster.

    ``thumbs`` - list of (2w + 1, channels) images (time offset from the ridge vertical,
    position along the cable horizontal); the ``per_cluster`` objects with the largest
    peak are shown with their ``captions``.
    """
    labels = np.asarray(labels)
    k = len(names)
    fig, axes = plt.subplots(k, per_cluster, figsize=(1.6 * per_cluster, 2.3 * k),
                             layout="constrained", squeeze=False)
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        strength = [np.nanmax(thumbs[i]) if np.isfinite(thumbs[i]).any() else -np.inf for i in idx]
        idx = idx[np.argsort(strength)[::-1]][:per_cluster]
        for j in range(per_cluster):
            ax = axes[c, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= len(idx):
                ax.set_visible(False)
                continue
            th = thumbs[idx[j]]
            h = (th.shape[0] - 1) / 2 * row_dt
            vmax = np.nanpercentile(th, 99) if np.isfinite(th).any() else 1.0
            ax.imshow(np.nan_to_num(th, nan=0.0), aspect="auto", cmap="magma", vmin=0,
                      vmax=max(vmax, 1e-6), extent=(0, th.shape[1] * dx, h, -h),
                      interpolation="nearest")
            ax.set_title(captions[idx[j]], fontsize=6.5)
            for side in ax.spines.values():
                side.set_color(cluster_colour(c))
                side.set_linewidth(2.0)
            if j == 0:
                ax.set_ylabel(f"{c}: {names[c]}\n(n={int(np.sum(labels == c))})", fontsize=7,
                              color=cluster_colour(c))
    fig.suptitle(title or "straightened lines per cluster (vertical: time offset from the "
                          "ridge, +-2 s; horizontal: cable position)", fontsize=9)
    if path is not None:
        save_figure(fig, path)
    return fig


def save_figure(fig: Figure, path: Path | str, dpi: int = 120) -> Path:
    """Save (creating parent directories) and close the figure. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
