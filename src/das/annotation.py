"""Manual labels of the real segments (plan: Stage 9) - storage, session state, labelling GUI.

The real recordings have no ground truth, so the visible lines are marked by hand on the
raw energy waterfall (log RMS, the image of the brief - *not* the detector output, to
keep the reference independent of the algorithm). A line is a polyline of >= 2 clicks
in physical units (x [m] along the cable, t [s] from the segment start) with the label
"thick" (tram / truck) or "thin" (car). Labels live in
``data/annotations/manual_labels.json`` (versioned)::

    {"segments": {"seg_kuba": {"start": "090452", "annotator": "...", "saved": "...",
                               "lines": [{"id": 1, "kind": "thin",
                                          "points": [[x, t], ...]}, ...]}}}

``Annotator`` is the matplotlib tool used by ``scripts/annotate_real.py``; the logic is
in ``AnnotationSession`` so it can be tested without a display.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config

LABEL_KINDS = ("thick", "thin")
KIND_STYLE = {"thick": dict(color="red", lw=1.6, ls="-"),
              "thin": dict(color="cyan", lw=1.1, ls="--")}

HELP = """Annotation keys (mouse over the image):
  left click      add a point of the current line (>= 2 points, along the line)
  right click     remove the last point
  t / n           finish the current line as THICK (tram, truck) / THIN (car)
  u               undo: last point, or the last finished line
  d               delete the finished line nearest to the mouse
  up / down       move the view by half a window (pageup / pagedown: a whole window)
  a               toggle: whole segment / window view
  b               next background image (raw RMS -> Stage-4 image)
  s               save           q  save and quit           h  this help
Labels are saved automatically after every finished / undone / deleted line and when
the window is closed; an unfinished line (no t / n yet) is not saved.
Use the toolbar zoom / pan freely - clicks are ignored while a toolbar mode is active."""


# ======================================================================================
# Storage
# ======================================================================================
def load_labels(path: Path | str | None = None) -> dict:
    """The labels file (an empty structure if it does not exist yet)."""
    path = Path(path or config.ANNOT_FILE)
    if not path.exists():
        return {"segments": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("segments", {})
    return data


def save_labels(data: dict, path: Path | str | None = None) -> Path:
    path = Path(path or config.ANNOT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def segment_lines(data: dict, name: str) -> list[dict]:
    """Labelled lines of one segment ([] if not annotated)."""
    return list(data.get("segments", {}).get(name, {}).get("lines", []))


# ======================================================================================
# Session state (no GUI)
# ======================================================================================
class AnnotationSession:
    """Lines finished so far + the points of the line being drawn."""

    def __init__(self, lines: list[dict] | None = None):
        self.lines: list[dict] = [dict(ln) for ln in (lines or [])]
        self.current: list[list[float]] = []

    def _next_id(self) -> int:
        return max((int(ln.get("id", 0)) for ln in self.lines), default=0) + 1

    def add_point(self, x: float, t: float) -> None:
        self.current.append([float(x), float(t)])

    def remove_point(self) -> bool:
        if self.current:
            self.current.pop()
            return True
        return False

    def finish(self, kind: str) -> dict | None:
        """Close the current polyline with a label; None if it has < 2 points."""
        if kind not in LABEL_KINDS:
            raise ValueError(f"kind must be one of {LABEL_KINDS}")
        if len(self.current) < 2:
            return None
        pts = np.asarray(self.current)
        if np.ptp(pts[:, 0]) <= 0:                      # vertical: no velocity
            return None
        line = {"id": self._next_id(), "kind": kind,
                "points": [[round(x, 3), round(t, 4)] for x, t in self.current]}
        self.lines.append(line)
        self.current = []
        return line

    def undo(self) -> str:
        """Remove the last point, or - with no line in progress - the last finished line."""
        if self.remove_point():
            return "point"
        if self.lines:
            self.lines.pop()
            return "line"
        return ""

    def delete_nearest(self, x: float, t: float, x_scale: float = 1.0, t_scale: float = 1.0
                       ) -> dict | None:
        """Delete the finished line closest to (x, t) (distance to its polyline vertices
        and the interpolated time at x, axes scaled by x_scale / t_scale)."""
        if not self.lines:
            return None
        best, best_d = None, np.inf
        for k, ln in enumerate(self.lines):              # loop over lines
            pts = np.asarray(ln["points"], float)
            pts = pts[np.argsort(pts[:, 0])]
            d_vert = np.min(np.hypot((pts[:, 0] - x) / x_scale, (pts[:, 1] - t) / t_scale))
            d = d_vert
            if pts[0, 0] <= x <= pts[-1, 0]:
                d = min(d, abs(np.interp(x, pts[:, 0], pts[:, 1]) - t) / t_scale)
            if d < best_d:
                best, best_d = k, d
        return self.lines.pop(best)

    def to_dict(self, start: str, annotator: str, background: str) -> dict:
        return {"start": start, "annotator": annotator, "background": background,
                "time_reference": "t [s] from the segment start, x [m] = (channel + 0.5) dx",
                "saved": datetime.now().isoformat(timespec="seconds"), "lines": self.lines}


# ======================================================================================
# Interactive tool
# ======================================================================================
class Annotator:
    """Matplotlib GUI around an :class:`AnnotationSession` (see ``HELP``).

    ``backgrounds`` - list of (name, image (rows, channels)); ``on_save(session)`` is
    called by "s" / "q". Event handlers are plain methods, so they can be driven by
    synthetic events in tests.
    """

    def __init__(self, backgrounds: list[tuple[str, np.ndarray]], row_dt: float, dx: float,
                 t0: datetime | None, session: AnnotationSession, on_save, title: str = "",
                 window_s: float = 30.0):
        import matplotlib.pyplot as plt

        from .visualization import hms_formatter

        self.plt = plt
        # free the keys used here from the default matplotlib shortcuts (s = save dialog,
        # q = close, h = home, ...)
        for name in ("keymap.save", "keymap.quit", "keymap.home", "keymap.back",
                     "keymap.forward", "keymap.yscale", "keymap.xscale", "keymap.grid"):
            plt.rcParams[name] = [k for k in plt.rcParams[name]
                                  if k not in ("s", "q", "h", "a", "b", "d", "n", "t", "u")]
        self.backgrounds, self.bg_index = backgrounds, 0
        self.row_dt, self.dx, self.session, self.on_save = row_dt, dx, session, on_save
        self.title, self.window_s = title, window_s
        img = backgrounds[0][1]
        self.width, self.height = img.shape[1] * dx, img.shape[0] * row_dt
        self.view_t0, self.full_view = 0.0, False
        self.fig, self.ax = plt.subplots(figsize=(8, 10), layout="constrained")
        self.im = self.ax.imshow(img, aspect="auto", cmap="viridis", interpolation="nearest",
                                 extent=(0, self.width, self.height, 0), **self._clim(img))
        self.ax.yaxis.set_major_formatter(hms_formatter(t0))
        self.ax.set_xlabel("position along the cable [m]")
        self.ax.set_ylabel("time")
        self.artists: list = []
        self.status = ""
        # matplotlib keeps only *weak* references to bound-method callbacks: without a
        # strong reference held by the figure, an unassigned Annotator would be garbage
        # collected and the window would silently ignore every click and key
        self.fig._das_annotator = self
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self._saved_lines = json.dumps(session.lines)   # state of the file on disk
        self._apply_view()
        self.redraw()

    def autosave(self, force: bool = False) -> bool:
        """Save the finished lines if they changed since the last save (or ``force``)."""
        state = json.dumps(self.session.lines)
        if not force and state == self._saved_lines:
            return False
        path = self.on_save(self.session)
        self._saved_lines = state
        self.status = f"saved {len(self.session.lines)} lines -> {Path(path).name}"
        return True

    @staticmethod
    def _clim(img: np.ndarray) -> dict:
        return {"vmin": float(np.percentile(img, 1)), "vmax": float(np.percentile(img, 99.5))}

    def _toolbar_active(self) -> bool:
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(tb is not None and getattr(tb, "mode", ""))

    def _apply_view(self) -> None:
        if self.full_view:
            self.ax.set_ylim(self.height, 0)
        else:
            self.view_t0 = float(np.clip(self.view_t0, 0, max(self.height - self.window_s, 0)))
            self.ax.set_ylim(self.view_t0 + self.window_s, self.view_t0)
        self.ax.set_xlim(0, self.width)

    def redraw(self) -> None:
        """Redraw the finished lines, the line being drawn and the status title."""
        for a in self.artists:
            a.remove()
        self.artists = []
        for ln in self.session.lines:
            pts = np.asarray(ln["points"])
            self.artists += self.ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3,
                                         **KIND_STYLE[ln["kind"]])
            self.artists.append(self.ax.text(pts[-1, 0], pts[-1, 1], f" {ln['id']}",
                                             color=KIND_STYLE[ln["kind"]]["color"], fontsize=8))
        if self.session.current:
            pts = np.asarray(self.session.current)
            self.artists += self.ax.plot(pts[:, 0], pts[:, 1], color="yellow", marker="x",
                                         lw=1.0)
        n_thick = sum(ln["kind"] == "thick" for ln in self.session.lines)
        self.ax.set_title(f"{self.title}\n[{self.backgrounds[self.bg_index][0]}] "
                          f"{len(self.session.lines)} lines ({n_thick} thick) - 'h' help  "
                          f"{self.status}", fontsize=9)
        self.fig.canvas.draw_idle()

    # --- events -------------------------------------------------------------------------
    def on_click(self, event) -> None:
        """Left click adds a point to the current line, right click removes the last one."""
        if event.inaxes is not self.ax or self._toolbar_active() or event.xdata is None:
            return
        if event.button == 1:
            self.session.add_point(event.xdata, event.ydata)
        elif event.button == 3:
            self.session.remove_point()
        self.status = ""
        self.redraw()

    def on_key(self, event) -> None:
        """Keyboard commands (finish / undo / delete a line, scroll, background, save; see HELP)."""
        key = event.key
        s = self.session
        if key in ("t", "n"):
            line = s.finish("thick" if key == "t" else "thin")
            self.status = f"line {line['id']} {line['kind']}" if line else "need >= 2 points"
        elif key == "u":
            self.status = f"undo {s.undo()}"
        elif key == "d" and event.xdata is not None:
            ln = s.delete_nearest(event.xdata, event.ydata, self.dx, 1.0)
            self.status = f"deleted line {ln['id']}" if ln else ""
        elif key in ("up", "down", "pageup", "pagedown"):
            step = self.window_s * (0.5 if key in ("up", "down") else 1.0)
            self.view_t0 += -step if key in ("up", "pageup") else step
            self.full_view = False
            self._apply_view()
        elif key == "a":
            self.full_view = not self.full_view
            self._apply_view()
        elif key == "b":
            self.bg_index = (self.bg_index + 1) % len(self.backgrounds)
            img = self.backgrounds[self.bg_index][1]
            self.im.set_data(img)
            self.im.set_clim(**self._clim(img))
        elif key in ("s", "q"):
            self.autosave(force=True)
            if key == "q":
                self.plt.close(self.fig)
                return
        elif key == "h":
            print(HELP)
        else:
            return
        if key in ("t", "n", "u", "d"):                 # finished lines changed: save now
            self.autosave()
        self.redraw()

    def on_close(self, event=None) -> None:
        """Closing the window (also with the window's X button) saves the labels."""
        self.autosave()
        if self.session.current:
            print(f"warning: {len(self.session.current)} points of an unfinished line "
                  "were not saved (finish a line with t / n)")
