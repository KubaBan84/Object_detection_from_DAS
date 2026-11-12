"""Tests of das.annotation (Stage 11): label storage, session logic and the GUI handlers
(driven by synthetic events on the Agg backend - no display needed)."""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from das.annotation import (AnnotationSession, Annotator, load_labels,  # noqa: E402
                            save_labels, segment_lines)
from das.evaluation import references_from_annotations  # noqa: E402


def test_session_add_finish_undo_delete():
    s = AnnotationSession()
    s.add_point(10, 5.0)
    assert s.finish("thin") is None                           # < 2 points
    s.add_point(10, 6.0)
    assert s.finish("thin") is None                           # vertical line: no velocity
    s.current = [[10, 5.0], [200, 15.0]]
    line = s.finish("thick")
    assert line["id"] == 1 and line["kind"] == "thick" and s.current == []
    s.add_point(30, 40.0)
    s.add_point(150, 45.0)
    assert s.finish("thin")["id"] == 2
    with pytest.raises(ValueError):
        s.finish("medium")
    s.add_point(1, 1)
    assert s.undo() == "point" and s.current == []
    assert s.delete_nearest(100, 44.0, 5.0, 1.0)["id"] == 2  # the thin line is closer
    assert s.undo() == "line" and s.lines == [] and s.undo() == ""
    assert s.delete_nearest(0, 0) is None


def test_labels_round_trip_and_references(tmp_path):
    path = tmp_path / "labels.json"
    assert load_labels(path) == {"segments": {}}
    s = AnnotationSession([{"id": 4, "kind": "thin", "points": [[0, 10], [100, 15]]}])
    s.current = [[20, 30.0], [220, 40.0]]
    s.finish("thick")
    data = {"segments": {"seg_x": s.to_dict("090452", "tester", "raw")}}
    save_labels(data, path)
    back = load_labels(path)
    lines = segment_lines(back, "seg_x")
    assert [ln["id"] for ln in lines] == [4, 5] and back["segments"]["seg_x"]["annotator"] == "tester"
    assert segment_lines(back, "missing") == []
    refs = references_from_annotations(lines)
    assert refs[0]["velocity_kmh"] == pytest.approx(72.0) and refs[1]["type"] == "thick"


def _event(ax, x=None, y=None, button=1, key=None):
    return SimpleNamespace(inaxes=ax, xdata=x, ydata=y, button=button, key=key)


def test_annotator_handlers_with_synthetic_events():
    rng = np.random.default_rng(0)
    bgs = [("raw", rng.normal(size=(3000, 52))), ("HF", rng.normal(size=(3000, 52)))]
    saved = []
    session = AnnotationSession()
    gui = Annotator(bgs, 0.04, 5.1065, None, session, lambda s: saved.append(len(s.lines)) or "x.json",
                    title="test", window_s=30.0)
    ax = gui.ax
    assert ax.get_ylim() == (30.0, 0.0)                       # window view
    gui.on_click(_event(ax, 20.0, 5.0))
    gui.on_click(_event(ax, 150.0, 12.0))
    gui.on_click(_event(None, 1.0, 1.0))                      # outside the axes: ignored
    gui.on_key(_event(ax, key="t"))
    assert len(session.lines) == 1 and session.lines[0]["kind"] == "thick"
    assert saved == [1]                                       # autosave after a finished line
    gui.on_click(_event(ax, 30.0, 20.0))
    gui.on_click(_event(ax, 30.0, 20.0, button=3))            # right click removes the point
    assert session.current == []
    gui.on_key(_event(ax, key="pagedown"))
    assert ax.get_ylim() == (60.0, 30.0)
    gui.on_key(_event(ax, key="a"))
    assert ax.get_ylim() == (120.0, 0.0)                      # whole segment
    gui.on_key(_event(ax, key="b"))
    assert gui.bg_index == 1
    gui.on_key(_event(ax, key="s"))                           # explicit save
    assert saved == [1, 1] and "saved" in gui.status
    gui.on_key(_event(ax, 100.0, 9.0, key="d"))
    assert session.lines == [] and saved == [1, 1, 0]          # autosave after delete
    gui.on_key(_event(ax, key="q"))                           # saves and closes (no re-save on close)
    assert saved == [1, 1, 0, 0] and not plt.fignum_exists(gui.fig.number)


def test_unassigned_annotator_still_receives_canvas_events():
    """Regression: mpl_connect keeps weak references - the tool must stay alive even when
    the caller does not keep it (as in scripts/annotate_real.py), else clicks do nothing."""
    import gc

    from matplotlib.backend_bases import KeyEvent, MouseEvent

    session = AnnotationSession()
    bgs = [("raw", np.random.default_rng(1).normal(size=(3000, 52)))]
    Annotator(bgs, 0.04, 5.1065, None, session, lambda s: "x.json", window_s=30.0)
    gc.collect()
    fig = plt.gcf()
    fig.canvas.draw()
    ax = fig.axes[0]
    for x, t in [(20.0, 5.0), (150.0, 12.0)]:
        px, py = ax.transData.transform((x, t))
        fig.canvas.callbacks.process("button_press_event",
                                     MouseEvent("button_press_event", fig.canvas, px, py, button=1))
    assert len(session.current) == 2
    fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, "n", px, py))
    assert len(session.lines) == 1 and session.lines[0]["kind"] == "thin"
    plt.close(fig)


def test_closing_the_window_saves_the_labels(tmp_path):
    """Closing with the window's X button (close_event) must not lose the lines."""
    path = tmp_path / "labels.json"
    session = AnnotationSession()
    bgs = [("raw", np.random.default_rng(2).normal(size=(3000, 52)))]
    gui = Annotator(bgs, 0.04, 5.1065, None, session,
                    lambda s: save_labels({"segments": {"seg": s.to_dict("0", "me", "raw")}}, path),
                    window_s=30.0)
    session.add_point(10.0, 1.0)                              # line changed without t / n hooks
    session.add_point(80.0, 6.0)
    session.finish("thick")
    gui.fig.canvas.callbacks.process("close_event", SimpleNamespace(name="close_event"))
    assert len(segment_lines(load_labels(path), "seg")) == 1
    plt.close(gui.fig)
