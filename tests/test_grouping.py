"""Tests of das.grouping (Stage 7): parametrisation, linking, merging, clean-up."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from das.config import GroupingParams
from das.detection import LineSegment
from das.grouping import (DetectedObject, _segment_arrays, build_objects, connected_components,
                          group_segments, interval_union_length, link_matrix, merge_group,
                          remove_duplicates, to_physical, weighted_line_fit)

ROW_DT, COL_DX, CH_DX = 0.04, 5.1065 / 8, 5.1065
X_REF = 0.5 * 416 * COL_DX


def seg_from_physical(v_kmh: float, t_ref: float, x0: float, x1: float, score: float = 5.0,
                      thick_px: float = 20.0) -> LineSegment:
    """Segment of the line t(x) = t_ref + (x - X_REF) / v between x0 and x1 [m]."""
    t = lambda x: t_ref + (x - X_REF) / (v_kmh / 3.6)  # noqa: E731
    col = lambda x: x / COL_DX - 0.5                    # noqa: E731
    row = lambda tt: tt / ROW_DT - 0.5                  # noqa: E731
    return LineSegment(col(x0), row(t(x0)), col(x1), row(t(x1)), score, int(np.sign(v_kmh)),
                       thick_px)


def test_to_physical_pixel_centres():
    x, t = to_physical(np.array([[0.0, 0.0], [9.5, 24.5]]), ROW_DT, COL_DX)
    assert x == pytest.approx([0.5 * COL_DX, 10 * COL_DX])
    assert t == pytest.approx([0.02, 1.0])


@pytest.mark.parametrize("v", [-50.0, 30.0, 120.0])
def test_segment_parametrisation_independent_of_fragment(v):
    a = _segment_arrays([seg_from_physical(v, 40.0, 10, 60), seg_from_physical(v, 40.0, 180, 250)],
                        ROW_DT, COL_DX)
    t_ref = a["t_c"] + a["p"] * (X_REF - a["x_c"])     # passage time at the reference position
    assert a["v"] == pytest.approx([v, v], rel=1e-6)
    assert t_ref == pytest.approx([40.0, 40.0], abs=1e-6)


def test_connected_components_labels():
    adj = np.zeros((6, 6), bool)
    adj[0, 2] = adj[2, 4] = True          # chain 0-2-4 (asymmetric input on purpose)
    adj[1, 5] = True
    lab = connected_components(adj)
    assert lab[0] == lab[2] == lab[4] and lab[1] == lab[5]
    assert len(set(lab)) == 3 and set(lab) == {0, 1, 2}
    assert connected_components(np.zeros((0, 0), bool)).size == 0


def test_fragments_of_one_line_form_one_group():
    p = GroupingParams()
    segs = [seg_from_physical(-50, 30.0, 10, 80),                 # fragments of one line
            seg_from_physical(-51, 30.1, 120, 200),
            seg_from_physical(-49, 29.9, 150, 260),
            seg_from_physical(-50, 34.0, 20, 250),                # parallel vehicle, 4 s later
            seg_from_physical(50, 30.0, 20, 250)]                 # opposite direction, crossing
    sizes = sorted(len(g) for g in group_segments(segs, p, ROW_DT, COL_DX))
    assert sizes == [1, 1, 3]


def test_tolerance_widened_by_line_thickness():
    p = GroupingParams()
    thick = seg_from_physical(-40, 50.0, 0, 260, thick_px=60)     # 2.4 s thick
    edge = seg_from_physical(-40, 51.1, 50, 200, thick_px=0)      # within thick / 2
    far = seg_from_physical(-40, 52.0, 50, 200, thick_px=0)       # beyond thick / 2
    adj = link_matrix([thick, edge, far], p, ROW_DT, COL_DX)
    assert adj[0, 1] and not adj[0, 2]


def test_continuation_links_pieces_of_a_bent_trajectory():
    """A braking vehicle: 60 km/h, then 35 km/h - the pieces meet at x = 150 m."""
    p = GroupingParams()
    a = seg_from_physical(-60, 0.0, 260, 150)
    t_meet = 0.0 + (150 - X_REF) / (-60 / 3.6)
    b = seg_from_physical(-35, t_meet - (150 - X_REF) / (-35 / 3.6), 145, 40)
    assert link_matrix([a, b], p, ROW_DT, COL_DX)[0, 1]
    # without the continuation criterion they are not linked (velocities differ)
    assert not link_matrix([a, b], replace(p, cont_max_gap_m=-1e9), ROW_DT, COL_DX)[0, 1]


def test_weighted_line_fit_rejects_outliers():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 250, 200)
    t = 10 + x / 15.0 + rng.normal(0, 0.02, x.size)
    t[::20] += 5.0                                     # outliers (crossing line)
    p, xc, tc, keep = weighted_line_fit(x, t, np.ones_like(x), iterations=3)
    assert 1 / p == pytest.approx(15.0, rel=0.01)
    assert not keep[::20].any() and keep.sum() > 170


def test_merge_group_average_line_and_score_weight():
    a = seg_from_physical(-50, 30.0, 10, 250, score=5)
    b = seg_from_physical(-55, 30.0, 10, 250, score=5)
    eq = merge_group([a, b], 0, ROW_DT, COL_DX, X_REF)
    wb = merge_group([a, replace(b, score=20)], 0, ROW_DT, COL_DX, X_REF)
    assert -55 < eq.velocity_kmh < -50
    assert abs(wb.velocity_kmh + 55) < abs(eq.velocity_kmh + 55)     # stronger candidate wins
    assert eq.t_ref_s == pytest.approx(30.0, abs=0.05) and eq.direction == -1
    # entry first in time: for v < 0 the object enters at the high-x end
    assert eq.t_start_s < eq.t_end_s and eq.x_start_m > eq.x_end_m
    assert eq.x_min == pytest.approx(10, abs=1) and eq.x_max == pytest.approx(250, abs=1)
    assert eq.support_m == pytest.approx(240, abs=1)
    assert eq.thickness_s == pytest.approx(20 * ROW_DT)
    assert float(eq.t_at(X_REF)) == pytest.approx(eq.t_ref_s)


def _obj(v, t_ref, x0, x1, score=5.0):
    return DetectedObject(0, v, t_ref, x0, x1, 0.0, 0.0, score, direction=int(np.sign(v)),
                          x_ref_m=X_REF, support_m=abs(x1 - x0))


def test_remove_duplicates_and_implausible_objects():
    p = GroupingParams()
    objs = [_obj(-50, 30.0, 0, 260, 8.0),
            _obj(-58, 30.3, 50, 200, 6.0),     # same vehicle, smaller support x score
            _obj(-50, 40.0, 0, 20, 20.0),      # too short (< 8 channels)
            _obj(-200, 50.0, 0, 260, 20.0),    # too fast
            _obj(50, 30.0, 0, 260, 8.0)]       # other direction: not a duplicate
    kept = remove_duplicates(objs, p, CH_DX)
    assert sorted((o.velocity_kmh, o.t_ref_s) for o in kept) == [(-50, 30.0), (50, 30.0)]


def test_build_objects_renumbers_by_time():
    p = GroupingParams()
    segs = [seg_from_physical(-50, 80.0, 0, 260), seg_from_physical(60, 20.0, 0, 260),
            seg_from_physical(-50, 50.0, 0, 120), seg_from_physical(-50, 50.1, 130, 260)]
    objs, groups = build_objects(segs, p, ROW_DT, COL_DX, X_REF, CH_DX)
    assert len(groups) == 3
    assert [o.obj_id for o in objs] == [0, 1, 2]
    assert [round(o.t_ref_s) for o in objs] == [20, 50, 80]
    assert [round(o.velocity_kmh) for o in objs] == [60, -50, -50]


def test_interval_union_length():
    lo, hi = np.array([50.0, 0.0, 10.0, 100.0]), np.array([60.0, 20.0, 30.0, 101.0])
    assert interval_union_length(lo, hi) == pytest.approx(30 + 10 + 1)
    assert interval_union_length(np.array([]), np.array([])) == 0.0
