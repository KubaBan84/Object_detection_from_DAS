"""Tests of das.detection (Stage 6): helpers, thick-line detection, full chain."""
from __future__ import annotations

import numpy as np
import pytest

from das.config import DetectionParams, EnhancementParams, GroupingParams
from das.detection import (LineSegment, detect_lines, detect_thick_lines, detection_threshold,
                           ridge_along_time,
                           robust_noise_sigma, sequential_ransac, slope_bounds, slope_ok,
                           velocity_to_slope)
from das.enhancement import enhance
from das.grouping import build_objects

ROW_DT, COL_DX = 0.04, 5.1065 / 8          # Stage-5 response grid
V_RANGE = (5.0, 150.0)


def _line_image(shape, v_kmh, r0, width_rows, amp):
    """Gaussian ridge (along time) of a line r(c) = r0 + c / slope, slope from v."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    slope = velocity_to_slope(v_kmh / 3.6, ROW_DT, COL_DX)
    return amp * np.exp(-0.5 * ((rows - (r0 + cols / slope)) / width_rows) ** 2)


# ---------------------------------------------------------------- helpers
def test_robust_noise_sigma_ignores_lines():
    rng = np.random.default_rng(0)
    img = rng.normal(0, 2.0, (400, 400))
    img[:, 200:204] += 100.0                       # strong outliers (a "line")
    assert robust_noise_sigma(img) == pytest.approx(2.0, rel=0.05)
    assert robust_noise_sigma(np.zeros((5, 5))) == 1.0      # MAD = 0 handled


def test_threshold_is_floored_in_pure_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (500, 300))
    p = DetectionParams()
    thr = detection_threshold(noise, p)
    assert thr == pytest.approx(np.median(noise) + p.min_sigma * robust_noise_sigma(noise))
    fixed = DetectionParams(threshold="6")
    assert detection_threshold(noise, fixed) == pytest.approx(
        np.median(noise) + 6 * robust_noise_sigma(noise))


def test_slope_bounds_sign_and_values():
    lo, hi = slope_bounds(V_RANGE, +1, ROW_DT, COL_DX)
    assert 0 < lo < hi
    assert lo == pytest.approx(5 / 3.6 * ROW_DT / COL_DX)
    nlo, nhi = slope_bounds(V_RANGE, -1, ROW_DT, COL_DX)
    assert (nlo, nhi) == pytest.approx((-hi, -lo))
    assert slope_ok(1.0, (lo, hi), 0.1) and not slope_ok(-1.0, (lo, hi), 0.1)
    assert not slope_ok(hi * 1.5, (lo, hi), 0.1)


def test_velocity_to_slope_known_values_and_sign():
    # 50 km/h = 13.9 m/s towards increasing x -> 13.9 * 0.04 / 0.638 = 0.871 columns per row
    s = velocity_to_slope(50 / 3.6, ROW_DT, COL_DX)
    assert float(s) == pytest.approx(0.8705, rel=1e-3)
    assert float(velocity_to_slope(-50 / 3.6, ROW_DT, COL_DX)) == pytest.approx(-s)


def test_ridge_separates_touching_parallel_lines():
    img = _line_image((400, 200), -60, 300, 6, 10) + _line_image((400, 200), -60, 330, 6, 10)
    img += np.random.default_rng(5).normal(0, 0.3, img.shape)   # bumpy flat tops
    ridge = ridge_along_time(img, img > 2.0, nms_rows=10)   # one connected band in the mask
    assert np.all(ridge[:, 20:180].sum(axis=0) == 2)


def test_sequential_ransac_finds_two_lines():
    c = np.arange(0, 300, dtype=float)
    s = velocity_to_slope(-60 / 3.6, ROW_DT, COL_DX)
    cols = np.concatenate([c, c])
    rows = np.concatenate([600 + c / s, 640 + c / s])
    bounds = slope_bounds(V_RANGE, -1, ROW_DT, COL_DX)
    lines = sequential_ransac(cols, rows, DetectionParams(), min_cols=50, bounds=bounds)
    assert len(lines) == 2
    assert sorted(len(i) for *_, i in lines) == [300, 300]
    for _, d, _ in lines:
        assert d[0] / d[1] == pytest.approx(s)


def test_line_segment_orders_endpoints_in_time():
    seg = LineSegment(10, 50, 0, 10, 1.0, +1)
    assert (seg.r0, seg.r1, seg.c0, seg.c1) == (10, 50, 0, 10)
    assert seg.slope == pytest.approx(0.25)
    assert seg.points().shape[1] == 2


# ---------------------------------------------------------------- thick-line detection
def test_two_touching_thick_lines_give_two_candidates():
    shape = (1200, 416)
    rng = np.random.default_rng(2)
    resp = rng.normal(0, 1, shape) + _line_image(shape, -50, 700, 8, 25) + \
        _line_image(shape, -50, 740, 8, 25)
    segs, stages, thr = detect_thick_lines(resp, DetectionParams(), ROW_DT, COL_DX, -1, V_RANGE)
    assert len(segs) == 2
    for s in segs:
        assert s.slope * COL_DX / ROW_DT * 3.6 == pytest.approx(-50, rel=0.05)
        assert s.direction == -1 and s.thickness_px > 10
    assert stages["mask"].sum() > 0 and thr >= 4.0
    assert set(stages) == {"binary", "ridge", "mask"}
    # nothing in pure noise (threshold floor) and nothing of the wrong direction
    assert detect_thick_lines(rng.normal(0, 1, shape), DetectionParams(), ROW_DT, COL_DX, -1,
                              V_RANGE)[0] == []
    assert detect_thick_lines(resp, DetectionParams(), ROW_DT, COL_DX, +1, V_RANGE)[0] == []


# ---------------------------------------------------------------- Stage-4 image -> objects
def test_thick_lines_detected_with_direction_and_velocity():
    """Stage-4-like image (0.04 s x 5.1 m): two thick lines in opposite directions, noise."""
    rng = np.random.default_rng(4)
    n_rows, n_ch, dx = 1500, 52, 5.1065
    t = np.arange(n_rows)[:, None] * ROW_DT
    x = np.arange(n_ch)[None, :] * dx

    def line(v_kmh, t0, sigma_t, amp):             # passage time t(x) = t0 + x / v
        return amp * np.exp(-0.5 * ((t - t0 - x / (v_kmh / 3.6)) / sigma_t) ** 2)

    img = np.abs(rng.normal(0, 1, (n_rows, n_ch))) + line(-50, 40, 0.8, 8.0) + line(70, 10, 0.6, 6.0)
    ep = EnhancementParams()
    enh = enhance(img, ep, ROW_DT, dx)
    det = detect_lines(enh, DetectionParams(), (ep.v_min_kmh, ep.v_max_kmh))
    assert set(det.thresholds) == {"pos", "neg"} and det.stages["neg_mask"].any()
    objs, _ = build_objects(det.segments, GroupingParams(), ROW_DT, enh.col_dx,
                            0.5 * n_ch * dx, dx)
    assert sorted(round(o.velocity_kmh, -1) for o in objs) == [-50, 70]
    for o in objs:
        assert o.velocity_kmh == pytest.approx(-50 if o.direction < 0 else 70, rel=0.05)
        assert o.x_max - o.x_min > 150
