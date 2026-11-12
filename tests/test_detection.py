"""Tests of das.detection (Stage 6): helpers, strong path 6A, weak path 6B, full chain."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from das.config import DetectionParams, EnhancementParams, GroupingParams
from das.detection import (LineSegment, detect_candidates, detect_strong_lines,
                           detect_weak_lines, hough_global, line_structuring_elements,
                           mask_strong, oriented_opening, ridge_along_time, robust_noise_sigma,
                           sequential_ransac, slope_bounds, slope_ok, velocity_to_slope)
from das.enhancement import enhance
from das.grouping import build_objects

ROW_DT, COL_DX = 0.04, 5.1065 / 8          # Stage-5 response grid
V_RANGE = (5.0, 150.0)


def _line_image(shape, v_kmh, r0, width_rows, amp, c_start=0, c_end=None):
    """Gaussian ridge (along time) of a line r(c) = r0 + c / slope, slope from v."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    slope = velocity_to_slope(v_kmh / 3.6, ROW_DT, COL_DX)
    r_line = r0 + cols / slope
    img = amp * np.exp(-0.5 * ((rows - r_line) / width_rows) ** 2)
    c_end = shape[1] if c_end is None else c_end
    img[:, :c_start] = 0
    img[:, c_end:] = 0
    return img


# ---------------------------------------------------------------- helpers
def test_robust_noise_sigma_ignores_lines_and_respects_mask():
    rng = np.random.default_rng(0)
    img = rng.normal(0, 2.0, (400, 400))
    img[:, 200:204] += 100.0                       # strong outliers (a "line")
    assert robust_noise_sigma(img) == pytest.approx(2.0, rel=0.05)
    mask = np.zeros_like(img, bool)
    mask[:, :100] = True
    assert robust_noise_sigma(img * 3, mask) == pytest.approx(6.0, rel=0.05)
    assert robust_noise_sigma(np.zeros((5, 5))) == 1.0      # MAD = 0 handled


def test_slope_bounds_sign_and_values():
    lo, hi = slope_bounds(V_RANGE, +1, ROW_DT, COL_DX)
    assert 0 < lo < hi
    assert lo == pytest.approx(5 / 3.6 * ROW_DT / COL_DX)
    nlo, nhi = slope_bounds(V_RANGE, -1, ROW_DT, COL_DX)
    assert (nlo, nhi) == pytest.approx((-hi, -lo))
    assert slope_ok(1.0, (lo, hi), 0.1) and not slope_ok(-1.0, (lo, hi), 0.1)
    assert not slope_ok(hi * 1.5, (lo, hi), 0.1)


def test_line_structuring_elements_length_and_orientation():
    for s in (0.3, 1.0, -2.0):
        se = line_structuring_elements(41, np.array([s]))[0]
        r, c = np.nonzero(se)
        assert np.hypot(np.ptp(r), np.ptp(c)) == pytest.approx(40, abs=1.5)
        fit = np.polyfit(r - r.mean(), c - c.mean(), 1)[0]
        assert fit == pytest.approx(s, rel=0.1)


def test_oriented_opening_keeps_slanted_line_removes_specks_and_streaks():
    mask = _line_image((300, 300), 50, 20, 2, 1.0) > 0.5
    rng = np.random.default_rng(1)
    specks = rng.random((300, 300)) > 0.995
    streak = np.zeros_like(mask)
    streak[200:203, 0:120] = True                   # horizontal streak (simultaneous event)
    vertical = np.zeros_like(mask)
    vertical[:, 280] = True                         # static channel (v = 0)
    lo, hi = slope_bounds(V_RANGE, +1, ROW_DT, COL_DX)
    out = oriented_opening(mask | specks | streak | vertical, 31, np.linspace(lo, hi, 9))
    assert out[mask].mean() > 0.9
    assert out[specks & ~mask].mean() < 0.05
    assert out[streak & ~mask].mean() < 0.05
    assert out[vertical & ~mask].mean() < 0.05


def test_ridge_separates_touching_parallel_lines():
    img = _line_image((400, 200), -60, 300, 6, 10) + _line_image((400, 200), -60, 330, 6, 10)
    img += np.random.default_rng(5).normal(0, 0.3, img.shape)   # bumpy flat tops
    ridge = ridge_along_time(img, img > 2.0, nms_rows=10)   # one connected band in the mask
    per_col = ridge[:, 20:180].sum(axis=0)
    assert np.all(per_col == 2)


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


# ---------------------------------------------------------------- 6A strong path
def test_strong_path_two_touching_thick_lines():
    shape = (1200, 416)
    rng = np.random.default_rng(2)
    resp = rng.normal(0, 1, shape) + _line_image(shape, -50, 700, 8, 25) + \
        _line_image(shape, -50, 740, 8, 25)
    segs, mask, info = detect_strong_lines(resp, DetectionParams(), ROW_DT, COL_DX, -1, V_RANGE)
    assert len(segs) == 2
    for s in segs:
        v = s.slope * COL_DX / ROW_DT * 3.6
        assert v == pytest.approx(-50, rel=0.05)
        assert s.source == "strong" and s.direction == -1
        assert s.thickness_px > 10
    assert mask.sum() > 0 and info["threshold"] >= 4.0
    # no strong line in pure noise (threshold floor) and none of the wrong direction
    assert detect_strong_lines(rng.normal(0, 1, shape), DetectionParams(), ROW_DT, COL_DX, -1,
                               V_RANGE)[0] == []
    assert detect_strong_lines(resp, DetectionParams(), ROW_DT, COL_DX, +1, V_RANGE)[0] == []


# ---------------------------------------------------------------- 6B weak path
def test_mask_strong_excludes_dilated_area():
    resp = np.ones((50, 50)) * 5
    resp[:25] = 0.0
    sm = np.zeros((50, 50), bool)
    sm[30, 30] = True
    out, excl = mask_strong(resp, sm, (3, 2))
    assert excl[27:34, 30].all() and excl[30, 28:33].all()
    assert not excl[26, 30] and not excl[30, 33] and excl.sum() < 7 * 5
    assert np.all(out[excl] == np.median(resp[~excl]))


def test_weak_path_finds_thin_line_and_rejects_wrong_direction():
    shape = (1000, 416)
    rng = np.random.default_rng(3)
    resp = rng.normal(0, 1, shape) + _line_image(shape, 70, 100, 1.5, 5)
    p = DetectionParams()
    segs, info = detect_weak_lines(resp, np.zeros(shape, bool), p, ROW_DT, COL_DX, +1, V_RANGE)
    assert segs, "thin line not found"
    vs = np.array([s.slope * COL_DX / ROW_DT * 3.6 for s in segs])
    assert np.all(np.abs(vs - 70) < 10)
    assert all(s.score > info["threshold_low"] for s in segs)
    neg, _ = detect_weak_lines(resp, np.zeros(shape, bool), p, ROW_DT, COL_DX, -1, V_RANGE)
    assert neg == []


def test_hough_global_bridges_gaps_of_a_dashed_line():
    shape = (800, 416)
    skel = np.zeros(shape, bool)
    s = velocity_to_slope(60 / 3.6, ROW_DT, COL_DX)
    r = np.arange(50, 430)
    on = (r // 40) % 2 == 0                          # 40 px dashes, 40 px gaps
    skel[r[on], np.rint((r[on] - 50) * s).astype(int)] = True
    p = DetectionParams()
    bounds = slope_bounds(V_RANGE, +1, ROW_DT, COL_DX)
    segs = hough_global(skel, skel.astype(float), bounds, +1, p, "lf")
    assert len(segs) == 1 and segs[0].source == "hough_global"
    assert segs[0].r1 - segs[0].r0 > 330
    assert segs[0].slope == pytest.approx(s, rel=0.03)


def test_line_segment_orders_endpoints_in_time():
    seg = LineSegment(10, 50, 0, 10, 1.0, "weak", +1)
    assert (seg.r0, seg.r1, seg.c0, seg.c1) == (10, 50, 0, 10)
    assert seg.slope == pytest.approx(0.25)
    assert seg.points().shape[1] == 2


# ---------------------------------------------------------------- full Stage 4-image -> objects
@pytest.fixture(scope="module")
def synthetic_scene():
    """Stage-4-like image (0.04 s x 5.1 m): one thick line, two thin lines (SNR ~1.5), noise."""
    rng = np.random.default_rng(4)
    n_rows, n_ch, dx = 1500, 52, 5.1065
    t = np.arange(n_rows)[:, None] * ROW_DT
    x = np.arange(n_ch)[None, :] * dx
    def line(v_kmh, t0, sigma_t, amp):             # passage time t(x) = t0 + x / v
        return amp * np.exp(-0.5 * ((t - t0 - x / (v_kmh / 3.6)) / sigma_t) ** 2)
    img = np.abs(rng.normal(0, 1, (n_rows, n_ch)))
    img += line(-50, 40, 0.8, 8.0)                 # thick, towards decreasing x
    img += line(70, 10, 0.12, 1.5)                 # thin, towards increasing x
    img += line(-90, 48, 0.12, 1.5)                # thin, same direction as the thick one
    return img, dx


def test_thick_and_thin_lines_detected_with_correct_direction(synthetic_scene):
    img, dx = synthetic_scene
    ep = EnhancementParams()
    enh = {b: enhance(img, ep, ROW_DT, dx) for b in ("hf", "lf")}
    det = detect_candidates(enh, DetectionParams(), (ep.v_min_kmh, ep.v_max_kmh))
    width = enh["hf"].response[1].shape[1] * enh["hf"].col_dx
    objs, _ = build_objects(det.segments, GroupingParams(), ROW_DT, enh["hf"].col_dx,
                            0.5 * width, dx)
    assert len(objs) == 3, [(o.kind, round(o.velocity_kmh, 1)) for o in objs]
    thick = min(objs, key=lambda o: abs(o.velocity_kmh + 50))
    pos = [o for o in objs if o.velocity_kmh > 0]
    thin_neg = min(objs, key=lambda o: abs(o.velocity_kmh + 90))
    assert thick.kind == "strong" and thick.velocity_kmh == pytest.approx(-50, rel=0.05)
    # in a direction without heavy traffic the strong threshold falls to its floor, so a
    # clear thin line may come from either path - only direction and velocity matter here
    assert len(pos) == 1 and pos[0].velocity_kmh == pytest.approx(70, rel=0.1)
    # the thin line of the busy direction is found by path 6B (strong objects masked)
    assert thin_neg.velocity_kmh == pytest.approx(-90, rel=0.1)
    assert thin_neg.kind == "weak" and set(thin_neg.sources) <= {"weak", "hough_global"}
    assert det.stages["hf_neg_strong_mask"].any()


def test_strong_path_only_misses_thin_lines(synthetic_scene):
    """Ablation: grade 3/4 configuration (no weak path) finds only the thick line."""
    img, dx = synthetic_scene
    ep = EnhancementParams()
    enh = {b: enhance(img, ep, ROW_DT, dx) for b in ("hf", "lf")}
    det = detect_candidates(enh, replace(DetectionParams(), weak_bands=()),
                            (ep.v_min_kmh, ep.v_max_kmh))
    assert {s.source for s in det.segments} == {"strong"}
