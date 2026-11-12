"""Tests of das.tracking (Stage 9): ridge search, RANSAC, v(t), speed variability, chain."""
from __future__ import annotations

import numpy as np
import pytest

from das.config import DetectionParams, EnhancementParams, GroupingParams, TrackingParams
from das.detection import detect_candidates
from das.enhancement import Enhanced, enhance
from das.grouping import DetectedObject, build_objects
from das.tracking import (eval_trajectory, fit_trajectory_ransac, longest_run, ridge_times,
                          speed_variability, track_objects, track_ridge, velocity_profile)

ROW_DT, DX = 0.04, 5.1065


def braking_track(v0_kmh=70.0, a=-0.6, n_ch=52, t0=5.0):
    """Exact passage times of a vehicle braking from v0 with deceleration a (x(t) quadratic).

    Defaults: 70 km/h -> 28 km/h over the 265 m of cable (it does not stop).
    """
    v0 = v0_kmh / 3.6
    x = (np.arange(n_ch) + 0.5) * DX
    t = t0 + (-v0 + np.sqrt(v0 ** 2 + 2 * a * x)) / a       # solve x = v0 t + a t^2 / 2
    return x, t


# ---------------------------------------------------------------- ridge search
@pytest.mark.parametrize("method", ["parabola", "centroid"])
def test_ridge_times_subrow_precision_and_edges(method):
    n_rows, n_cols = 400, 5
    true_t = np.array([3.013, 5.5, 7.981, 10.04, 15.77])      # [s]
    r = np.arange(n_rows)[:, None]
    img = np.exp(-0.5 * (((r + 0.5) * ROW_DT - true_t[None, :]) / 0.15) ** 2)
    t, peak, ok = ridge_times(img, np.arange(n_cols), true_t + 0.2, 1.0, ROW_DT, method)
    assert ok.all() and np.allclose(peak, 1.0, atol=0.02)
    assert np.allclose(t, true_t, atol=0.25 * ROW_DT)
    # window entirely outside the image / peak on the window edge -> invalid, no warnings
    t2, _, ok2 = ridge_times(img, np.arange(2), np.array([-50.0, 3.013 + 1.3]), 1.0, ROW_DT, method)
    assert not ok2.any()


def test_longest_run():
    cols = np.array([0, 1, 2, 10, 11, 12, 13, 15, 30])
    assert longest_run(cols, 1).tolist() == [False] * 3 + [True] * 5 + [False]
    assert longest_run(np.array([], int), 2).size == 0


# ---------------------------------------------------------------- trajectory RANSAC
def test_ransac_polynomial_rejects_outliers():
    x, t = braking_track(70, -0.3)                               # t(x) close to quadratic
    t_noisy = t + np.random.default_rng(0).normal(0, 0.02, t.size)
    t_noisy[[5, 20, 33]] += [2.0, -1.5, 3.0]                     # crossing vehicle / noise
    model, inl = fit_trajectory_ransac(np.column_stack([x, t_noisy]), 0.2, order=2, seed=1)
    assert not inl[[5, 20, 33]].any() and inl.sum() >= x.size - 4
    assert np.max(np.abs(eval_trajectory(model, x) - t)) < 0.1


def test_ransac_degenerate_inputs():
    tr = np.array([[0.0, 1.0], [5.0, 1.5]])
    model, inl = fit_trajectory_ransac(tr, 0.2, order=2)
    assert inl.all() and np.allclose(eval_trajectory(model, tr[:, 0]), tr[:, 1])


# ---------------------------------------------------------------- velocity profile
def test_velocity_profile_constant_and_braking():
    x = (np.arange(40) + 0.5) * DX
    t, v = velocity_profile(np.column_stack([x, 2.0 + x / (50 / 3.6)]), 7)
    assert np.allclose(v, 50.0) and np.all(np.diff(t) > 0)
    tn, vn = velocity_profile(np.column_stack([x[::-1], -(x / (50 / 3.6))[::-1]]), 7)
    assert np.allclose(vn, -50.0)                               # sign = direction
    xb, tb = braking_track(70, -0.6)
    tp, vp = velocity_profile(np.column_stack([xb, tb]), 5)
    v_true = 70 / 3.6 + (-0.6) * (tp - 5.0)
    assert np.max(np.abs(vp - 3.6 * v_true)) < 1.0              # km/h
    assert vp[0] > vp[-1]


def test_speed_variability_labels():
    p = TrackingParams()
    x = (np.arange(52) + 0.5) * DX
    rng = np.random.default_rng(2)
    const = np.column_stack([x, 1.0 + x / (60 / 3.6) + rng.normal(0, 0.03, x.size)])
    s = speed_variability(const, None, p)
    assert s["speed_class"] == "constant" and abs(s["accel_ms2"]) < 0.3
    xb, tb = braking_track(70, -0.6)
    s = speed_variability(np.column_stack([xb, tb + rng.normal(0, 0.03, xb.size)]), None, p)
    assert s["speed_class"] == "varying"
    assert s["accel_ms2"] == pytest.approx(-0.6, abs=0.15)
    assert s["v_start_kmh"] == pytest.approx(70, abs=3) and s["v_end_kmh"] < 50
    assert speed_variability(const[:3], None, p)["speed_class"] == ""
    # the same braking towards decreasing x: acceleration is reported along the motion
    s_neg = speed_variability(np.column_stack([-xb, tb]), None, p)
    assert s_neg["accel_ms2"] == pytest.approx(-0.6, abs=0.05)          # noiseless track
    assert s_neg["v_start_kmh"] < 0 and abs(s_neg["v_end_kmh"]) < abs(s_neg["v_start_kmh"])


# ---------------------------------------------------------------- track of one object
def _fake_enhanced(fk_pos: np.ndarray) -> dict:
    up = 8
    e = Enhanced(row_dt=ROW_DT, col_dx=DX / up,
                 response={1: np.zeros((fk_pos.shape[0], fk_pos.shape[1] * up)),
                           -1: np.zeros((fk_pos.shape[0], fk_pos.shape[1] * up))},
                 stages={"fk_pos": fk_pos, "fk_neg": np.zeros_like(fk_pos)})
    return {"hf": e}


def test_track_follows_a_curved_trajectory_from_a_straight_line():
    """The Stage-7 line is straight; the track must follow the true (braking) curve."""
    xb, tb = braking_track(70, -0.6, t0=5.0)
    rows = np.arange(1200)[:, None]
    rng = np.random.default_rng(3)
    img = 6 * np.exp(-0.5 * (((rows + 0.5) * ROW_DT - tb[None, :]) / 0.2) ** 2) + \
        rng.normal(0, 1, (1200, xb.size))
    p = np.polyfit(xb, tb, 1)                                    # straight averaged line
    obj = DetectedObject(0, 3.6 / p[0], float(np.polyval(p, 130.0)), xb[0], xb[-1],
                         tb[0], tb[-1], "strong", 10.0, direction=1, x_ref_m=130.0)
    tr = track_ridge(_fake_enhanced(img)["hf"].stages["fk_pos"], obj, TrackingParams(),
                     ROW_DT, DX)
    assert len(tr) > 40
    assert np.median(np.abs(tr[:, 1] - np.interp(tr[:, 0], xb, tb))) < 0.05
    track_objects([obj], _fake_enhanced(img), TrackingParams(track_band="hf", smooth_rows=0),
                  ROW_DT)
    assert obj.speed_class == "varying"
    assert obj.accel_ms2 == pytest.approx(-0.6, abs=0.2)
    assert obj.r2 > 0.98 and obj.profile.shape[1] == 3
    assert obj.v_start_kmh > obj.v_end_kmh


def test_edge_margin_drops_peaks_biased_by_the_image_end():
    """A vehicle leaving through the last image row: for the channels whose ridge centre
    lies beyond the image the noisy, flat ridge flank has its maximum a few rows before
    the border (compressed passage times -> exploding v(t)). With the margin these peaks
    are dropped and v(t) stays close to the true 60 km/h."""
    n_rows = 600                                                 # 24 s image
    x = (np.arange(52) + 0.5) * DX
    v = 60 / 3.6
    tb = 18.0 + x / v                                            # ridge leaves the image at x ~ 100 m
    rows = np.arange(n_rows)[:, None]
    img = 6 * np.exp(-0.5 * (((rows + 0.5) * ROW_DT - tb[None, :]) / 0.4) ** 2) +         np.random.default_rng(5).normal(0, 0.3, (n_rows, x.size))
    t_end = n_rows * ROW_DT
    res = {}
    for margin in (0.0, 0.5):
        o = DetectedObject(0, 60.0, float(18 + 130 / v), x[0], x[-1], tb[0], tb[-1], "strong",
                           10.0, direction=1, x_ref_m=130.0)
        track_objects([o], _fake_enhanced(img), TrackingParams(track_band="hf", smooth_rows=0,
                                                               edge_margin_s=margin), ROW_DT)
        res[margin] = o
    with_m = res[0.5]
    assert len(with_m.track) >= 10
    assert with_m.track[:, 1].max() <= t_end - 0.5
    assert np.nanmax(np.abs(with_m.profile[:, 2])) < 70
    assert with_m.v_track_kmh == pytest.approx(60, rel=0.08)
    assert res[0.0].track[:, 1].max() > t_end - 0.5              # the biased border peaks


def test_no_track_on_pure_noise_keeps_line_velocity():
    rng = np.random.default_rng(4)
    img = rng.normal(0, 1, (1200, 52))
    obj = DetectedObject(0, 50.0, 20.0, 10.0, 250.0, 0.0, 0.0, "weak", 3.0, direction=1,
                         x_ref_m=130.0)
    track_objects([obj], _fake_enhanced(img), TrackingParams(track_band="hf", min_peak_z=3.5),
                  ROW_DT)
    assert len(obj.track) == 0 and np.isnan(obj.v_track_kmh) and obj.speed_class == ""


# ---------------------------------------------------------------- full chain Stage 4 image -> v(t)
def test_chain_detects_and_tracks_a_braking_vehicle():
    """Stage-4-like image with a braking car -> enhance -> detect -> group -> track."""
    xb, tb = braking_track(80, -0.6, t0=8.0)
    rng = np.random.default_rng(5)
    rows = np.arange(1500)[:, None]
    img = np.abs(rng.normal(0, 1, (1500, xb.size))) + \
        3.0 * np.exp(-0.5 * (((rows + 0.5) * ROW_DT - tb[None, :]) / 0.3) ** 2)
    ep = EnhancementParams()
    enh = {b: enhance(img, ep, ROW_DT, DX) for b in ("hf", "lf")}
    det = detect_candidates(enh, DetectionParams(), (ep.v_min_kmh, ep.v_max_kmh))
    col_dx = enh["hf"].col_dx
    objs, _ = build_objects(det.segments, GroupingParams(), ROW_DT, col_dx, 130.0, DX)
    o = max(objs, key=lambda ob: ob.score)                       # the vehicle (strong line)
    assert o.kind == "strong" and o.x_max - o.x_min > 200
    track_objects(objs, enh, TrackingParams(), ROW_DT)
    v_mean = (xb[-1] - xb[0]) / (tb[-1] - tb[0]) * 3.6
    assert o.v_track_kmh == pytest.approx(v_mean, rel=0.05)
    assert o.speed_class == "varying" and o.accel_ms2 == pytest.approx(-0.6, abs=0.25)
