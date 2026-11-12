"""Tests of das.features (Stage 10): cross-profile width, sub-lines, raw spectrum, and
the features of whole objects on synthetic maps."""
from __future__ import annotations

import numpy as np
import pytest

from das.config import FeatureParams
from das.features import (FEATURE_NAMES, FeatureContext, background_power, count_sublines,
                          cross_section, features_matrix, fwhm_rows, object_features,
                          object_trajectory, section_thumbnail, spectral_features)
from das.grouping import DetectedObject

ROW_DT, DX, FS = 0.04, 5.1065, 625.0


def gauss(n, centre, sigma, amp=1.0):
    return amp * np.exp(-0.5 * ((np.arange(n) - centre) / sigma) ** 2)


# ---------------------------------------------------------------- profile measures
def test_fwhm_of_gaussian_profiles():
    sig = np.array([2.0, 4.0, 7.5])
    p = np.stack([gauss(101, 50, s) for s in sig] + [-gauss(101, 50, 3.0)])
    w = fwhm_rows(p)
    assert np.allclose(w[:3], 2.3548 * sig, rtol=0.03)          # FWHM = 2 sqrt(2 ln 2) sigma
    assert np.isnan(w[3])                                       # no positive peak
    wide = fwhm_rows(np.ones((1, 21)))                          # plateau to the window edges
    assert np.isfinite(wide[0]) and wide[0] >= 20


def test_count_sublines():
    one = gauss(101, 50, 4)
    two = gauss(101, 40, 3) + 0.8 * gauss(101, 62, 3)
    merged = gauss(101, 48, 4) + 0.8 * gauss(101, 53, 4)        # no dip between them
    assert count_sublines(one, 0.5, 0.2) == 1
    assert count_sublines(two, 0.5, 0.2) == 2
    assert count_sublines(merged, 0.5, 0.2) == 1
    assert count_sublines(two, 0.9, 0.2) == 1                   # 2nd peak below 0.9 x main
    assert count_sublines(np.zeros(50), 0.5, 0.2) == 0


# ---------------------------------------------------------------- objects on a synthetic map
def line_map(n_rows=1500, n_ch=52, v_kmh=50.0, t0=10.0, sigma_s=0.2, amp=8.0, seed=0):
    """Robust-z-like map with one vehicle line (Gaussian across time) + unit noise."""
    x = (np.arange(n_ch) + 0.5) * DX
    t_line = t0 + x / (v_kmh / 3.6)
    rows = (np.arange(n_rows)[:, None] + 0.5) * ROW_DT
    img = amp * np.exp(-0.5 * ((rows - t_line[None, :]) / sigma_s) ** 2)
    return img + np.random.default_rng(seed).normal(0, 1, (n_rows, n_ch)), x, t_line


def make_obj(x, t_line, v_kmh, kind="strong", track=False):
    p = np.polyfit(x, t_line, 1)
    o = DetectedObject(0, v_kmh, float(np.polyval(p, 130.0)), x[0], x[-1], t_line[0],
                       t_line[-1], kind, 10.0, direction=1, x_ref_m=130.0)
    if track:
        o.track = np.column_stack([x, t_line])
    return o


def context(img, raw=None):
    zero = np.zeros_like(img)
    ctx = FeatureContext({1: img, -1: zero}, {"hf": {1: img, -1: zero}, "lf": {1: 0.5 * img, -1: zero}},
                         ROW_DT, DX)
    if raw is not None:
        ctx.raw, ctx.fs = raw, FS
        f, ctx.bg_power = background_power(raw, int(round(FeatureParams().spec_window_s * FS)))
        ctx.freqs = f * FS
    return ctx


def test_object_trajectory_track_or_line():
    img, x, t_line = line_map()
    o = make_obj(x[5:40], t_line[5:40], 50.0)
    cols, t = object_trajectory(o, 52, DX)
    assert cols[0] == 5 and cols[-1] == 39 and np.allclose(t, t_line[5:40], atol=1e-6)
    o.track = np.column_stack([x[10:20], t_line[10:20]])
    cols, _ = object_trajectory(o, 52, DX)
    assert cols.tolist() == list(range(10, 20))


@pytest.mark.parametrize("sigma_s", [0.12, 0.3])
def test_cross_section_thickness(sigma_s):
    img, x, t_line = line_map(sigma_s=sigma_s, amp=10.0)
    p = FeatureParams()
    cs = cross_section(img, np.arange(52), t_line + 0.08, ROW_DT, p)   # line off by 2 rows
    err = np.abs((cs.r_peak + 0.5) * ROW_DT - t_line)
    assert np.median(err) <= 1.5 * ROW_DT and err.max() <= 6 * ROW_DT       # re-centred on the ridge
    f, _ = object_features(make_obj(x, t_line, 50.0), context(img), p, return_section=True)
    assert f["thickness_s"] == pytest.approx(2.3548 * sigma_s, rel=0.25)
    assert f["n_sublines"] == 1 and f["peak_z"] > 7


def test_object_features_thick_vs_thin_and_motion_fields():
    p = FeatureParams()
    thin_img, x, t_line = line_map(sigma_s=0.1, amp=4.0, seed=1)
    thick_img, _, _ = line_map(sigma_s=0.4, amp=12.0, seed=2)
    f_thin = object_features(make_obj(x, t_line, 50.0, "weak"), context(thin_img), p)
    f_thick = object_features(make_obj(x, t_line, 50.0, track=True), context(thick_img), p)
    assert set(FEATURE_NAMES) <= set(f_thick)
    assert f_thick["thickness_s"] > 2 * f_thin["thickness_s"]
    assert f_thick["log_area_zs"] > f_thin["log_area_zs"] and f_thick["peak_z"] > f_thin["peak_z"]
    assert f_thin["strong"] == 0.0 and f_thick["strong"] == 1.0
    assert f_thick["abs_v_kmh"] == pytest.approx(50.0)
    assert f_thick["z_lf"] == pytest.approx(0.5 * f_thick["z_hf"], rel=0.05)
    assert np.isnan(f_thick["spec_centroid_hz"])                 # no raw signal given
    assert np.isnan(f_thick["dv_rel"])                           # not tracked (Stage 9 fields)
    X = features_matrix([f_thin, f_thick], ["thickness_s", "missing"])
    assert X.shape == (2, 2) and np.isnan(X[:, 1]).all()
    th = section_thumbnail(None, 52, 50)
    assert th.shape == (101, 52) and np.isnan(th).all()


def test_object_near_the_image_edge_gives_no_warnings():
    img, x, t_line = line_map(n_rows=300, t0=-5.0)               # starts before the image
    f = object_features(make_obj(x, t_line, 50.0), context(img), FeatureParams())
    assert f["n_channels"] == 52 and np.isfinite(f["thickness_s"])
    short = make_obj(x[:2], t_line[:2], 50.0)
    assert np.isnan(object_features(short, context(img), FeatureParams())["thickness_s"])


# ---------------------------------------------------------------- spectrum of the raw signal
@pytest.mark.parametrize("freq,lf_rich", [(4.0, True), (150.0, False)])
def test_spectral_features_tone_on_the_trajectory(freq, lf_rich):
    rng = np.random.default_rng(3)
    n_ch, dur = 52, 60.0
    raw = rng.normal(0, 1, (int(dur * FS), n_ch))
    x = (np.arange(n_ch) + 0.5) * DX
    t_line = 10.0 + x / (50 / 3.6)
    ts = np.arange(raw.shape[0]) / FS
    env = np.exp(-0.5 * ((ts[:, None] - t_line[None, :]) / 0.3) ** 2)
    raw += 20 * env * np.sin(2 * np.pi * freq * ts)[:, None]
    p = FeatureParams()
    out = spectral_features(context(np.zeros((1500, n_ch)), raw), np.arange(n_ch), t_line, p)
    assert out["spec_centroid_hz"] == pytest.approx(freq, abs=1.25)    # 1 bin of 0.8 s
    assert (out["lf_hf_db"] > 10) == lf_rich and (out["lf_hf_db"] < -10) == (not lf_rich)
    no_raw = spectral_features(context(np.zeros((10, 2))), np.arange(2), np.zeros(2), p)
    assert all(np.isnan(v) for v in no_raw.values())
