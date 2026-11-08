"""Tests of das.preprocessing (Stage 4)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from das.config import PreprocessingParams
from das.preprocessing import (bandpass_fft, bandpass_response, detrend_channels,
                               envelope_downsample, interpolate_channels, log_ratio,
                               normalize_envelope, preprocess, remove_common_mode,
                               robust_normalize, smooth_anisotropic, upsample_space)

FS = 625.0


def _tone(f0: float, n: int = 62500) -> np.ndarray:
    return np.sin(2 * np.pi * f0 * np.arange(n) / FS)


# ---------------------------------------------------------------- raw-signal operations
def test_bandpass_passes_in_band_and_rejects_out_of_band():
    x = np.column_stack([_tone(80.0), _tone(10.0), _tone(250.0)]).astype(np.float32)
    y = bandpass_fft(x, FS, 60.0, 200.0, taper_hz=1.0)
    assert y.dtype == np.float32 and y.shape == x.shape
    core = slice(5000, -5000)
    assert np.max(np.abs(y[core, 0] - x[core, 0])) < 0.01            # in band: error < 1 %
    for c in (1, 2):                                                   # out of band: > 40 dB
        assert 20 * np.log10(np.std(y[core, c]) / np.std(x[core, c])) < -40


def test_bandpass_is_zero_phase():
    pulse = np.zeros(4096)
    pulse[2000] = 1.0
    y = bandpass_fft(pulse, FS, 20.0, 100.0)
    assert np.argmax(np.abs(y)) == 2000          # no time shift
    np.testing.assert_allclose(y[2000 - 50:2000], y[2001:2051][::-1], atol=1e-12)


def test_bandpass_response_edges():
    f = np.array([0.0, 59.0, 60.0, 61.0, 100.0, 312.5])
    h = bandpass_response(f, 60.0, 312.5, taper_hz=2.0)
    assert h[0] == 0 and h[2] == pytest.approx(0.5) and h[4] == 1 and h[5] == 1
    h_notch = bandpass_response(np.array([49.0, 50.0, 51.0, 80.0]), 0, 400, notches=(50.0,),
                                notch_width_hz=4.0)
    assert h_notch[1] == 0 and 0 < h_notch[0] < 1 and h_notch[3] == 1


def test_remove_common_mode():
    rng = np.random.default_rng(0)
    common = rng.standard_normal((1000, 1))
    local = np.zeros((1000, 9))
    local[:, 4] = 5.0                              # one "vehicle" channel
    y = remove_common_mode(common + local, "median")
    np.testing.assert_allclose(y[:, 4], 5.0, atol=1e-12)    # local signal kept
    np.testing.assert_allclose(np.delete(y, 4, axis=1), 0.0, atol=1e-12)
    assert remove_common_mode(local, "none") is local
    with pytest.raises(ValueError):
        remove_common_mode(local, "bogus")


def test_detrend_channels():
    t = np.arange(1000, dtype=np.float64)
    x = np.column_stack([3 + 0.01 * t, -2 - 0.5 * t]).astype(np.float32)
    np.testing.assert_allclose(detrend_channels(x), 0.0, atol=1e-3)
    assert detrend_channels(x).dtype == np.float32


def test_envelope_methods():
    x = np.tile(_tone(50.0, 2500)[:, None], (1, 2)) * np.array([1.0, 3.0])
    rms = envelope_downsample(x, 25, "rms")
    assert rms.shape == (100, 2)
    np.testing.assert_allclose(rms.mean(axis=0), [1 / np.sqrt(2), 3 / np.sqrt(2)], rtol=0.02)
    np.testing.assert_allclose(envelope_downsample(x, 25, "abs_mean").mean(axis=0),
                               [2 / np.pi, 6 / np.pi], rtol=0.02)
    assert envelope_downsample(x, 25, "decimate").shape == (100, 2)
    with pytest.raises(ValueError):
        envelope_downsample(x, 25, "bogus")


# ---------------------------------------------------------------- image operations
def test_robust_normalize_equalises_gains():
    rng = np.random.default_rng(1)
    img = np.abs(rng.standard_normal((5000, 3))) * np.array([1.0, 10.0, 100.0])
    z = robust_normalize(img)
    np.testing.assert_allclose(np.median(z, axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.median(np.abs(z), axis=0), 1 / 1.4826, rtol=1e-6)
    assert np.isfinite(robust_normalize(np.zeros((10, 2)))).all()      # MAD = 0 is safe


def test_log_ratio_and_normalize_envelope():
    env = np.ones((100, 3)) * np.array([1.0, 5.0, 50.0])
    env[50] *= 100.0                                                   # an event on all channels
    lr = log_ratio(env)
    np.testing.assert_allclose(lr[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(lr[50], 2.0, atol=1e-6)                 # gain-independent
    img = normalize_envelope(env, "log_ratio", 100.0, clip_low=True)
    assert img.min() >= 0 and img.max() == pytest.approx(2.0)
    img_z = normalize_envelope(env + np.random.default_rng(0).random((100, 3)), "robust_z",
                               99.5, True)
    assert img_z.min() >= 0
    with pytest.raises(ValueError):
        normalize_envelope(env, "bogus", 99.5, True)


def test_interpolate_channels():
    img = np.tile(np.arange(5, dtype=float), (4, 1))
    img[:, 2] = 100.0
    out = interpolate_channels(img, np.array([False, False, True, False, False]))
    np.testing.assert_allclose(out[:, 2], 2.0)
    assert interpolate_channels(img, np.zeros(5, bool)) is img


def test_smooth_and_upsample():
    img = np.zeros((50, 10))
    img[25, 5] = 1.0
    s = smooth_anisotropic(img, sigma_t=2.0, sigma_x=0.0)
    assert s[:, 5].sum() == pytest.approx(1.0, rel=1e-3) and s[:, 4].sum() == 0  # time only
    assert smooth_anisotropic(img, 0, 0) is img
    up = upsample_space(img, 8)
    assert up.shape == (50, 80)


# ---------------------------------------------------------------- full chain
def test_preprocess_shapes_crop_and_stages():
    rng = np.random.default_rng(2)
    pad, n = 625 * 2, 625 * 8
    x = rng.standard_normal((pad + n + pad, 6)).astype(np.float32) * 1e-7
    out = preprocess(x, PreprocessingParams(), FS, dx=5.0, crop=(pad, pad))
    rows = n // 25
    assert out.row_dt == pytest.approx(0.04) and out.col_dx == 5.0
    assert set(out.images) == {"hf", "lf"}
    for st in out.bands.values():
        for k in ("envelope", "removed_by_bandpass", "normalized", "image"):
            assert st[k].shape == (rows, 6)
    for k in ("raw", "common_mode", "after_common_mode"):
        assert out.common[k].shape == (rows, 6)
    with pytest.raises(ValueError):
        preprocess(x, PreprocessingParams(), FS, dx=5.0, crop=(7, 0))  # not divisible by 25


def test_preprocess_makes_moving_vehicle_stand_out():
    """A burst of 80 Hz vibration moving along the cable must dominate the HF image."""
    rng = np.random.default_rng(3)
    T, C = 625 * 20, 20
    x = 0.1 * rng.standard_normal((T, C))
    t = np.arange(T) / FS
    for c in range(C):                              # vehicle passes channel c at t = 2 + c * 0.5 s
        on = np.abs(t - (2 + 0.5 * c)) < 0.2
        x[on, c] += np.sin(2 * np.pi * 80 * t[on])
    params = replace(PreprocessingParams(), common_mode="median")
    img = preprocess(x.astype(np.float32), params, FS, dx=5.0, bands=("hf",)).images["hf"]
    rows_on = ((2 + 0.5 * np.arange(C)) / 0.04).astype(int)
    on_track = img[rows_on, np.arange(C)].mean()
    assert on_track > 5 * np.median(img) + 0.5
