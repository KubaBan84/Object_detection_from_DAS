"""Tests of das.enhancement (Stage 5)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from das.config import EnhancementParams
from das.enhancement import (angle_to_velocity, bank_angles, directional_filter_bank, enhance,
                             fk_fan_mask, fk_velocity_filter, gabor_kernels, line_snr,
                             oriented_line_kernels, ridge_filter, robust_z,
                             subtract_background, tukey_window, velocity_to_angle)

DT, DX = 0.04, 5.1065
T, X = 1500, 52


def _line(v: float, x0: float = 130.0, width: float = 6.0) -> np.ndarray:
    t = np.arange(T)[:, None] * DT
    x = np.arange(X)[None, :] * DX
    return np.exp(-0.5 * ((x - v * t - x0) / width) ** 2)


def _residual_energy(a: np.ndarray, pad_mode: str = "zero") -> float:
    out = fk_velocity_filter(a, DT, DX, 10 / 3.6, 150 / 3.6, direction=0, pad_mode=pad_mode)
    return float((out ** 2).sum() / ((a - a.mean()) ** 2).sum())


# ---------------------------------------------------------------- f-k filter
@pytest.mark.parametrize("pad_mode", ["zero", "reflect", "tukey", "none"])
def test_fan_filter_keeps_moving_line_removes_stripes(pad_mode):
    line = _line(-14.0, x0=400.0)               # 50 km/h towards decreasing position
    vertical = np.zeros((T, X))
    vertical[:, 20] = 3.0                       # static channel stripe (v = 0)
    assert _residual_energy(line, pad_mode) > 0.6
    assert _residual_energy(vertical, pad_mode) < 1e-6


def test_reflect_padding_removes_edge_leakage_of_horizontal_stripes():
    horizontal = np.zeros((T, X))
    horizontal[700:705, :] = 3.0                # simultaneous event over the whole cable
    assert _residual_energy(horizontal, "zero") > 0.005       # zero padding leaks at the edges
    assert _residual_energy(horizontal, "reflect") < 1e-6     # mirrored: no edges at all
    with pytest.raises(ValueError):
        fk_velocity_filter(horizontal, DT, DX, 1, 2, pad_mode="bogus")


@pytest.mark.parametrize("pad_mode", ["zero", "reflect"])
def test_fan_filter_direction_selectivity(pad_mode):
    pos, neg = _line(+14.0, x0=-130.0), _line(-14.0, x0=400.0)   # both fully inside
    img = pos + neg
    c = lambda a, b: np.corrcoef(a.ravel(), (b - b.mean()).ravel())[0, 1]  # noqa: E731
    keep_pos = fk_velocity_filter(img, DT, DX, 10 / 3.6, 150 / 3.6, +1, pad_mode=pad_mode)
    keep_neg = fk_velocity_filter(img, DT, DX, 10 / 3.6, 150 / 3.6, -1, pad_mode=pad_mode)
    assert c(keep_pos, pos) > 0.8 and c(keep_pos, neg) < 0.2
    assert c(keep_neg, neg) > 0.8 and c(keep_neg, pos) < 0.2


def test_fan_mask_values():
    f = np.array([0.0, 1.0, -1.0])
    k = np.array([0.0, 0.1, -0.1])
    m = fk_fan_mask(f, k, 2.0, 50.0, direction=0, taper=0.0)
    assert (m[:, 0] == 0).all()                            # k = 0 rejected
    assert m[1, 2] == 1 and m[2, 1] == 1                   # |v| = 10 m/s, both signs
    m_pos = fk_fan_mask(f, k, 2.0, 50.0, direction=+1, taper=0.0)
    assert m_pos[1, 2] == 1 and m_pos[2, 2] == 0           # v = -f/k: (1, -0.1) -> +10


def test_tukey_window():
    w = tukey_window(52, 0.2)
    assert w[0] == pytest.approx(0) and w[26] == 1 and np.allclose(w, w[::-1])


# ---------------------------------------------------------------- image enhancement
def test_velocity_angle_round_trip():
    v = np.array([-30.0, -2.8, 2.8, 41.7])
    a = velocity_to_angle(v, 0.04, 0.64)
    np.testing.assert_allclose(angle_to_velocity(a, 0.04, 0.64), v)
    assert (np.sign(a) == np.sign(v)).all()
    angs = bank_angles(10 / 3.6, 150 / 3.6, -1, 24, 0.04, 0.64)
    assert len(angs) == 24 and (angs < 0).all()


def _slanted_image(angle: float, shape=(200, 200), width: float = 1.5) -> np.ndarray:
    r, c = np.mgrid[:shape[0], :shape[1]].astype(float)
    r -= shape[0] / 2
    c -= shape[1] / 2
    perp = -r * np.sin(angle) + c * np.cos(angle)
    return np.exp(-0.5 * (perp / width) ** 2)


@pytest.mark.parametrize("kind", ["line", "gabor"])
def test_filter_bank_finds_line_orientation(kind):
    angles = np.deg2rad(np.linspace(-60, 60, 13))
    true = 4                                                     # -20 deg
    img = _slanted_image(angles[true])
    kernels = (oriented_line_kernels(41, angles, 1.5) if kind == "line"
               else gabor_kernels(angles, wavelength=8.0, sigma=4.0))
    for k in kernels:
        assert abs(float(k.mean())) < 1e-6 and np.linalg.norm(k) == pytest.approx(1, rel=1e-5)
    resp, idx = directional_filter_bank(img, kernels)
    assert idx[100, 100] == true and resp[100, 100] > 0


def test_filter_bank_raises_snr_of_weak_line():
    rng = np.random.default_rng(0)
    angle = np.deg2rad(30)
    img = 0.7 * _slanted_image(angle, (300, 300)) + rng.standard_normal((300, 300))
    resp, _ = directional_filter_bank(img, oriented_line_kernels(41, np.array([angle]), 1.5))
    on = _slanted_image(angle, (300, 300)) > 0.9
    z_in, z_out = robust_z(img), robust_z(resp)
    assert z_out[on].mean() > 2.5 * z_in[on].mean()


def test_subtract_background_keeps_narrow_lines():
    img = np.tile(np.linspace(0, 5, 300), (100, 1))      # slow ramp along space
    img[:, 150:154] += 3.0                                # narrow bright line
    out = subtract_background(img, 51)
    assert out[:, 152].mean() == pytest.approx(3.0, abs=0.2)
    assert np.abs(out[:, :100]).max() < 0.2


def test_ridge_filter_runs():
    out = ridge_filter(_slanted_image(0.3, (80, 80)), (1.0, 2.0), "sato")
    assert out.shape == (80, 80) and out[40, 40] > np.median(out)


def test_line_snr():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((500, 50))
    p99_noise, frac_noise = line_snr(noise)
    assert 2.0 < p99_noise < 2.7 and frac_noise < 1e-3
    img = noise.copy()
    img[:, 25] += 10.0
    p99, frac = line_snr(img)
    assert p99 > 5 and frac > 0.015


@pytest.mark.parametrize("enhancer", ["filterbank", "gabor", "sato", "none"])
def test_enhance_outputs(enhancer):
    rng = np.random.default_rng(1)
    img = _line(-14.0, x0=400.0) + 0.2 * rng.standard_normal((T, X))
    ep = replace(EnhancementParams(), enhancer=enhancer, n_angles=6)
    out = enhance(img, ep, DT, DX)
    assert out.col_dx == pytest.approx(DX / ep.space_upsample)
    for d in (1, -1):
        assert out.response[d].shape == (T, X * ep.space_upsample)
    # the line moves towards -x, so the negative-direction map must respond more strongly
    assert out.response[-1].max() > out.response[1].max()
    if enhancer in ("filterbank", "gabor"):
        assert len(out.velocities[-1]) == 6 and (out.velocities[-1] < 0).all()
    assert {"input", "fk_both", "fk_removed", "fk_pos", "fk_neg"} <= set(out.stages)
