"""Tests of das.spectral (Stage 3)."""
from __future__ import annotations

import numpy as np
import pytest

from das import spectral

FS = 625.0


def _sine(f0: float, amp: float = 2.0, n: int = 62500, n_ch: int = 3) -> np.ndarray:
    t = np.arange(n) / FS
    return amp * np.sin(2 * np.pi * f0 * t)[:, None] * np.ones((1, n_ch))


def test_channel_fft_amplitude_and_independence():
    x = _sine(50.0)
    x[:, 2] = _sine(120.0, amp=0.5, n_ch=1)[:, 0]   # different content on one channel
    freqs, amp = spectral.channel_fft(x, FS)
    assert freqs[np.argmax(amp[:, 0])] == pytest.approx(50.0, abs=0.02)
    assert freqs[np.argmax(amp[:, 2])] == pytest.approx(120.0, abs=0.02)
    assert amp[:, 0].max() == pytest.approx(2.0, rel=0.02)
    assert amp[:, 2].max() == pytest.approx(0.5, rel=0.02)


def test_welch_parseval_and_peak():
    rng = np.random.default_rng(0)
    x = _sine(30.0, amp=1.0, n=40000, n_ch=2) + 0.1 * rng.standard_normal((40000, 2))
    freqs, psd = spectral.welch_psd(x, FS, nperseg=1024)
    df = freqs[1] - freqs[0]
    np.testing.assert_allclose(psd.sum(axis=0) * df, x.var(axis=0), rtol=0.05)
    assert freqs[np.argmax(psd[:, 0])] == pytest.approx(30.0, abs=df)


def test_band_power_matches_variance():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((20000, 2))
    freqs, psd = spectral.welch_psd(x, FS, 512)
    bp = spectral.band_power(freqs, psd, [(0, 100), (100, FS / 2 + 1)])
    np.testing.assert_allclose(bp.sum(axis=0), x.var(axis=0), rtol=0.05)
    assert bp.shape == (2, 2)


def test_stft_tracks_frequency_change():
    t = np.arange(int(4 * FS)) / FS
    x = np.where(t < 2, np.sin(2 * np.pi * 20 * t), np.sin(2 * np.pi * 80 * t))
    f, times, z = spectral.stft(x, FS, nperseg=250, overlap=0.5)
    peak = f[np.argmax(z, axis=0)]
    assert np.all(np.abs(peak[times < 1.5] - 20) < 3)
    assert np.all(np.abs(peak[times > 2.5] - 80) < 3)
    with pytest.raises(ValueError):
        spectral.stft(np.zeros((10, 2)), FS)


def test_find_tonal_peaks():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(60000) + 0.5 * np.sin(2 * np.pi * 50 * np.arange(60000) / FS)
    freqs, psd = spectral.welch_psd(x, FS, 2048)
    peaks = spectral.find_tonal_peaks(freqs, psd[:, 0])
    assert len(peaks) == 1 and peaks[0]["freq_hz"] == pytest.approx(50, abs=0.5)


def _moving_line(v: float, x0: float, dt=0.04, dx=5.1065, T=1500, X=52, width=6.0):
    """Gaussian ridge x = v t + x0 on a (time, space) grid."""
    t = np.arange(T)[:, None] * dt
    x = np.arange(X)[None, :] * dx
    return np.exp(-0.5 * ((x - v * t - x0) / width) ** 2)


@pytest.mark.parametrize("v, x0", [(14.0, -130.0), (-14.0, 400.0)])
def test_fk_sign_convention(v, x0):
    """A pattern moving with velocity v concentrates on f = -v k (apparent v = -f/k)."""
    img = _moving_line(v, x0=x0)        # line fully inside the image
    f, k, s = spectral.fk_spectrum(img, 0.04, 5.1065, pad_x=128)
    e = s ** 2
    e[:, np.abs(k) < 0.01] = 0          # ignore the low-k smear of the line ends
    i, j = np.unravel_index(np.argmax(e), e.shape)
    assert -f[i] / k[j] == pytest.approx(v, rel=0.1)
    pos, neg = spectral.fk_fan_fraction(img, 0.04, 5.1065, 10 / 3.6, 150 / 3.6)
    assert (pos > 0.9 and neg < 0.05) if v > 0 else (neg > 0.9 and pos < 0.05)


def test_apparent_velocity_k0_is_infinite():
    vel = spectral.apparent_velocity(np.array([1.0, 2.0]), np.array([-0.1, 0.0, 0.1]))
    assert np.isinf(vel[:, 1]).all()
    np.testing.assert_allclose(vel[:, 2], [-10.0, -20.0])


def test_cell_spectra_and_separability():
    rng = np.random.default_rng(3)
    x = 0.1 * rng.standard_normal((12500, 4))
    t = np.arange(12500) / FS
    burst = (t > 10) & (t < 14)
    x[burst] += np.sin(2 * np.pi * 80 * t[burst])[:, None]   # "vehicle" at 80 Hz
    freqs, power, rms = spectral.cell_spectra(x, FS, 1250)
    assert power.shape == (10, 4, 626) and rms.shape == (10, 4)
    veh, bg = spectral.select_cells(rms, 0.75, 0.5)
    assert veh[5:7].all()
    d = spectral.band_separability(freqs, power, veh, bg, np.array([5.0, 60.0]),
                                   np.array([40.0, 100.0]))
    assert d[1, 1] > d[0, 0]           # 60-100 Hz separates better than 5-40 Hz
    assert np.isnan(d[1, 0])           # invalid band (f_low >= f_high)
