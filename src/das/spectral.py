"""Frequency analysis: per-channel FFT, Welch PSD, STFT, f-k spectrum (plan: Stage 3).

Implemented with ``numpy.fft`` only (scipy is not on the allowed library list).
All functions take (time, space) arrays and transform **each channel independently**
along axis 0, as required by the project brief.

f-k sign convention (numpy FFT, exp(-2*pi*i*(f*t + k*x))): a pattern moving with velocity
``v`` (x = v*t + c) concentrates its energy on the line ``f = -v * k``. Hence
``apparent_velocity(f, k) = -f / k`` (positive = towards increasing cable position).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ======================================================================================
# Helpers
# ======================================================================================
def _frames(x: np.ndarray, nperseg: int, step: int) -> np.ndarray:
    """(T, C) -> (n_frames, nperseg, C) view of overlapping frames along time."""
    if x.shape[0] < nperseg:
        raise ValueError(f"Signal shorter ({x.shape[0]}) than nperseg ({nperseg})")
    return sliding_window_view(x, nperseg, axis=0)[::step].transpose(0, 2, 1)


def _step(nperseg: int, overlap: float) -> int:
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    return max(1, int(round(nperseg * (1.0 - overlap))))


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _one_sided_scale(n_freq: int, nfft: int) -> np.ndarray:
    """Factor 2 for all bins except DC (and Nyquist for even nfft) -> one-sided PSD."""
    scale = np.full(n_freq, 2.0)
    scale[0] = 1.0
    if nfft % 2 == 0:
        scale[-1] = 1.0
    return scale


# ======================================================================================
# Spectra along time (per channel)
# ======================================================================================
def channel_fft(x: np.ndarray, fs: float, window: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum of every channel (rfft along axis 0, independently per channel).

    The channel mean is removed and a Hann window applied (``window=True``). The amplitude
    is normalised so that a sinusoid of amplitude A gives a peak of height ~A.
    Returns (freqs [Hz], amplitude (n_freq, n_channels)).
    """
    v = _as_2d(x)
    v = v - v.mean(axis=0)
    w = np.hanning(v.shape[0]) if window else np.ones(v.shape[0])
    spec = np.fft.rfft(v * w[:, None], axis=0)
    amp = np.abs(spec) * _one_sided_scale(spec.shape[0], v.shape[0])[:, None] / w.sum()
    return np.fft.rfftfreq(v.shape[0], 1.0 / fs), amp


def welch_psd(x: np.ndarray, fs: float, nperseg: int = 1024, overlap: float = 0.5
              ) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density per channel [unit^2 / Hz].

    Hann window, per-frame mean removal, one-sided density scaling, so that
    ``sum(psd) * df`` equals the signal variance (Parseval).
    Returns (freqs, psd (n_freq, n_channels)).
    """
    v = _as_2d(x)
    fr = _frames(v, nperseg, _step(nperseg, overlap))
    w = np.hanning(nperseg)
    fr = (fr - fr.mean(axis=1, keepdims=True)) * w[None, :, None]
    spec = np.fft.rfft(fr, axis=1)
    psd = (np.abs(spec) ** 2).mean(axis=0) / (fs * np.sum(w ** 2))
    psd *= _one_sided_scale(psd.shape[0], nperseg)[:, None]
    return np.fft.rfftfreq(nperseg, 1.0 / fs), psd


def stft(x1d: np.ndarray, fs: float, nperseg: int = 512, overlap: float = 0.75
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform of one channel.

    Returns (freqs [Hz], frame-centre times [s], |Z| (n_freq, n_frames)) with
    Hann-window amplitude normalisation.
    """
    v = np.asarray(x1d, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("stft expects a 1-D signal (one channel)")
    step = _step(nperseg, overlap)
    fr = sliding_window_view(v, nperseg)[::step]
    w = np.hanning(nperseg)
    z = np.fft.rfft((fr - fr.mean(axis=1, keepdims=True)) * w, axis=1).T
    mag = np.abs(z) * _one_sided_scale(z.shape[0], nperseg)[:, None] / w.sum()
    times = (np.arange(fr.shape[0]) * step + nperseg / 2) / fs
    return np.fft.rfftfreq(nperseg, 1.0 / fs), times, mag


def band_power(freqs: np.ndarray, psd: np.ndarray, bands: list[tuple[float, float]]
               ) -> np.ndarray:
    """Integrate a PSD (n_freq, ...) over [lo, hi) bands -> (n_bands, ...)."""
    df = freqs[1] - freqs[0]
    out = [psd[(freqs >= lo) & (freqs < hi)].sum(axis=0) * df for lo, hi in bands]
    return np.asarray(out)


def band_energy(x: np.ndarray, fs: float, bands: list[tuple[float, float]],
                nperseg: int = 1024) -> np.ndarray:
    """Power of every channel in every band (Welch PSD integrated) -> (n_bands, n_channels)."""
    freqs, psd = welch_psd(x, fs, nperseg)
    return band_power(freqs, psd, bands)


# ======================================================================================
# Vehicle vs background contrast (choice of the pass band)
# ======================================================================================
def cell_spectra(x: np.ndarray, fs: float, win: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power spectra of non-overlapping (window x channel) cells.

    Returns (freqs, power (n_windows, n_channels, n_freq), broadband RMS (n_windows, n_channels)).
    """
    n = (x.shape[0] // win) * win
    cells = np.asarray(x[:n], dtype=np.float64).reshape(n // win, win, x.shape[1])
    cells = cells - cells.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(cells ** 2, axis=1))
    w = np.hanning(win)
    spec = np.fft.rfft(cells * w[None, :, None], axis=1)
    power = (np.abs(spec) ** 2 / (fs * np.sum(w ** 2))).transpose(0, 2, 1)
    power *= _one_sided_scale(power.shape[2], win)[None, None, :]
    return np.fft.rfftfreq(win, 1.0 / fs), power, rms


def select_cells(rms: np.ndarray, q_vehicle: float = 0.9, q_background: float = 0.3
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of 'vehicle' and 'background' cells, chosen per channel by RMS quantile.

    Per-channel quantiles make the selection independent of channel sensitivity.
    """
    hi = np.quantile(rms, q_vehicle, axis=0)
    lo = np.quantile(rms, q_background, axis=0)
    return rms >= hi[None, :], rms <= lo[None, :]


def band_separability(freqs: np.ndarray, power: np.ndarray, vehicle: np.ndarray,
                      background: np.ndarray, f_lows: np.ndarray, f_highs: np.ndarray
                      ) -> np.ndarray:
    """Separability d' of vehicle and background cells for every band [f_low, f_high).

    For each band the cell power is integrated (cumulative sums - vectorised over the
    whole grid) and log10-transformed; d' = (mean_v - mean_b) / sqrt((var_v + var_b) / 2).
    Narrow bands are penalised automatically by their noisier power estimates.
    Returns (len(f_lows), len(f_highs)); invalid bands (f_low >= f_high) are NaN.
    """
    df = freqs[1] - freqs[0]
    cs = np.concatenate([np.zeros(power.shape[:2] + (1,)), np.cumsum(power, axis=2)], axis=2)
    i_lo = np.searchsorted(freqs, f_lows)
    i_hi = np.searchsorted(freqs, f_highs)
    bp = (cs[..., i_hi[None, :]] - cs[..., i_lo[:, None]]) * df       # (W, C, nL, nH)
    lp = np.log10(np.maximum(bp, np.finfo(np.float64).tiny))
    v, b = lp[vehicle], lp[background]                                  # (n_cells, nL, nH)
    d = (v.mean(axis=0) - b.mean(axis=0)) / np.sqrt(0.5 * (v.var(axis=0) + b.var(axis=0)))
    d[f_lows[:, None] >= f_highs[None, :]] = np.nan
    return d


def find_tonal_peaks(freqs: np.ndarray, psd_1d: np.ndarray, prominence_db: float = 10.0,
                     baseline_bins: int = 51, f_min: float = 1.0) -> list[dict]:
    """Narrow spectral lines: bins exceeding a running-median baseline by ``prominence_db``.

    Adjacent qualifying bins are merged; the strongest bin of each group is reported.
    """
    half = baseline_bins // 2
    padded = np.pad(psd_1d, half, mode="edge")
    baseline = np.median(sliding_window_view(padded, baseline_bins), axis=1)
    excess = 10 * np.log10(np.maximum(psd_1d, 1e-300) / np.maximum(baseline, 1e-300))
    idx = np.flatnonzero((excess > prominence_db) & (freqs >= f_min))
    peaks = []
    for group in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []:
        j = group[np.argmax(excess[group])]
        peaks.append({"freq_hz": float(freqs[j]), "excess_db": float(excess[j])})
    return peaks


# ======================================================================================
# Frequency-wavenumber (f-k) analysis
# ======================================================================================
def fk_spectrum(img: np.ndarray, dt: float, dx: float, pad_x: int | None = None,
                taper: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D amplitude spectrum of a (time, space) image.

    The image mean is removed, a Hann taper is applied in both directions (``taper``) and
    the space axis is zero-padded to ``pad_x`` columns (finer k sampling).
    Returns (f [Hz] ascending, k [cycles/m] ascending, |F| (n_f, n_k)) after fftshift.
    """
    v = np.asarray(img, dtype=np.float64)
    v = v - v.mean()
    if taper:
        v = v * np.hanning(v.shape[0])[:, None] * np.hanning(v.shape[1])[None, :]
    nx = max(pad_x or v.shape[1], v.shape[1])
    spec = np.fft.fftshift(np.abs(np.fft.fft2(v, s=(v.shape[0], nx))))
    f = np.fft.fftshift(np.fft.fftfreq(v.shape[0], dt))
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    return f, k, spec


def apparent_velocity(f: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Apparent velocity [m/s] of every (f, k) cell: v = -f / k (inf where k == 0)."""
    ff, kk = np.meshgrid(f, k, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(kk != 0, -ff / np.where(kk != 0, kk, 1.0), np.inf)


def fk_fan_fraction(img: np.ndarray, dt: float, dx: float, v_min_ms: float, v_max_ms: float,
                    pad_x: int = 128) -> tuple[float, float]:
    """Fraction of f-k energy of an image inside the vehicle velocity fan, per direction.

    Returns (fraction with v in [v_min, v_max], fraction with v in [-v_max, -v_min]).
    Used as a label-free criterion: the better a band shows moving objects, the more of
    the envelope-image energy lies inside the fan.
    """
    f, k, spec = fk_spectrum(img, dt, dx, pad_x=pad_x)
    edges = np.array([-v_max_ms, -v_min_ms, v_min_ms, v_max_ms])
    frac, _ = fk_velocity_energy(f, k, spec, edges)
    return float(frac[2]), float(frac[0])


def fk_velocity_energy(f: np.ndarray, k: np.ndarray, spec: np.ndarray,
                       v_edges_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of f-k energy (|F|^2, DC excluded) per apparent-velocity bin.

    ``v_edges_ms`` are signed bin edges [m/s]; energy outside all bins (|v| beyond the
    edges, incl. k = 0) is returned as the last element. -> (fractions per bin, rest).
    """
    vel = apparent_velocity(f, k)
    e = spec ** 2
    e[np.argmin(np.abs(f)), np.argmin(np.abs(k))] = 0.0  # remove DC
    total = e.sum()
    idx = np.digitize(vel, v_edges_ms) - 1
    valid = (idx >= 0) & (idx < len(v_edges_ms) - 1) & np.isfinite(vel)
    frac = np.bincount(idx[valid], weights=e[valid], minlength=len(v_edges_ms) - 1) / total
    return frac, 1.0 - frac.sum()
