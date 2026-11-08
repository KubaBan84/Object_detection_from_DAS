"""Noise removal and conversion of the raw signal into energy images (plan: Stage 4).

Chain (each step stored for the report's before/after figures)::

    raw (T, C) --detrend--> --common-mode removal--> x_cm
    x_cm --band-pass HF (60-312.5 Hz)--> RMS envelope / 25 --> normalise + log + clip --> image_hf
    x_cm --band-pass LF (1-8 Hz)-------> RMS envelope / 25 --> normalise + log + clip --> image_lf

The two bands were chosen in Stage 3 (``scripts/run_spectral.py``). Images have one row
per 0.04 s and one column per channel; spatial upsampling is done in Stage 5 after the
f-k filter (which needs the true channel spacing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PreprocessingParams
from .stats import MAD_TO_SIGMA, block_rms

EPS = 1e-12


# ======================================================================================
# Raw-signal operations (time, space), full sampling rate
# ======================================================================================
def detrend_channels(x: np.ndarray) -> np.ndarray:
    """Remove the mean and the linear trend of every channel (least squares, vectorised)."""
    v = np.asarray(x, dtype=np.float64)
    t = np.arange(v.shape[0], dtype=np.float64)
    t -= t.mean()
    v = v - v.mean(axis=0)
    slope = (t @ v) / (t @ t)
    return (v - t[:, None] * slope[None, :]).astype(x.dtype, copy=False)


def common_mode_signal(x: np.ndarray, method: str = "median") -> np.ndarray:
    """Across-channel median (or mean) at every time sample -> (T,)."""
    if method == "median":
        return np.median(x, axis=1)
    if method == "mean":
        return np.mean(x, axis=1)
    raise ValueError(f"Unknown common-mode method {method!r}")


def remove_common_mode(x: np.ndarray, method: str = "median") -> np.ndarray:
    """Subtract the across-channel median (or mean) at every time sample.

    Removes disturbances recorded simultaneously along the whole fibre (horizontal
    stripes, infinite apparent velocity). The median is robust to a vehicle that occupies
    only a few channels. ``method="none"`` returns the input unchanged. Returns x.dtype.
    """
    if method == "none":
        return x
    return (x - common_mode_signal(x, method)[:, None]).astype(x.dtype, copy=False)


def bandpass_response(freqs: np.ndarray, f_low: float, f_high: float, taper_hz: float = 0.5,
                      notches: tuple[float, ...] = (), notch_width_hz: float = 1.0
                      ) -> np.ndarray:
    """Real, non-negative frequency response with raised-cosine (Hann) transitions.

    The gain is 0.5 exactly at ``f_low`` and ``f_high``; each transition is ``taper_hz``
    wide. ``f_low <= 0`` gives a low-pass, ``f_high >= Nyquist`` a high-pass. Optional
    ``notches`` remove narrow lines (raised-cosine dips of width ``notch_width_hz``).
    """
    f = np.abs(np.asarray(freqs, dtype=np.float64))
    half = max(taper_hz, 1e-12) / 2.0

    def rising(edge: float) -> np.ndarray:  # 0 below edge - half, 1 above edge + half
        t = np.clip((f - (edge - half)) / (2 * half), 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * t)

    h = np.ones_like(f)
    if f_low > 0:
        h *= rising(f_low)
    if np.isfinite(f_high) and f_high < f.max():
        h *= 1.0 - rising(f_high)
    for fn in notches:
        d = np.clip(np.abs(f - fn) / (notch_width_hz / 2.0), 0.0, 1.0)
        h *= 0.5 - 0.5 * np.cos(np.pi * d)
    return h


def bandpass_fft(x: np.ndarray, fs: float, f_low: float, f_high: float,
                 taper_hz: float = 0.5, notches: tuple[float, ...] = (),
                 notch_width_hz: float = 1.0) -> np.ndarray:
    """Zero-phase band-pass of every channel in the frequency domain (rfft along time).

    The response is real, so there is no phase shift - trajectories are not moved in
    time. Circular wrap-around affects only the segment edges; load the segment with
    padding (``io.load_segment(pad_files=1)``) and crop afterwards. Returns x.dtype.
    """
    x = np.asarray(x)
    spec = np.fft.rfft(np.asarray(x, dtype=np.float64), axis=0)
    h = bandpass_response(np.fft.rfftfreq(x.shape[0], 1.0 / fs), f_low, f_high, taper_hz,
                          notches, notch_width_hz)
    spec *= h.reshape((-1,) + (1,) * (x.ndim - 1))
    return np.fft.irfft(spec, n=x.shape[0], axis=0).astype(x.dtype, copy=False)


def envelope_downsample(x: np.ndarray, factor: int, method: str = "rms") -> np.ndarray:
    """Energy envelope and decimation in one step -> (T // factor, C).

    ``rms``: block RMS (default); ``abs_mean``: block mean of |x|; ``decimate``: |x| at
    every ``factor``-th sample (no envelope - kept only as a negative reference: it aliases
    the oscillating signal).
    """
    if method == "rms":
        return block_rms(x, factor)
    n = (x.shape[0] // factor) * factor
    if method == "abs_mean":
        return np.abs(np.asarray(x[:n], dtype=np.float64)).reshape(
            n // factor, factor, x.shape[1]).mean(axis=1)
    if method == "decimate":
        return np.abs(np.asarray(x[:n:factor], dtype=np.float64))
    raise ValueError(f"Unknown envelope method {method!r}")


# ======================================================================================
# Image operations (rows = 0.04 s, columns = channels)
# ======================================================================================
def interpolate_channels(img: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace bad channels by linear interpolation between the nearest good neighbours."""
    bad = np.asarray(bad, dtype=bool)
    if not bad.any():
        return img
    good = np.flatnonzero(~bad)
    if good.size < 2:
        raise ValueError("Need at least two good channels to interpolate")
    out = np.array(img, dtype=np.float64)
    for c in np.flatnonzero(bad):
        left, right = good[good < c], good[good > c]
        lo = left[-1] if left.size else right[0]
        hi = right[0] if right.size else left[-1]
        w = 0.5 if lo == hi else (c - lo) / (hi - lo)
        out[:, c] = (1 - w) * out[:, lo] + w * out[:, hi]
    return out


def robust_normalize(img: np.ndarray, axis: int = 0) -> np.ndarray:
    """(x - median) / (1.4826 * MAD) along ``axis`` (per channel for axis=0).

    Equalises channel sensitivity (removes vertical stripes); robust to the heavy-tailed
    data (EDA: kurtosis 435). A zero MAD (quantised, silent channel) is replaced by eps.
    """
    v = np.asarray(img, dtype=np.float64)
    med = np.median(v, axis=axis, keepdims=True)
    mad = np.median(np.abs(v - med), axis=axis, keepdims=True)
    return (v - med) / np.maximum(MAD_TO_SIGMA * mad, EPS)


def log_ratio(env: np.ndarray) -> np.ndarray:
    """log10(envelope / channel median): multiplicative gain differences become offsets."""
    v = np.asarray(env, dtype=np.float64)
    med = np.median(v, axis=0, keepdims=True)
    return np.log10((v + EPS) / (med + EPS))


def log_compress_and_clip(img: np.ndarray, clip_percentile: float, clip_low: float | None = 0.0
                          ) -> np.ndarray:
    """log1p of the positive part, clipped at the given upper percentile.

    Compresses the dynamic range so trams do not dominate the weak car lines. Used for the
    ``robust_z`` normalisation (z-scores); see ``normalize_envelope``.
    """
    v = np.log1p(np.maximum(np.asarray(img, dtype=np.float64), 0.0))
    hi = np.percentile(v, clip_percentile)
    return np.clip(v, clip_low if clip_low is not None else v.min(), hi)


def normalize_envelope(env: np.ndarray, method: str, clip_percentile: float,
                       clip_low: bool) -> np.ndarray:
    """Envelope -> normalised, log-compressed, clipped image.

    ``log_ratio``: log10(env / channel median), clipped to [0 (if clip_low), p_hi];
    ``robust_z``: log1p(max(z, 0)) of the per-channel robust z-score, clipped at p_hi.
    """
    if method == "log_ratio":
        v = log_ratio(env)
        hi = np.percentile(v, clip_percentile)
        return np.clip(v, 0.0 if clip_low else v.min(), hi)
    if method == "robust_z":
        return log_compress_and_clip(robust_normalize(env), clip_percentile, 0.0)
    raise ValueError(f"Unknown normalisation {method!r}")


def smooth_anisotropic(img: np.ndarray, sigma_t: float, sigma_x: float) -> np.ndarray:
    """Gaussian smoothing with separate widths along time (rows) and space (columns)."""
    if sigma_t <= 0 and sigma_x <= 0:
        return img
    k = lambda s: 2 * int(np.ceil(3 * s)) + 1 if s > 0 else 1  # noqa: E731
    return cv2.GaussianBlur(np.asarray(img, dtype=np.float32), (k(sigma_x), k(sigma_t)),
                            sigmaX=max(sigma_x, 0.0), sigmaY=max(sigma_t, 0.0)).astype(np.float64)


def clahe(img: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation (skimage) of a [0, max] image."""
    from skimage.exposure import equalize_adapthist

    v = np.asarray(img, dtype=np.float64)
    span = v.max() - v.min()
    return equalize_adapthist((v - v.min()) / (span if span > 0 else 1.0), clip_limit=clip_limit)


def upsample_space(img: np.ndarray, factor: int) -> np.ndarray:
    """Linear interpolation along the cable (cv2.resize) so slopes are closer to 45 deg."""
    if factor == 1:
        return np.asarray(img, dtype=np.float64)
    v = np.asarray(img, dtype=np.float32)
    return cv2.resize(v, (v.shape[1] * factor, v.shape[0]),
                      interpolation=cv2.INTER_LINEAR).astype(np.float64)


# ======================================================================================
# Full chain
# ======================================================================================
@dataclass
class Preprocessed:
    """Result of :func:`preprocess` - energy images of both bands plus all stages."""
    row_dt: float                                   # [s] per image row
    col_dx: float                                   # [m] per image column
    common: dict[str, np.ndarray] = field(default_factory=dict)
    bands: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def images(self) -> dict[str, np.ndarray]:
        """Final image of every band, e.g. {'hf': (rows, C), 'lf': (rows, C)}."""
        return {name: st["image"] for name, st in self.bands.items()}


def _crop_rows(img: np.ndarray, crop: tuple[int, int], factor: int) -> np.ndarray:
    b, a = crop
    if b % factor or a % factor:
        raise ValueError(f"Padding {crop} not divisible by the decimation factor {factor}")
    return img[b // factor:img.shape[0] - a // factor]


def band_definitions(params: PreprocessingParams) -> dict[str, tuple[float, float]]:
    return {"hf": (params.hf_low, params.hf_high), "lf": (params.lf_low, params.lf_high)}


def preprocess(x: np.ndarray, params: PreprocessingParams, fs: float, dx: float,
               crop: tuple[int, int] = (0, 0), bad_channels: np.ndarray | None = None,
               bands: tuple[str, ...] = ("hf", "lf")) -> Preprocessed:
    """Full Stage-4 chain for the requested bands.

    ``x`` is the (padded) raw matrix; ``crop`` = (pad_before, pad_after) in samples is
    removed after the envelope. Every intermediate image (one row per 0.04 s) is stored:

    - common: ``raw`` (RMS of the raw signal), ``common_mode`` (RMS of the removed
      common-mode signal, repeated over channels), ``after_common_mode``;
    - per band: ``envelope`` (RMS of the band-passed signal), ``removed_by_bandpass``
      (RMS of what the band-pass removed), ``normalized`` and the final ``image``.
    """
    f = params.time_decimation
    env = lambda s: _crop_rows(envelope_downsample(s, f, params.envelope), crop, f)  # noqa: E731
    out = Preprocessed(row_dt=f / fs, col_dx=dx)

    xd = detrend_channels(x) if params.detrend else np.asarray(x)
    if params.common_mode == "none":
        xc, cm = xd, np.zeros(xd.shape[0], dtype=xd.dtype)
    else:
        cm = common_mode_signal(xd, params.common_mode)
        xc = (xd - cm[:, None]).astype(xd.dtype, copy=False)
    out.common = {
        "raw": env(x),
        "common_mode": np.repeat(env(cm[:, None]), x.shape[1], axis=1),
        "after_common_mode": env(xc),
    }
    bad = None if bad_channels is None or not params.interpolate_bad_channels else bad_channels
    for name in bands:
        lo, hi = band_definitions(params)[name]
        taper = min(params.taper_hz, 0.5 * lo) if lo > 0 else params.taper_hz
        xb = bandpass_fft(xc, fs, lo, hi, taper_hz=taper)
        e = env(xb)
        if bad is not None:
            e = interpolate_channels(e, bad)
        norm = normalize_envelope(e, params.normalization, params.clip_percentile,
                                  params.clip_low)
        img = smooth_anisotropic(norm, params.smooth_sigma_t, params.smooth_sigma_x)
        if params.clahe:
            img = clahe(img)
        out.bands[name] = {"envelope": e, "removed_by_bandpass": env(xc - xb),
                           "normalized": norm, "image": img}
    return out


def preprocess_segment(seg, params: PreprocessingParams, **kwargs) -> Preprocessed:
    """:func:`preprocess` for an ``io.Segment`` (padding is cropped automatically)."""
    return preprocess(seg.data, params, fs=1.0 / seg.dt, dx=seg.dx,
                      crop=(seg.pad_before, seg.pad_after), **kwargs)
