"""Noise removal and conversion of the raw signal into an energy image (plan: Stage 4).

Chain (every step is stored for the before / after figures of the report)::

    raw (T, C) --detrend--> --common-mode removal--> x_cm
    x_cm --band-pass 60-312.5 Hz (FFT, zero phase)--> x_band
    x_band --RMS envelope of 25-sample blocks--> env (one row per 0.04 s)
    env --log10(env / channel median), clipped--> image

The pass band was chosen in Stage 3 (``scripts/run_spectral.py``). The image has one row
per 0.04 s and one column per channel; the spatial upsampling is done in Stage 5 after
the f-k filter (which needs the true channel spacing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PreprocessingParams
from .stats import block_rms

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


def common_mode_signal(x: np.ndarray) -> np.ndarray:
    """Across-channel median at every time sample -> (T,).

    A disturbance recorded at the same moment along the whole fibre (horizontal stripe,
    infinite apparent velocity) is the same in every channel, so the median keeps it,
    while a vehicle that occupies only a few channels does not change the median.
    """
    return np.median(x, axis=1)


def remove_common_mode(x: np.ndarray) -> np.ndarray:
    """Subtract the across-channel median at every time sample. Returns x.dtype."""
    return (x - common_mode_signal(x)[:, None]).astype(x.dtype, copy=False)


def bandpass_response(freqs: np.ndarray, f_low: float, f_high: float, taper_hz: float = 0.5
                      ) -> np.ndarray:
    """Real, non-negative frequency response with raised-cosine (Hann) edges.

    The gain is 0.5 exactly at ``f_low`` and ``f_high``; each edge is ``taper_hz`` wide.
    ``f_low <= 0`` gives a low-pass, ``f_high >= Nyquist`` a high-pass.
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
    return h


def bandpass_fft(x: np.ndarray, fs: float, f_low: float, f_high: float,
                 taper_hz: float = 0.5) -> np.ndarray:
    """Zero-phase band-pass of every channel in the frequency domain (rfft along time).

    The response is real, so there is no phase shift and trajectories are not moved in
    time. The circular wrap-around of the FFT affects only the segment edges; segments
    are therefore loaded with one file of padding on each side and cropped after the
    envelope. Returns x.dtype.
    """
    x = np.asarray(x)
    spec = np.fft.rfft(np.asarray(x, dtype=np.float64), axis=0)
    h = bandpass_response(np.fft.rfftfreq(x.shape[0], 1.0 / fs), f_low, f_high, taper_hz)
    spec *= h.reshape((-1,) + (1,) * (x.ndim - 1))
    return np.fft.irfft(spec, n=x.shape[0], axis=0).astype(x.dtype, copy=False)


def envelope_downsample(x: np.ndarray, factor: int, method: str = "rms") -> np.ndarray:
    """Energy envelope and decimation in one step -> (T // factor, C).

    ``rms``: RMS of blocks of ``factor`` samples (default). ``decimate``: |x| of every
    ``factor``-th sample, without an envelope (ablation: it aliases the oscillating signal).
    """
    if method == "rms":
        return block_rms(x, factor)
    if method == "decimate":
        n = (x.shape[0] // factor) * factor
        return np.abs(np.asarray(x[:n:factor], dtype=np.float64))
    raise ValueError(f"Unknown envelope method {method!r}")


# ======================================================================================
# Image operations (rows = 0.04 s, columns = channels)
# ======================================================================================
def log_ratio(env: np.ndarray) -> np.ndarray:
    """log10(envelope / channel median).

    Dividing by the channel median removes the different sensitivity (gain) of the
    channels; the logarithm compresses the dynamic range, so trams do not dominate the
    image.
    """
    v = np.asarray(env, dtype=np.float64)
    med = np.median(v, axis=0, keepdims=True)
    return np.log10((v + EPS) / (med + EPS))


def normalize_envelope(env: np.ndarray, method: str, clip_percentile: float,
                       clip_low: bool) -> np.ndarray:
    """Envelope -> normalised, clipped image.

    ``log_ratio``: log10(env / channel median); ``ratio``: env / channel median - 1,
    without log compression (ablation). Both are clipped to [0 (if ``clip_low``),
    the ``clip_percentile``-th percentile].
    """
    if method == "log_ratio":
        v = log_ratio(env)
    elif method == "ratio":
        v = np.asarray(env, dtype=np.float64)
        v = (v + EPS) / (np.median(v, axis=0, keepdims=True) + EPS) - 1.0
    else:
        raise ValueError(f"Unknown normalisation {method!r}")
    return np.clip(v, 0.0 if clip_low else v.min(), np.percentile(v, clip_percentile))


def upsample_space(img: np.ndarray, factor: int) -> np.ndarray:
    """Linear interpolation along the cable (cv2.resize), so line slopes are closer to 45 deg."""
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
    """Result of :func:`preprocess`: the energy image and every intermediate step.

    ``stages`` (all at image resolution, rows = ``row_dt``): ``raw`` (RMS of the raw
    signal), ``common_mode`` (RMS of the removed common-mode signal), ``after_common_mode``,
    ``removed_by_bandpass`` (RMS of what the band-pass removed), ``envelope`` (RMS of the
    band-passed signal) and ``image`` (the final normalised image).
    """
    row_dt: float                                   # [s] per image row
    col_dx: float                                   # [m] per image column
    stages: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def image(self) -> np.ndarray:
        return self.stages["image"]


def _crop_rows(img: np.ndarray, crop: tuple[int, int], factor: int) -> np.ndarray:
    b, a = crop
    if b % factor or a % factor:
        raise ValueError(f"Padding {crop} not divisible by the decimation factor {factor}")
    return img[b // factor:img.shape[0] - a // factor]


def preprocess(x: np.ndarray, params: PreprocessingParams, fs: float, dx: float,
               crop: tuple[int, int] = (0, 0)) -> Preprocessed:
    """Full Stage-4 chain.

    ``x`` is the (padded) raw matrix (time, space); ``crop`` = (pad_before, pad_after)
    in samples is removed after the envelope.
    """
    f = params.time_decimation
    env = lambda s: _crop_rows(envelope_downsample(s, f, params.envelope), crop, f)  # noqa: E731
    out = Preprocessed(row_dt=f / fs, col_dx=dx)

    xd = detrend_channels(x) if params.detrend else np.asarray(x)
    if params.common_mode == "median":
        cm = common_mode_signal(xd)
        xc = (xd - cm[:, None]).astype(xd.dtype, copy=False)
    elif params.common_mode == "none":
        cm, xc = np.zeros(xd.shape[0], dtype=xd.dtype), xd
    else:
        raise ValueError(f"Unknown common-mode method {params.common_mode!r}")
    lo, hi = params.band_low, params.band_high
    taper = min(params.taper_hz, 0.5 * lo) if lo > 0 else params.taper_hz
    xb = bandpass_fft(xc, fs, lo, hi, taper_hz=taper)
    e = env(xb)
    out.stages = {
        "raw": env(x),
        "common_mode": np.repeat(env(cm[:, None]), x.shape[1], axis=1),
        "after_common_mode": env(xc),
        "removed_by_bandpass": env(xc - xb),
        "envelope": e,
        "image": normalize_envelope(e, params.normalization, params.clip_percentile,
                                    params.clip_low),
    }
    return out


def preprocess_segment(seg, params: PreprocessingParams) -> Preprocessed:
    """:func:`preprocess` for an ``io.Segment`` (the padding is cropped automatically)."""
    return preprocess(seg.data, params, fs=1.0 / seg.dt, dx=seg.dx,
                      crop=(seg.pad_before, seg.pad_after))
