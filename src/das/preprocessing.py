"""Noise removal and conversion of the raw signal into an energy image (plan: Stage 4)."""
from __future__ import annotations

import numpy as np

from .config import PreprocessingParams


def detrend_channels(x: np.ndarray) -> np.ndarray:
    """Remove per-channel mean / linear trend."""
    raise NotImplementedError


def remove_common_mode(x: np.ndarray, method: str = "median") -> np.ndarray:
    """Subtract the across-channel median at every time sample (horizontal stripes)."""
    raise NotImplementedError


def bandpass_fft(x: np.ndarray, fs: float, f_low: float, f_high: float,
                 taper_hz: float = 0.5) -> np.ndarray:
    """Zero-phase band-pass in the frequency domain with raised-cosine edges."""
    raise NotImplementedError


def envelope_downsample(x: np.ndarray, factor: int, method: str = "rms") -> np.ndarray:
    """Block RMS (or mean |x|) over `factor` samples -> decimated energy image."""
    raise NotImplementedError


def robust_normalize(img: np.ndarray, axis: int = 0) -> np.ndarray:
    """(x - median) / (1.4826 * MAD) per channel (vertical stripes / gain differences)."""
    raise NotImplementedError


def log_compress_and_clip(img: np.ndarray, clip_percentile: float) -> np.ndarray:
    raise NotImplementedError


def upsample_space(img: np.ndarray, factor: int) -> np.ndarray:
    """cv2.resize along the cable so that typical slopes are close to 45 deg."""
    raise NotImplementedError


def preprocess(x: np.ndarray, params: PreprocessingParams, fs: float
               ) -> dict[str, np.ndarray]:
    """Full chain; returns every intermediate stage (for report figures)."""
    raise NotImplementedError
