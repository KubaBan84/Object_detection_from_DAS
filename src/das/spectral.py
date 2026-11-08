"""Frequency analysis: per-channel FFT, Welch PSD, STFT, f-k spectrum (plan: Stage 3).

Implemented with numpy.fft only (scipy is not on the allowed library list).
"""
from __future__ import annotations

import numpy as np


def channel_fft(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """rfft applied independently to every channel (axis=0) -> (freqs, |X|)."""
    raise NotImplementedError


def welch_psd(x: np.ndarray, fs: float, nperseg: int = 1024, overlap: float = 0.5
              ) -> tuple[np.ndarray, np.ndarray]:
    """Averaged periodogram per channel (Hann window, sliding_window_view)."""
    raise NotImplementedError


def stft(x1d: np.ndarray, fs: float, nperseg: int = 512, overlap: float = 0.75
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (freqs, times, |Z|) for one channel; vectorised framing."""
    raise NotImplementedError


def band_energy(x: np.ndarray, fs: float, bands: list[tuple[float, float]]) -> np.ndarray:
    """Energy per (band, channel) - used to choose the band-pass range."""
    raise NotImplementedError


def fk_spectrum(img: np.ndarray, dt: float, dx: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D FFT of a (time, space) image -> (f, k, |F|) with fftshift."""
    raise NotImplementedError
