"""Exploratory statistics of the whole dataset and of segments (plan: Etap 2)."""
from __future__ import annotations

import numpy as np


def describe(x: np.ndarray) -> dict:
    """min, max, mean, std, median, MAD, percentiles, skewness, kurtosis."""
    raise NotImplementedError


def per_channel_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    """Statistics along time for every channel (detects dead / noisy channels)."""
    raise NotImplementedError


def per_time_stats(x: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Windowed statistics along time (RMS, std, max) averaged over channels."""
    raise NotImplementedError


class RunningStats:
    """Streaming (Welford) mean/var + fixed-bin histogram over all 259 files."""

    def __init__(self, n_channels: int, bins: np.ndarray):
        raise NotImplementedError

    def update(self, x: np.ndarray) -> None:
        raise NotImplementedError

    def result(self) -> dict:
        raise NotImplementedError


def detect_bad_channels(ch_stats: dict, z_thresh: float = 3.5) -> np.ndarray:
    """Boolean mask of outlier channels (robust z-score of channel RMS)."""
    raise NotImplementedError
