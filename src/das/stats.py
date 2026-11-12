"""Exploratory statistics of the whole dataset and of segments (plan: Stage 2).

All functions treat the input as a (time, space) matrix and are fully vectorised.
Statistics are computed in float64 to avoid precision loss on values of order 1e-6.
"""
from __future__ import annotations

import numpy as np

PERCENTILES = (0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)
MAD_TO_SIGMA = 1.4826  # MAD -> standard deviation for Gaussian data


# ======================================================================================
# Basic descriptive statistics
# ======================================================================================
def describe(x: np.ndarray) -> dict[str, float]:
    """Descriptive statistics of all values of ``x``.

    Returns n, mean, std, rms, min, max, max_abs, median, MAD, robust sigma (1.4826 MAD),
    percentiles ``p<q>`` (see PERCENTILES), skewness and excess kurtosis (0 for Gaussian).
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    mean = v.mean()
    d = v - mean
    m2 = np.mean(d * d)
    m3 = np.mean(d ** 3)
    m4 = np.mean(d ** 4)
    pct = np.percentile(v, PERCENTILES)
    median = pct[PERCENTILES.index(50.0)]
    mad = np.median(np.abs(v - median))
    out = {
        "n": int(v.size),
        "mean": float(mean),
        "std": float(np.sqrt(m2)),
        "rms": float(np.sqrt(np.mean(v * v))),
        "min": float(v.min()),
        "max": float(v.max()),
        "max_abs": float(np.abs(v).max()),
        "median": float(median),
        "mad": float(mad),
        "robust_sigma": float(MAD_TO_SIGMA * mad),
        "skewness": float(m3 / m2 ** 1.5) if m2 > 0 else 0.0,
        "kurtosis": float(m4 / m2 ** 2 - 3.0) if m2 > 0 else 0.0,
    }
    out.update({f"p{q:g}": float(p) for q, p in zip(PERCENTILES, pct)})
    return out


def per_channel_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    """Statistics along time for every channel -> dict of arrays of length n_channels.

    Used to find dead / noisy channels (vertical stripes in the waterfall).
    """
    v = np.asarray(x, dtype=np.float64)
    mean = v.mean(axis=0)
    d = v - mean
    m2 = np.mean(d * d, axis=0)
    safe_m2 = np.where(m2 > 0, m2, 1.0)
    median = np.median(v, axis=0)
    return {
        "mean": mean,
        "std": np.sqrt(m2),
        "rms": np.sqrt(np.mean(v * v, axis=0)),
        "median": median,
        "mad": np.median(np.abs(v - median), axis=0),
        "max_abs": np.abs(v).max(axis=0),
        "p99_abs": np.percentile(np.abs(v), 99.0, axis=0),
        "skewness": np.where(m2 > 0, np.mean(d ** 3, axis=0) / safe_m2 ** 1.5, 0.0),
        "kurtosis": np.where(m2 > 0, np.mean(d ** 4, axis=0) / safe_m2 ** 2 - 3.0, 0.0),
    }


def block_rms(x: np.ndarray, block: int) -> np.ndarray:
    """RMS over non-overlapping blocks of ``block`` samples along time.

    (T, C) -> (T // block, C); trailing samples that do not fill a block are dropped.
    """
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    n = (x.shape[0] // block) * block
    xb = np.asarray(x[:n], dtype=np.float64).reshape(n // block, block, x.shape[1])
    return np.sqrt(np.mean(xb * xb, axis=1))


def per_time_stats(x: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Windowed statistics along time.

    Returns ``rms_map`` (n_windows, n_channels) and per-window series aggregated over
    channels: ``rms_mean``, ``rms_median`` (robust to single noisy channels), ``max_abs``.
    """
    rms_map = block_rms(x, window)
    n = rms_map.shape[0] * window
    xb = np.abs(np.asarray(x[:n])).reshape(rms_map.shape[0], window, x.shape[1])
    return {
        "window": np.asarray(window),
        "rms_map": rms_map,
        "rms_mean": rms_map.mean(axis=1),
        "rms_median": np.median(rms_map, axis=1),
        "max_abs": xb.max(axis=(1, 2)).astype(np.float64),
    }


def neighbour_correlation(x: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of neighbouring channels (length C-1).

    Measures the spatial coherence of the signal along the fibre.
    """
    v = np.asarray(x, dtype=np.float64)
    v = v - v.mean(axis=0)
    std = v.std(axis=0)
    z = v / np.where(std > 0, std, 1.0)
    return np.mean(z[:, :-1] * z[:, 1:], axis=0)


def channel_correlation_matrix(x: np.ndarray) -> np.ndarray:
    """(C, C) Pearson correlation matrix between channels."""
    return np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=False)


# ======================================================================================
# Streaming statistics over the whole dataset
# ======================================================================================
def _combine_moments(na: float, a: tuple, nb: float, b: tuple) -> tuple:
    """Merge (mean, M2, M3, M4) of two samples (Chan et al. / Pebay parallel formulas).

    M_k are sums of k-th powers of deviations from the mean. Works element-wise on arrays.
    """
    mean_a, m2a, m3a, m4a = a
    mean_b, m2b, m3b, m4b = b
    n = na + nb
    delta = mean_b - mean_a
    mean = mean_a + delta * nb / n
    m2 = m2a + m2b + delta ** 2 * na * nb / n
    m3 = (m3a + m3b + delta ** 3 * na * nb * (na - nb) / n ** 2
          + 3.0 * delta * (na * m2b - nb * m2a) / n)
    m4 = (m4a + m4b + delta ** 4 * na * nb * (na * na - na * nb + nb * nb) / n ** 3
          + 6.0 * delta ** 2 * (na * na * m2b + nb * nb * m2a) / n ** 2
          + 4.0 * delta * (na * m3b - nb * m3a) / n)
    return mean, m2, m3, m4


def _moments_to_stats(n: float, mean, m2, m3, m4) -> dict:
    var = m2 / n
    safe = np.where(var > 0, var, 1.0)
    return {
        "n": n,
        "mean": mean,
        "std": np.sqrt(var),
        "rms": np.sqrt(var + mean ** 2),
        "skewness": np.where(var > 0, (m3 / n) / safe ** 1.5, 0.0),
        "kurtosis": np.where(var > 0, (m4 / n) / safe ** 2 - 3.0, 0.0),
    }


class RunningStats:
    """Streaming per-channel moments + fixed-bin histograms over many files.

    Moments are merged batch-wise with the parallel (Welford / Chan / Pebay) update, so the
    whole dataset never has to be in memory and the result is numerically stable.

    Parameters
    ----------
    n_channels : number of channels (columns).
    bins : edges of the histogram of signed values (values outside are counted
        separately as ``under`` / ``over``).
    abs_bins : edges (typically log-spaced) of the histogram of |x|.
    """

    def __init__(self, n_channels: int, bins: np.ndarray, abs_bins: np.ndarray):
        self.n_channels = n_channels
        self.n = 0
        self._mom = tuple(np.zeros(n_channels) for _ in range(4))
        self.min = np.full(n_channels, np.inf)
        self.max = np.full(n_channels, -np.inf)
        self.bins = np.asarray(bins, dtype=np.float64)
        self.abs_bins = np.asarray(abs_bins, dtype=np.float64)
        self.hist = np.zeros(len(self.bins) - 1, dtype=np.int64)
        self.abs_hist = np.zeros(len(self.abs_bins) - 1, dtype=np.int64)
        self.under = 0
        self.over = 0
        self.abs_below = 0  # |x| below the first abs bin edge (e.g. exact zeros)

    def update(self, x: np.ndarray) -> None:
        """Add one (T, n_channels) batch: merge its moments, extrema and histograms."""
        v = np.asarray(x, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != self.n_channels:
            raise ValueError(f"Expected (T, {self.n_channels}) array, got {v.shape}")
        nb = v.shape[0]
        mean_b = v.mean(axis=0)
        d = v - mean_b
        d2 = d * d
        batch = (mean_b, d2.sum(axis=0), (d2 * d).sum(axis=0), (d2 * d2).sum(axis=0))
        self._mom = batch if self.n == 0 else _combine_moments(self.n, self._mom, nb, batch)
        self.n += nb
        self.min = np.minimum(self.min, v.min(axis=0))
        self.max = np.maximum(self.max, v.max(axis=0))
        self.hist += np.histogram(v, self.bins)[0]
        self.under += int(np.count_nonzero(v < self.bins[0]))
        self.over += int(np.count_nonzero(v > self.bins[-1]))
        a = np.abs(v)
        self.abs_hist += np.histogram(a, self.abs_bins)[0]
        self.abs_below += int(np.count_nonzero(a < self.abs_bins[0]))

    def result(self) -> dict:
        """Per-channel and global statistics + histograms."""
        if self.n == 0:
            raise ValueError("No data passed to RunningStats.update()")
        per_channel = _moments_to_stats(self.n, *self._mom)
        per_channel.update(min=self.min, max=self.max)
        # Global moments: merge the channels as if they were consecutive batches.
        g = tuple(m[0] for m in self._mom)
        n_g = self.n
        for c in range(1, self.n_channels):
            g = _combine_moments(n_g, g, self.n, tuple(m[c] for m in self._mom))
            n_g += self.n
        glob = {k: float(v) for k, v in _moments_to_stats(n_g, *g).items()}
        glob.update(min=float(self.min.min()), max=float(self.max.max()))
        glob["max_abs"] = max(abs(glob["min"]), abs(glob["max"]))
        abs_pct = hist_percentiles(self.abs_bins, self.abs_hist, PERCENTILES,
                                   below=self.abs_below)
        glob.update({f"abs_p{q:g}": float(p) for q, p in zip(PERCENTILES, abs_pct)})
        return {
            "global": glob,
            "per_channel": per_channel,
            "hist": {"edges": self.bins, "counts": self.hist,
                     "under": self.under, "over": self.over},
            "abs_hist": {"edges": self.abs_bins, "counts": self.abs_hist,
                         "below": self.abs_below},
        }


def hist_percentiles(edges: np.ndarray, counts: np.ndarray, qs, below: int = 0) -> np.ndarray:
    """Approximate percentiles from a histogram (linear interpolation inside bins).

    ``below`` is the number of samples smaller than ``edges[0]`` (assigned to ``edges[0]``).
    """
    cdf = np.concatenate([[below], below + np.cumsum(counts)]).astype(np.float64)
    target = np.asarray(qs, dtype=np.float64) / 100.0 * cdf[-1]
    return np.interp(target, cdf, edges)


# ======================================================================================
# Robust outlier detection and activity measures
# ======================================================================================
def robust_zscore(v: np.ndarray) -> np.ndarray:
    """(v - median) / (1.4826 * MAD); zeros if MAD == 0."""
    v = np.asarray(v, dtype=np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        return np.zeros_like(v)
    return (v - med) / (MAD_TO_SIGMA * mad)


def detect_bad_channels(channel_rms: np.ndarray, z_thresh: float = 3.5
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Outlier channels by the robust z-score of log10(channel RMS).

    The logarithm makes the test symmetric for too quiet (dead) and too loud (noisy)
    channels. Returns (boolean mask, z-scores).
    """
    rms = np.asarray(channel_rms, dtype=np.float64)
    z = robust_zscore(np.log10(np.maximum(rms, np.finfo(np.float64).tiny)))
    return np.abs(z) > z_thresh, z


def activity_metrics(rms_windows: np.ndarray, channel_ref: np.ndarray,
                     active_factor: float = 3.0, strong_factor: float = 10.0,
                     busy_fraction: float = 0.25) -> dict[str, float]:
    """Traffic activity of a block of windowed RMS values (n_windows, n_channels).

    ``channel_ref`` is a per-channel reference level (e.g. dataset median of the windowed
    RMS), so channels with different sensitivity are comparable. A cell is *active* if its
    RMS exceeds ``active_factor`` x reference, *strong* above ``strong_factor`` x reference.
    A window is *busy* if at least ``busy_fraction`` of channels are active; ``n_events``
    counts separate busy episodes (rising edges).
    """
    ratio = np.asarray(rms_windows, dtype=np.float64) / np.asarray(channel_ref)[None, :]
    active = ratio > active_factor
    busy = active.mean(axis=1) >= busy_fraction
    n_events = int(busy[0]) + int(np.count_nonzero(busy[1:] & ~busy[:-1]))
    return {
        "active_fraction": float(active.mean()),
        "strong_fraction": float(np.mean(ratio > strong_factor)),
        "busy_fraction": float(busy.mean()),
        "n_events": n_events,
        "median_ratio": float(np.median(ratio)),
        "p99_ratio": float(np.percentile(ratio, 99.0)),
    }
