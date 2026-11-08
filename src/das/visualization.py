"""All plotting helpers (time on the vertical axis, space horizontal - as in the brief)."""
from __future__ import annotations

import numpy as np


def plot_waterfall(img, t0, row_dt, col_dx, x0_m=0.0, ax=None, title="", cmap="viridis",
                   vmin=None, vmax=None):
    """imshow with physical extent, HH:MM:SS time ticks, colorbar."""
    raise NotImplementedError


def plot_stages(stages: dict[str, np.ndarray], row_dt: float, col_dx: float, path=None):
    """Grid of before/after images for every processing step (+ removed component)."""
    raise NotImplementedError


def plot_detections(img, objects, row_dt, col_dx, t0, ax=None, path=None):
    """Overlay red lines + velocity labels [km/h] (strong vs weak in different styles)."""
    raise NotImplementedError


def plot_spectra(freqs, amp, path=None):
    raise NotImplementedError


def plot_stft(freqs, times, Z, path=None):
    raise NotImplementedError


def plot_fk(f, k, F, v_lines_kmh=(), path=None):
    raise NotImplementedError


def plot_velocity_profiles(objects, path=None):
    raise NotImplementedError


def plot_clusters(X2d, labels, names, path=None):
    raise NotImplementedError


def save_figure(fig, path, dpi: int = 120) -> None:
    raise NotImplementedError
