"""Velocity computation and unit conversion (plan: Etap 8)."""
from __future__ import annotations

import numpy as np


def slope_to_velocity(d_col: float, d_row: float, row_dt: float, col_dx: float) -> float:
    """Pixel slope -> velocity [m/s]; sign = direction along the cable."""
    raise NotImplementedError


def ms_to_kmh(v: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(v) * 3.6


def fit_constant_velocity(track: np.ndarray) -> tuple[float, float, float]:
    """Least-squares x(t) = v t + b on tracked points -> (v [m/s], b, R^2)."""
    raise NotImplementedError
