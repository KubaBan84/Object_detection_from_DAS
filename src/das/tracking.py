"""Ridge tracking of each object and velocity-over-time profile (plan: Etap 9, extra 1)."""
from __future__ import annotations

import numpy as np

from .config import TrackingParams
from .grouping import DetectedObject


def track_ridge(energy: np.ndarray, obj: DetectedObject, params: TrackingParams,
                row_dt: float, col_dx: float) -> np.ndarray:
    """For each channel search the energy maximum around the predicted time
    (vectorised, parabolic sub-sample refinement) -> (N, 2) [x_m, t_s]."""
    raise NotImplementedError


def fit_trajectory_ransac(track: np.ndarray, residual_s: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """skimage.measure.ransac(LineModelND) -> (model params, inlier mask)."""
    raise NotImplementedError


def velocity_profile(track: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Local slope dx/dt in a sliding window -> (t [s], v [km/h])."""
    raise NotImplementedError


def speed_variability(track: np.ndarray) -> dict:
    """Linear vs quadratic fit residuals, std of v(t), acceleration estimate."""
    raise NotImplementedError
