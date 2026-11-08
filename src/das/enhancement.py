"""Enhancement of slanted lines and suppression of everything else (plan: Etap 5)."""
from __future__ import annotations

import numpy as np

from .config import EnhancementParams


def fk_velocity_filter(img: np.ndarray, dt: float, dx: float, v_min: float, v_max: float,
                       direction: int, taper: float = 0.2) -> np.ndarray:
    """Keep only energy with apparent velocity v = f/k in [v_min, v_max] [m/s] moving in
    `direction` (+1 increasing / -1 decreasing cable position). Tapered fan mask."""
    raise NotImplementedError


def velocity_to_angle(v_ms: float, row_dt: float, col_dx: float) -> float:
    """Velocity [m/s] -> line angle in image pixel coordinates [rad]."""
    raise NotImplementedError


def oriented_line_kernels(length: int, angles: np.ndarray, width_sigma: float = 1.5
                          ) -> list[np.ndarray]:
    """Zero-mean, unit-norm matched line kernels (one per angle)."""
    raise NotImplementedError


def directional_filter_bank(img: np.ndarray, kernels: list[np.ndarray]
                            ) -> tuple[np.ndarray, np.ndarray]:
    """cv2.filter2D for every kernel -> (max response, argmax angle index)."""
    raise NotImplementedError


def subtract_background(img: np.ndarray, kernel_px: int) -> np.ndarray:
    """White top-hat / median background removal."""
    raise NotImplementedError


def ridge_filter(img: np.ndarray, sigmas: tuple[float, ...] = (1, 2, 3)) -> np.ndarray:
    """Alternative path: skimage.filters.sato / meijering (Hessian ridge detector)."""
    raise NotImplementedError


def enhance(img: np.ndarray, params: EnhancementParams, row_dt: float, col_dx: float
            ) -> dict[str, np.ndarray]:
    """Per-direction enhanced images + intermediate stages."""
    raise NotImplementedError
