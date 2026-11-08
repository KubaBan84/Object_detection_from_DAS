"""Per-object descriptors used for clustering (plan: Stage 10, extra 2)."""
from __future__ import annotations

import numpy as np

from .grouping import DetectedObject


def line_thickness(energy: np.ndarray, obj: DetectedObject, row_dt: float) -> float:
    """FWHM [s] of the profile perpendicular to the trajectory, averaged over channels."""
    raise NotImplementedError


def object_features(obj: DetectedObject, energy: np.ndarray, raw: np.ndarray,
                    fs: float, row_dt: float, col_dx: float) -> dict[str, float]:
    """|v|, direction, v std, curvature, R^2, thickness, peak/mean energy, duration,
    spatial extent, spectral centroid and low/high band ratio of raw signal on track."""
    raise NotImplementedError


def features_matrix(feature_dicts: list[dict[str, float]], keys: list[str]) -> np.ndarray:
    raise NotImplementedError
