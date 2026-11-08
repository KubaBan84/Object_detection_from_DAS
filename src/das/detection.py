"""Line detection: strong (thick) path and weak (thin) path (plan: Stage 6)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectionParams


@dataclass
class LineSegment:
    """Line segment in image pixel coordinates (col = space, row = time)."""
    c0: float
    r0: float
    c1: float
    r1: float
    score: float
    source: str            # "strong" | "weak" | "hough_global"
    direction: int         # +1 / -1


def detect_strong_lines(img: np.ndarray, params: DetectionParams
                        ) -> tuple[list[LineSegment], np.ndarray]:
    """Otsu threshold -> oriented morphological opening -> connected components ->
    regionprops / RANSAC line fit. Returns segments and binary mask of strong objects."""
    raise NotImplementedError


def detect_weak_lines(response: np.ndarray, strong_mask: np.ndarray,
                      params: DetectionParams) -> list[LineSegment]:
    """Mask strong objects -> hysteresis threshold -> skeletonize -> cv2.HoughLinesP
    restricted to the allowed angle range."""
    raise NotImplementedError


def hough_global(response: np.ndarray, angles: np.ndarray, n_peaks: int = 50
                 ) -> list[LineSegment]:
    """Complementary path: skimage hough_line + hough_line_peaks (whole-image lines)."""
    raise NotImplementedError


def robust_noise_sigma(img: np.ndarray) -> float:
    """1.4826 * MAD of the image - basis for data-driven thresholds."""
    raise NotImplementedError
