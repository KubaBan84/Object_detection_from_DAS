"""Quantitative evaluation: synthetic ground truth and manual real-data labels (plan: Etap 11)."""
from __future__ import annotations

import numpy as np

from .grouping import DetectedObject


def gt_objects_from_mask(gt: np.ndarray, meta: dict, dt: float, dx: float) -> list[dict]:
    """For each marker: points (x, t), LSQ line -> true velocity, t_ref, type, direction."""
    raise NotImplementedError


def match_detections(detections: list[DetectedObject], references: list[dict],
                     tol_t_s: float, tol_v_kmh: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by normalised (dt_ref, dv) cost with gating."""
    raise NotImplementedError


def detection_metrics(n_det: int, n_ref: int, matches: list[tuple[int, int]]) -> dict:
    """TP, FP, FN, precision, recall, F1."""
    raise NotImplementedError


def velocity_errors(detections: list[DetectedObject], references: list[dict],
                    matches: list[tuple[int, int]]) -> dict:
    """MAE, RMSE, MAPE of velocity [km/h] over matched pairs."""
    raise NotImplementedError


def trajectory_iou(det_mask: np.ndarray, gt_mask: np.ndarray, tol_px: int = 3) -> float:
    raise NotImplementedError
