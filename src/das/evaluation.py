"""Quantitative evaluation: synthetic ground truth and manual real-data labels (plan: Stage 11).

Pixel-level helpers (``gt_envelope_mask``, ``pixel_auc``, ``enhancement_auc``) are used
already in Stages 4-5 to compare preprocessing / enhancement variants on synthetic data.
"""
from __future__ import annotations

import cv2
import numpy as np

from .grouping import DetectedObject

HEAVY_TYPES = ("tram", "truck")       # thick lines
LIGHT_TYPES = ("car", "van")          # thin lines


# ======================================================================================
# Pixel-level evaluation of enhanced images (Stages 4-5)
# ======================================================================================
def gt_envelope_mask(gt: np.ndarray, factor: int, markers: list[int] | None = None,
                     upsample: int = 1) -> np.ndarray:
    """Ground-truth marker mask at the image resolution.

    (T, C) marker array -> boolean (T // factor, C * upsample): a cell is True if any of
    its ``factor`` samples carries one of ``markers`` (all vehicles if None).
    """
    n = (gt.shape[0] // factor) * factor
    sel = gt[:n] > 0 if markers is None else np.isin(gt[:n], markers)
    m = sel.reshape(n // factor, factor, gt.shape[1]).any(axis=1)
    return np.repeat(m, upsample, axis=1) if upsample > 1 else m


def pixel_auc(score: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    """Area under the ROC curve of ``score`` for positive vs negative pixels.

    Mann-Whitney U statistic computed from ranks (vectorised, ties broken by order -
    scores are continuous). 0.5 = no separation, 1.0 = perfect.
    """
    pos, neg = score[positive], score[negative]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.empty(pos.size + neg.size)
    ranks[np.argsort(np.concatenate([pos, neg]), kind="stable")] = np.arange(1, ranks.size + 1)
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def enhancement_auc(score: np.ndarray, gt: np.ndarray, meta: dict, factor: int,
                    upsample: int = 1, guard_rows: int = 25, guard_cols: int = 2) -> dict:
    """AUC of an enhanced image against the synthetic ground truth, per vehicle class.

    Positives: pixels on the trajectories of the class (heavy = tram/truck, light =
    car/van, all). Negatives: pixels farther than ``guard_rows`` rows / ``guard_cols``
    channels from any vehicle (so the halo of a line is not counted as background).
    """
    types = {v["marker_value"]: v["type"] for v in meta["ground_truth"]["vehicles"]}
    any_v = gt_envelope_mask(gt, factor, upsample=1)
    se = np.ones((2 * guard_rows + 1, 2 * guard_cols + 1), np.uint8)
    near = cv2.dilate(any_v.astype(np.uint8), se).astype(bool)
    negative = np.repeat(~near, upsample, axis=1) if upsample > 1 else ~near
    out = {}
    for name, sel in (("heavy", HEAVY_TYPES), ("light", LIGHT_TYPES), ("all", None)):
        markers = None if sel is None else [m for m, t in types.items() if t in sel]
        if markers == []:
            out[name] = float("nan")
            continue
        out[name] = pixel_auc(score, gt_envelope_mask(gt, factor, markers, upsample), negative)
    return out


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
