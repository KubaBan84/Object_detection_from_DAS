"""Grouping of line candidates into objects and averaging of each group (plan: Etap 7)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GroupingParams
from .detection import LineSegment


@dataclass
class DetectedObject:
    obj_id: int
    velocity_kmh: float               # signed: + towards increasing cable position
    t_ref_s: float                    # time when the object crosses the reference position
    x_start_m: float
    x_end_m: float
    t_start_s: float
    t_end_s: float
    kind: str                         # "strong" | "weak"
    score: float
    segments: list[LineSegment] = field(default_factory=list)
    track: np.ndarray | None = None   # (N, 2) [x_m, t_s] from tracking


def segment_to_params(seg: LineSegment, row_dt: float, col_dx: float, x_ref_m: float
                      ) -> tuple[float, float]:
    """-> (velocity [km/h], t at x_ref [s])."""
    raise NotImplementedError


def group_segments(segments: list[LineSegment], params: GroupingParams, row_dt: float,
                   col_dx: float) -> list[list[LineSegment]]:
    """Single-linkage grouping in (velocity, t_ref) space + collinearity/overlap check."""
    raise NotImplementedError


def merge_group(group: list[LineSegment], obj_id: int, row_dt: float, col_dx: float
                ) -> DetectedObject:
    """Score/length-weighted average line (or refit on all endpoints)."""
    raise NotImplementedError


def remove_duplicates(objects: list[DetectedObject], params: GroupingParams
                      ) -> list[DetectedObject]:
    """Remove strong/weak duplicates of the same vehicle (keep the stronger one)."""
    raise NotImplementedError
