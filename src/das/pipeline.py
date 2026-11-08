"""End-to-end pipeline: raw (time, space) matrix -> detected objects (plan: Stage 12)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PipelineParams
from .grouping import DetectedObject


@dataclass
class PipelineResult:
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    objects: list[DetectedObject] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    row_dt: float = 0.0
    col_dx: float = 0.0


def run_pipeline(x: np.ndarray, params: PipelineParams, fs: float, dx: float,
                 keep_stages: bool = True) -> PipelineResult:
    """preprocess -> enhance (per direction) -> strong + weak detection -> grouping ->
    velocity -> tracking -> features. Every stage timed with time.perf_counter."""
    raise NotImplementedError


def save_result(result: PipelineResult, name: str, t0) -> None:
    """objects -> CSV (csv module) + JSON with params; stages -> cache .npy."""
    raise NotImplementedError
