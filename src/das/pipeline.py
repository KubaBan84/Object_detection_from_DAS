"""End-to-end pipeline: raw (time, space) matrix -> detected objects with velocities
(plan: Stage 10).

``run_pipeline`` chains Stages 4 -> 5 -> 6 -> 7 -> 8 (preprocessing, enhancement,
thick-line detection, grouping with the line velocity) and times every stage with
``time.perf_counter``. ``save_result`` writes the object table (CSV), a JSON summary
(parameters, timings, counts) and optionally a cache of the intermediate images
(``.npy``), which ``load_cache`` reads back.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .config import PipelineParams
from .detection import Detections, detect_lines
from .enhancement import Enhanced, enhance
from .grouping import DetectedObject, build_objects
from .preprocessing import Preprocessed, preprocess
from .velocity import write_objects_csv

STAGE_TIMERS = ("preprocess_s", "enhance_s", "detect_s", "group_s")


@dataclass
class PipelineResult:
    """Output of ``run_pipeline``: intermediate results, detected objects, timings."""
    pre: Preprocessed | None = None
    enhanced: Enhanced | None = None
    detections: Detections | None = None
    groups: list = field(default_factory=list)
    objects: list[DetectedObject] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    channel_dx: float = 0.0             # cable channel spacing [m]

    @property
    def row_dt(self) -> float:
        """Time per image row [s]."""
        return self.enhanced.row_dt

    @property
    def col_dx(self) -> float:
        """Column spacing of the response maps (after upsampling) [m]."""
        return self.enhanced.col_dx

    @property
    def n_channels(self) -> int:
        return self.pre.image.shape[1]

    @property
    def counts(self) -> dict[str, int]:
        """Number of objects: total and per direction."""
        return {"n_objects": len(self.objects),
                "n_pos": sum(o.direction > 0 for o in self.objects),
                "n_neg": sum(o.direction < 0 for o in self.objects)}


def run_pipeline(x: np.ndarray, params: PipelineParams, fs: float, dx: float,
                 crop: tuple[int, int] = (0, 0), pre: Preprocessed | None = None,
                 enhanced: Enhanced | None = None) -> PipelineResult:
    """preprocess -> enhance -> detect thick lines -> group (line velocity).

    ``crop`` = (pad_before, pad_after) samples removed after the envelope (padded
    segments). ``pre`` / ``enhanced``: results of earlier stages computed with the *same*
    upstream parameters, reused instead of recomputed (experiments, Stage 11); a reused
    stage gets a timing of 0. Every stage is timed; ``timings["total_s"]`` is their sum.
    """
    res = PipelineResult(channel_dx=dx)
    t = time.perf_counter()
    res.pre = pre if pre is not None else preprocess(x, params.pre, fs, dx, crop=crop)
    res.timings["preprocess_s"] = time.perf_counter() - t
    t = time.perf_counter()
    res.enhanced = enhanced if enhanced is not None else enhance(
        res.pre.image, params.enh, res.pre.row_dt, res.pre.col_dx)
    res.timings["enhance_s"] = time.perf_counter() - t
    t = time.perf_counter()
    res.detections = detect_lines(res.enhanced, params.det,
                                  (params.enh.v_min_kmh, params.enh.v_max_kmh))
    res.timings["detect_s"] = time.perf_counter() - t
    t = time.perf_counter()
    x_ref = 0.5 * res.pre.image.shape[1] * dx           # middle of the observed cable
    res.objects, res.groups = build_objects(res.detections.segments, params.grp, res.row_dt,
                                            res.col_dx, x_ref, dx)
    res.timings["group_s"] = time.perf_counter() - t
    res.timings["total_s"] = sum(res.timings[k] for k in STAGE_TIMERS)
    return res


def run_segment(seg, params: PipelineParams) -> PipelineResult:
    """:func:`run_pipeline` for an ``io.Segment`` (the padding is cropped automatically)."""
    return run_pipeline(seg.data, params, 1.0 / seg.dt, seg.dx,
                        crop=(seg.pad_before, seg.pad_after))


# ======================================================================================
# Saving
# ======================================================================================
def to_jsonable(obj):
    """Recursively convert numpy types, NaN, datetimes and paths for ``json.dumps``."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def stage_images(result: PipelineResult) -> dict[str, np.ndarray]:
    """Intermediate images worth caching: raw log RMS, Stage-4 image, f-k maps and
    filter-bank responses per direction."""
    out = {"raw_log_rms": np.log10(result.pre.stages["raw"] + 1e-12),
           "image": result.pre.image}
    for d, dn in ((1, "pos"), (-1, "neg")):
        out[f"fk_{dn}"] = result.enhanced.stages[f"fk_{dn}"]
        out[f"response_{dn}"] = result.enhanced.response[d]
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def save_result(result: PipelineResult, name: str, t0: datetime | None = None,
                params: PipelineParams | None = None, out_dir: Path | str | None = None,
                cache_dir: Path | str | None = None, extra: dict | None = None
                ) -> dict[str, Path]:
    """Write the result of one recording.

    - ``<out_dir>/pipeline_<name>_objects.csv``: one row per object;
    - ``<out_dir>/pipeline_<name>.json``: axes, timings, counts, parameters, ``extra``;
    - with ``cache_dir``: ``<cache_dir>/<name>/<image>.npy`` for every image of
      :func:`stage_images` (read back with :func:`load_cache`).

    Returns {"csv": path, "json": path[, "cache": dir]}.
    """
    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"csv": write_objects_csv(result.objects, out_dir / f"pipeline_{name}_objects.csv", t0),
             "json": out_dir / f"pipeline_{name}.json"}
    summary = {
        "name": name, "start": t0, "row_dt_s": result.row_dt, "col_dx_m": result.col_dx,
        "channel_dx_m": result.channel_dx, "image_shape": list(result.pre.image.shape),
        "timings_s": result.timings, "counts": result.counts,
        "thresholds": result.detections.thresholds,
        "params": asdict(params) if params is not None else None, **(extra or {}),
    }
    if cache_dir is not None:
        d = Path(cache_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        imgs = stage_images(result)
        for k, img in imgs.items():
            np.save(d / f"{k}.npy", img)
        summary["cache"] = {"dir": d, "images": sorted(imgs)}
        paths["cache"] = d
    paths["json"].write_text(json.dumps(to_jsonable(summary), indent=1), encoding="utf-8")
    return paths


def load_cache(name: str, cache_dir: Path | str | None = None) -> dict[str, np.ndarray]:
    """Cached images of :func:`save_result` ({image name: array}); {} if not cached."""
    d = Path(cache_dir or config.CACHE_DIR) / name
    if not d.is_dir():
        return {}
    return {p.stem: np.load(p) for p in sorted(d.glob("*.npy"))}
