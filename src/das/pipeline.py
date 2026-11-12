"""End-to-end pipeline: raw (time, space) matrix -> detected objects (plan: Stage 12).

``run_pipeline`` chains Stages 4 -> 5 -> 6A/6B -> 7 -> 8/9 -> 10 (preprocessing,
enhancement, strong + weak detection, grouping, tracking with velocities and v(t), object
features) and times every stage with ``time.perf_counter``. ``save_result`` writes the
object table (CSV), a JSON summary (parameters, timings, counts) and optionally a cache
of the intermediate images (``.npy``) read back by ``load_cache``.

Clustering (Stage 10) needs many objects, so it runs over the objects of a whole run
(``clustering.cluster_objects``, see ``scripts/run_pipeline.py``), not per segment.
"""
from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .config import PipelineParams, TrackingParams
from .detection import Detections, detect_candidates
from .enhancement import Enhanced, enhance
from .features import FEATURE_NAMES, feature_context, object_features
from .grouping import DetectedObject, build_objects
from .preprocessing import Preprocessed, preprocess
from .tracking import track_objects
from .velocity import TABLE_FIELDS, objects_table

STAGE_TIMERS = ("preprocess_s", "enhance_s", "detect_s", "group_s", "track_s", "features_s")
# features added to the object table (the motion ones are already in TABLE_FIELDS)
TABLE_FEATURES = ["peak_z", "area_zs", "n_sublines", "tail_asym", "z_hf", "z_lf",
                  "spec_centroid_hz", "lf_hf_db", "snr_lf_db", "snr_hf_db", "dv_rel",
                  "abs_accel_ms2", "duration_s", "extent_m", "n_channels"]


@dataclass
class PipelineResult:
    """Output of ``run_pipeline``: detected objects, optional intermediate images, timings."""
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    objects: list[DetectedObject] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    row_dt: float = 0.0
    col_dx: float = 0.0                 # column spacing of the response maps [m]
    channel_dx: float = 0.0             # cable channel spacing [m]
    pre: Preprocessed | None = None
    enhanced: dict[str, Enhanced] = field(default_factory=dict)
    detections: Detections | None = None
    groups: list = field(default_factory=list)
    features: list[dict] = field(default_factory=list)     # Stage 10, one dict per object
    clusters: list[str] = field(default_factory=list)      # cluster name per object (optional)

    @property
    def x_ref_m(self) -> float:
        """Reference position of t_ref: the middle of the observed cable section."""
        return 0.5 * next(iter(self.enhanced.values())).response[1].shape[1] * self.col_dx

    @property
    def counts(self) -> dict[str, int]:
        """Number of objects: total, strong / weak per direction, speed classes."""
        out = {"n_objects": len(self.objects)}
        for kind in ("strong", "weak"):
            for d, dn in ((1, "pos"), (-1, "neg")):
                out[f"{kind}_{dn}"] = sum(o.kind == kind and o.direction == d
                                          for o in self.objects)
        for c in ("constant", "varying"):
            out[f"speed_{c}"] = sum(o.speed_class == c for o in self.objects)
        return out


def run_pipeline(x: np.ndarray, params: PipelineParams, fs: float, dx: float,
                 keep_stages: bool = True, crop: tuple[int, int] = (0, 0),
                 features: bool = True, pre: Preprocessed | None = None,
                 enhanced: dict[str, Enhanced] | None = None,
                 detections: Detections | None = None) -> PipelineResult:
    """preprocess -> enhance (per band and direction) -> strong + weak detection ->
    grouping -> tracking / velocities / v(t) -> object features. Every stage is timed
    (time.perf_counter); ``timings["total_s"]`` is their sum.

    ``crop`` = (pad_before, pad_after) samples removed after the envelope (padded
    segments); the spectral features use the raw samples of the cropped part. With
    ``keep_stages`` False the large intermediate images are dropped at the end (after
    tracking and features, which need them). ``features`` False skips Stage 10.

    ``pre`` / ``enhanced`` / ``detections``: results of earlier stages computed with the
    *same* upstream parameters, reused instead of recomputed (parameter studies, Stage
    13); a reused stage gets a timing of 0. They are not modified (with ``keep_stages``
    False pass copies).
    """
    res = PipelineResult(channel_dx=dx)
    t = time.perf_counter()
    res.pre = pre if pre is not None else preprocess(x, params.pre, fs, dx, crop=crop)
    res.timings["preprocess_s"] = time.perf_counter() - t
    t = time.perf_counter()
    res.enhanced = enhanced if enhanced is not None else {
        b: enhance(img, params.enh, res.pre.row_dt, res.pre.col_dx)
        for b, img in res.pre.images.items()}
    res.timings["enhance_s"] = time.perf_counter() - t
    e = next(iter(res.enhanced.values()))
    res.row_dt, res.col_dx = e.row_dt, e.col_dx
    t = time.perf_counter()
    res.detections = detections if detections is not None else detect_candidates(
        res.enhanced, params.det, (params.enh.v_min_kmh, params.enh.v_max_kmh))
    res.timings["detect_s"] = time.perf_counter() - t
    t = time.perf_counter()
    res.objects, res.groups = build_objects(res.detections.segments, params.grp, res.row_dt,
                                            res.col_dx, res.x_ref_m, dx)
    res.timings["group_s"] = time.perf_counter() - t
    t = time.perf_counter()
    track_objects(res.objects, res.enhanced, params.trk, res.row_dt)
    res.timings["track_s"] = time.perf_counter() - t
    if features:
        t = time.perf_counter()
        ctx = feature_context(res.enhanced, res.row_dt, params.feat, x, fs, crop[0])
        res.features = [object_features(o, ctx, params.feat) for o in res.objects]
        res.timings["features_s"] = time.perf_counter() - t
    res.timings["total_s"] = sum(res.timings.get(k, 0.0) for k in STAGE_TIMERS)
    if not keep_stages:
        res.pre.common, res.detections.stages = {}, {}
        for en in res.enhanced.values():
            en.stages = {}
    return res


def run_segment(seg, params: PipelineParams, keep_stages: bool = True,
                features: bool = True) -> PipelineResult:
    """:func:`run_pipeline` for an ``io.Segment`` (padding cropped automatically)."""
    return run_pipeline(seg.data, params, 1.0 / seg.dt, seg.dx, keep_stages,
                        crop=(seg.pad_before, seg.pad_after), features=features)


def retrack(result: PipelineResult, params: TrackingParams) -> list[DetectedObject]:
    """Copies of the objects re-tracked with other tracking parameters (variant studies)."""
    objs = copy.deepcopy(result.objects)
    return track_objects(objs, result.enhanced, params, result.row_dt)


# ======================================================================================
# Saving
# ======================================================================================
def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def result_table(result: PipelineResult, t0: datetime | None = None) -> list[dict]:
    """Object table: Stage 7-9 fields (``velocity.objects_table``) + Stage-10 features
    + cluster name (if clustered)."""
    rows = objects_table(result.objects, t0)
    for k, row in enumerate(rows):
        f = result.features[k] if k < len(result.features) else {}
        for name in TABLE_FEATURES:
            v = f.get(name, float("nan"))
            row[name] = "" if not np.isfinite(v) else round(float(v), 4)
        row["cluster"] = result.clusters[k] if k < len(result.clusters) else ""
    return rows


def stage_images(result: PipelineResult) -> dict[str, np.ndarray]:
    """Intermediate images worth caching: raw log RMS, Stage-4 band images, f-k maps per
    direction, filter-bank responses, Stage-6 masks (whatever is still kept)."""
    out = {}
    if result.pre is not None:
        if "raw" in result.pre.common:
            out["raw_log_rms"] = np.log10(result.pre.common["raw"] + 1e-12)
        for b, img in result.pre.images.items():
            out[f"image_{b}"] = img
    for b, e in result.enhanced.items():             # names: <stage>_<band>_<direction>
        for d, dn in ((1, "pos"), (-1, "neg")):
            if f"fk_{dn}" in e.stages:
                out[f"fk_{b}_{dn}"] = e.stages[f"fk_{dn}"]
            out[f"response_{b}_{dn}"] = e.response[d]
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def save_result(result: PipelineResult, name: str, t0: datetime | None = None,
                params: PipelineParams | None = None, out_dir: Path | str | None = None,
                cache_dir: Path | str | None = None, extra: dict | None = None
                ) -> dict[str, Path]:
    """Write the result of one recording.

    - ``<out_dir>/pipeline_<name>_objects.csv``: one row per object (csv module);
    - ``<out_dir>/pipeline_<name>.json``: axes, timings, counts, parameters, ``extra``;
    - with ``cache_dir``: ``<cache_dir>/<name>/<image>.npy`` for every image of
      :func:`stage_images` (read back with :func:`load_cache`).

    Returns {"csv": path, "json": path[, "cache": dir]}.
    """
    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out_dir / f"pipeline_{name}_objects.csv",
             "json": out_dir / f"pipeline_{name}.json"}
    with open(paths["csv"], "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_FIELDS + TABLE_FEATURES + ["cluster"])
        w.writeheader()
        w.writerows(result_table(result, t0))
    summary = {
        "name": name, "start": t0, "row_dt_s": result.row_dt, "col_dx_m": result.col_dx,
        "channel_dx_m": result.channel_dx,
        "image_shape": list(next(iter(result.pre.images.values())).shape) if result.pre else None,
        "timings_s": result.timings, "counts": result.counts,
        "clusters": {c: result.clusters.count(c) for c in sorted(set(result.clusters))},
        "params": asdict(params) if params is not None else None,
        "feature_names": FEATURE_NAMES, **(extra or {}),
    }
    if cache_dir is not None:
        d = Path(cache_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        imgs = stage_images(result)
        for k, img in imgs.items():
            np.save(d / f"{k}.npy", img)
        summary["cache"] = {"dir": d, "images": sorted(imgs)}
        paths["cache"] = d
    paths["json"].write_text(json.dumps(_jsonable(summary), indent=1), encoding="utf-8")
    return paths


def load_cache(name: str, cache_dir: Path | str | None = None) -> dict[str, np.ndarray]:
    """Cached images of :func:`save_result` ({image name: array}); {} if not cached."""
    d = Path(cache_dir or config.CACHE_DIR) / name
    if not d.is_dir():
        return {}
    return {p.stem: np.load(p) for p in sorted(d.glob("*.npy"))}
