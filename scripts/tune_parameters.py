"""Stage 13: parameter studies, ablation and method comparison at the detection level.

Every variant = the final parameters (``config.PipelineParams``) with a few fields
changed. It runs Stages 4 -> 9 on all recordings and is scored at the *object* level
with the Stage-11 evaluation:

- synthetic samples: one-to-one matching with the ground truth (P / R / F1, recall of
  thick = tram / truck and thin = car / van lines, |v| MAE);
- real segments: the same against the manual labels (indicative reference, only the
  scored labels, see ``evaluation.split_references``), plus the false positives split
  into "near a label" / "no label".

Studies (``--study``, several allowed, default all):

- ``ablation``: switch off one step at a time (common-mode removal, log compression,
  f-k filter, filter bank, masking of strong lines, hysteresis, LF band, weak path,
  ridge tracking, edge margin);
- ``methods``: alternatives of single steps (Hough variants, Sato / Meijering / Gabor,
  envelope, normalisation, f-k padding, strong threshold) and two classical baselines
  (Canny + HoughLinesP on the raw / preprocessed image);
- ``sweep``: one-at-a-time grid around the defaults (band edges, decimation, kernel
  length, number of angles, upsampling, strong floor, minimal extent / support);
- ``hysteresis``: grid of the weak-path thresholds (low x high) + the sensitivity curve
  F1(threshold) with low = high / 2;
- ``grouping``: grid of the grouping tolerances (time x velocity);
- ``combined``: combinations of the candidate changes found by the other studies.

Speed: the recordings are processed in parallel (``--jobs``); inside a worker the
results of Stages 4 / 5 / 6 are cached and reused by all variants with the same
upstream parameters (``run_pipeline(pre=, enhanced=, detections=)``), so e.g. a
grouping grid costs only grouping + tracking. Timings of reused stages are taken from
the run that computed them.

The detector is selected on the *synthetic* F1 (true ground truth); the real-data
scores (rough manual labels) are a guard against changes that only fit the simulator.
We do not tune for individual lines.

Outputs: ``outputs/results/tuning_<study>.csv``, ``tuning.json``,
``outputs/figures/tuning/<study>*.png``.

Usage::

    uv run python scripts/tune_parameters.py                    # all studies (~0.5-1 h)
    uv run python scripts/tune_parameters.py --study ablation methods --jobs 4
    uv run python scripts/tune_parameters.py --quick            # 2 synthetic + 1 real

Plan: Stage 13.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das.annotation import load_labels, segment_lines  # noqa: E402
from das.config import PipelineParams  # noqa: E402
from das.detection import (Detections, LineSegment, detect_candidates,  # noqa: E402
                           sample_image, slope_bounds, slope_ok)
from das.enhancement import enhance, robust_z  # noqa: E402
from das.evaluation import (evaluate_objects, gt_objects_from_mask,  # noqa: E402
                            references_from_annotations, split_references,
                            summarize_evaluations)
from das.pipeline import run_pipeline  # noqa: E402
from das.preprocessing import preprocess, upsample_space  # noqa: E402
from das.visualization import save_figure  # noqa: E402

FIG_DIR = config.FIG_DIR / "tuning"
STUDIES = ("ablation", "methods", "sweep", "hysteresis", "grouping", "combined")
NO_TRACK = (("trk.min_points", 10 ** 6),)          # velocity of the averaged Stage-7 line
FIELDS = ["study", "variant", "changes", "syn_precision", "syn_recall", "syn_f1",
          "syn_recall_thick", "syn_recall_thin", "syn_f1_thin", "syn_fp", "syn_vel_mae",
          "real_precision", "real_recall", "real_f1", "real_recall_thick", "real_recall_thin",
          "real_fp", "real_fp_near_label", "real_fp_no_label", "real_vel_mae", "real_n_det",
          "time_s", "d_syn_f1", "d_real_f1"]


@dataclass(frozen=True)
class Variant:
    """One parameter setting of a study: changes vs the defaults, or a baseline detector."""
    study: str
    name: str
    changes: tuple = ()              # (("section.field", value), ...) vs PipelineParams()
    baseline: str = ""               # "" | "canny_raw" | "canny_pre" (own Stage-6 detector)

    @property
    def key(self) -> tuple:
        return self.changes, self.baseline


# ======================================================================================
# Variants
# ======================================================================================
def _sweep(study: str, field: str | tuple, values: list) -> list[Variant]:
    fields = field if isinstance(field, tuple) else (field,)
    out = []
    for v in values:
        vals = v if isinstance(v, tuple) else (v,)
        name = ", ".join(f"{f.split('.')[1]} = {x}" for f, x in zip(fields, vals))
        out.append(Variant(study, name, tuple(zip(fields, vals))))
    return out


def make_variants(studies: list[str]) -> list[Variant]:
    """All variants of the selected studies."""
    final = "final parameters"
    out: list[Variant] = []
    if "ablation" in studies:
        out += [Variant("ablation", final)] + [Variant("ablation", n, c) for n, c in (
            ("no common-mode removal", (("pre.common_mode", "none"),)),
            ("no log compression (linear ratio)", (("pre.normalization", "ratio"),)),
            ("no f-k fan filter", (("enh.fk_filter", False),)),
            ("no directional filter bank", (("enh.enhancer", "none"),)),
            ("no masking of strong lines", (("det.mask_strong", False),)),
            ("no hysteresis (single 4 sigma threshold)", (("det.weak_low_sigma", 4.0),)),
            ("no LF band in the weak path", (("det.weak_bands", ("hf",)),)),
            ("no weak path (strong lines only)", (("det.weak_bands", ()),)),
            ("no ridge tracking (line velocity)", NO_TRACK),
            ("no edge margin in tracking", (("trk.edge_margin_s", 0.0),)),
        )]
    if "methods" in studies:
        out += [Variant("methods", final)] + [Variant("methods", n, c, b) for n, c, b in (
            ("Hough: HoughLinesP only", (("det.use_hough_global", False),), ""),
            ("Hough: global hough_line only", (("det.use_hough_p", False),), ""),
            ("enhancer: Sato", (("enh.enhancer", "sato"),), ""),
            ("enhancer: Meijering", (("enh.enhancer", "meijering"),), ""),
            ("enhancer: Gabor", (("enh.enhancer", "gabor"),), ""),
            ("envelope: mean |x|", (("pre.envelope", "abs_mean"),), ""),
            ("envelope: plain decimation", (("pre.envelope", "decimate"),), ""),
            ("normalisation: robust z", (("pre.normalization", "robust_z"),), ""),
            ("f-k padding: reflect", (("enh.fk_pad_mode", "reflect"),), ""),
            ("f-k padding: Tukey taper", (("enh.fk_pad_mode", "tukey"),), ""),
            ("f-k padding: none (circular)", (("enh.fk_pad_mode", "none"),), ""),
            ("strong threshold: fixed 6 sigma", (("det.strong_threshold", "6"),), ""),
            ("baseline: Canny + HoughLinesP, raw image", NO_TRACK, "canny_raw"),
            ("baseline: Canny + HoughLinesP, preprocessed HF", NO_TRACK, "canny_pre"),
        )]
    if "sweep" in studies:
        out += [Variant("sweep", final)]
        for field, values in (("pre.hf_low", [30.0, 100.0]),
                              (("pre.lf_low", "pre.lf_high"), [(0.5, 5.0), (2.0, 12.0)]),
                              ("pre.time_decimation", [10, 50]),
                              ("enh.kernel_length_px", [41, 121]),
                              ("enh.n_angles", [12, 36]),
                              ("enh.space_upsample", [4]),
                              ("det.strong_min_sigma", [3.0, 5.0, 6.0]),
                              ("det.min_extent_m", [30.0, 60.0]),
                              ("grp.weak_min_support_m", [40.0, 80.0])):
            out += _sweep("sweep", field, values)
    if "hysteresis" in studies:
        pairs = [(lo, hi) for lo in (1.5, 2.0, 2.5, 3.0) for hi in (3.0, 4.0, 5.0, 6.0) if lo < hi]
        pairs += [(hi / 2, hi) for hi in (2.5, 3.5, 7.0, 8.0)]
        out += _sweep("hysteresis", ("det.weak_low_sigma", "det.weak_high_sigma"), pairs)
    if "grouping" in studies:
        out += _sweep("grouping", ("grp.tol_time_s", "grp.tol_velocity_kmh"),
                      [(t, v) for t in (0.5, 1.0, 1.5, 2.0) for v in (4.0, 8.0, 12.0)])
    if "combined" in studies:
        # candidates neutral on the synthetic data that change the real-data scores
        # (tie-break rule: synthetic F1 first, real F1 only among synthetic-equivalent settings)
        out += [Variant("combined", final)] + [Variant("combined", n, c) for n, c in (
            ("tol_time_s = 1.5", (("grp.tol_time_s", 1.5),)),
            ("tol_time_s = 2.0", (("grp.tol_time_s", 2.0),)),
            ("strong_min_sigma = 5", (("det.strong_min_sigma", 5.0),)),
            ("tol_time_s = 1.5 + strong_min_sigma = 5", (("grp.tol_time_s", 1.5),
                                                         ("det.strong_min_sigma", 5.0))),
            ("tol_time_s = 2.0 + strong_min_sigma = 5", (("grp.tol_time_s", 2.0),
                                                         ("det.strong_min_sigma", 5.0))),
            ("tol_time_s = 1.5, tol_velocity_kmh = 12", (("grp.tol_time_s", 1.5),
                                                          ("grp.tol_velocity_kmh", 12.0))),
        )]
    return out


def apply_changes(changes: tuple) -> PipelineParams:
    """Default parameters with the variant's changes applied."""
    pp = PipelineParams()
    for key, value in changes:
        section, name = key.split(".")
        if not hasattr(getattr(pp, section), name):
            raise AttributeError(f"Unknown parameter {key}")
        setattr(getattr(pp, section), name, value)
    return pp


def stage_keys(pp: PipelineParams, baseline: str = "") -> tuple[str, str, str]:
    """Cache keys of Stages 4, 5 and 6 (everything upstream of a stage is in its key)."""
    kp = repr(asdict(pp.pre))
    ke = kp + repr(asdict(pp.enh))
    kd = ke + (f"baseline={baseline}" + repr(asdict(pp.det)) if baseline else repr(asdict(pp.det)))
    return kp, ke, kd


# ======================================================================================
# Classical baseline (unsuccessful approach of the brief / plan): Canny + Hough
# ======================================================================================
def canny_hough_segments(img: np.ndarray, pp: PipelineParams, row_dt: float, col_dx: float
                         ) -> list[LineSegment]:
    """Edges (Canny with median-based thresholds after a 5x5 Gaussian blur) of ``img``
    upsampled to the response grid -> HoughLinesP (same settings as the weak path) ->
    segments whose slope is inside the velocity fan of one direction. The score is the
    robust z of the image along the segment."""
    up = pp.enh.space_upsample
    u = upsample_space(img, up)
    lo, hi = np.percentile(u, [1.0, 99.5])
    u8 = np.clip((u - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (5, 5), 0)
    med = float(np.median(u8))
    edges = cv2.Canny(u8, int(max(0.0, 0.66 * med)), int(min(255.0, 1.33 * med)))
    d = pp.det
    lines = cv2.HoughLinesP(edges, 1, np.radians(d.hough_theta_deg), d.hough_threshold,
                            minLineLength=d.hough_min_line_length, maxLineGap=d.hough_max_line_gap)
    if lines is None:
        return []
    z = robust_z(u)
    cdx = col_dx / up
    vr = (pp.enh.v_min_kmh, pp.enh.v_max_kmh)
    out = []
    for c0, r0, c1, r1 in np.asarray(lines).reshape(-1, 4).astype(float):
        for direction in (1, -1):
            seg = LineSegment(c0, r0, c1, r1, 0.0, "weak", direction, "hf")
            if slope_ok(seg.slope, slope_bounds(vr, direction, row_dt, cdx), d.slope_margin):
                seg.score = float(np.mean(sample_image(z, seg.points())))
                out.append(seg)
                break
    out.sort(key=lambda s: -s.length_px)
    return out[:1500]                       # the grouping link matrix is O(n^2)


# ======================================================================================
# Worker: all variants on one recording
# ======================================================================================
def load_recording(spec: tuple[str, str], labels_path: str) -> dict:
    """Load one synthetic sample or labelled real segment with its reference."""
    kind, name = spec
    if kind == "synthetic":
        sig, gt, meta = dio.load_synthetic(name)
        dt, dx = dio.synthetic_axes(meta)
        return {"name": f"synthetic_{name}", "kind": kind, "x": sig, "fs": 1.0 / dt, "dx": dx,
                "crop": (0, 0), "refs": gt_objects_from_mask(gt, meta, dt, dx)}
    seg = dio.load_named_segment(name, pad_files=1)
    refs_all = references_from_annotations(segment_lines(load_labels(labels_path), name))
    ev = PipelineParams().ev
    refs, _ = split_references(refs_all, ev.ref_v_range_kmh, ev.ref_min_extent_m)
    return {"name": name, "kind": kind, "x": seg.data, "fs": 1.0 / seg.dt, "dx": seg.dx,
            "crop": (seg.pad_before, seg.pad_after), "refs": refs}


def _strip(enhanced: dict) -> dict:
    """Keep only what detection and tracking need (memory of the cache)."""
    for e in enhanced.values():
        e.stages = {k: v for k, v in e.stages.items() if k in ("fk_pos", "fk_neg")}
    return enhanced


class StageCache:
    """Results of Stages 4 / 5 / 6 of one recording: the default and the latest key."""

    def __init__(self, default_keys: tuple[str, str, str]):
        self.default = dict(zip(("pre", "enh", "det"), default_keys))
        self.data: dict[tuple[str, str], tuple[object, float]] = {}

    def get(self, stage: str, key: str):
        return self.data.get((stage, key))

    def put(self, stage: str, key: str, value, seconds: float) -> tuple[object, float]:
        """Store (evicting other non-default keys of the stage) and return (value, s)."""
        for s, k in list(self.data):
            if s == stage and k not in (key, self.default[stage]):
                del self.data[(s, k)]
        self.data[(stage, key)] = (value, seconds)
        return value, seconds


def evaluate_recording(spec: tuple[str, str], variants: list[Variant], labels_path: str,
                       single_thread: bool) -> list[dict]:
    """Run all variants on one recording (worker process), reusing cached stages."""
    if single_thread:
        cv2.setNumThreads(1)
    rec = load_recording(spec, labels_path)
    cache = StageCache(stage_keys(PipelineParams()))
    order = sorted(variants, key=lambda v: stage_keys(apply_changes(v.changes), v.baseline))
    out = []
    for v in order:
        pp = apply_changes(v.changes)
        kp, ke, kd = stage_keys(pp, v.baseline)
        times = {}
        if (hit := cache.get("pre", kp)) is None:
            t = time.perf_counter()
            pre = preprocess(rec["x"], pp.pre, rec["fs"], rec["dx"], crop=rec["crop"])
            hit = cache.put("pre", kp, pre, time.perf_counter() - t)
        pre, times["preprocess_s"] = hit
        if (hit := cache.get("enh", ke)) is None:
            t = time.perf_counter()
            enh = {b: enhance(img, pp.enh, pre.row_dt, pre.col_dx) for b, img in pre.images.items()}
            hit = cache.put("enh", ke, _strip(enh), time.perf_counter() - t)
        enh, times["enhance_s"] = hit
        if (hit := cache.get("det", kd)) is None:
            t = time.perf_counter()
            if v.baseline:
                img = (np.log10(pre.common["raw"] + 1e-12) if v.baseline == "canny_raw"
                       else pre.images["hf"])
                det = Detections(segments=canny_hough_segments(img, pp, pre.row_dt, pre.col_dx))
            else:
                det = detect_candidates(enh, pp.det, (pp.enh.v_min_kmh, pp.enh.v_max_kmh))
                det.stages = {}
            hit = cache.put("det", kd, det, time.perf_counter() - t)
        det, times["detect_s"] = hit
        res = run_pipeline(rec["x"], pp, rec["fs"], rec["dx"], crop=rec["crop"], features=False,
                           pre=pre, enhanced=enh, detections=det)
        times["group_track_s"] = res.timings["group_s"] + res.timings["track_s"]
        ev = evaluate_objects(res.objects, rec["refs"], pp.ev, pre.images["hf"].shape,
                              res.row_dt, rec["dx"])
        ev = {k: val for k, val in ev.items() if k not in ("matches", "fp_kinds")}
        out.append({"key": v.key, "recording": rec["name"], "kind": rec["kind"],
                    "n_det": len(res.objects), "time_s": sum(times.values()), "ev": ev})
    return out


# ======================================================================================
# Aggregation
# ======================================================================================
def summarize_variant(v: Variant, results: list[dict]) -> dict:
    """Aggregate the metrics of one variant over the recordings."""
    row = {"study": v.study, "variant": v.name,
           "changes": "; ".join(f"{k} = {val}" for k, val in v.changes)
           + (f"; detector = {v.baseline}" if v.baseline else "")}
    mine = [r for r in results if r["key"] == v.key]
    for kind, pre in (("synthetic", "syn"), ("real", "real")):
        evs = [r["ev"] for r in mine if r["kind"] == kind]
        if not evs:
            continue
        s = summarize_evaluations(evs)
        row.update({f"{pre}_precision": s["precision"], f"{pre}_recall": s["recall"],
                    f"{pre}_f1": s["f1"], f"{pre}_recall_thick": s["recall_heavy"],
                    f"{pre}_recall_thin": s["recall_light"], f"{pre}_fp": s["fp"],
                    f"{pre}_vel_mae": s["vel_mae"]})
        if kind == "synthetic":
            row["syn_f1_thin"] = s["f1_light"]
        else:
            row.update(real_fp_near_label=s["fp_near_label"], real_fp_no_label=s["fp_no_label"],
                       real_n_det=sum(r["n_det"] for r in mine if r["kind"] == kind))
    row["time_s"] = float(np.mean([r["time_s"] for r in mine])) if mine else float("nan")
    return row


def add_deltas(rows: list[dict], variants: list[Variant]) -> None:
    """Change of the F1 scores vs the final parameters of the same study."""
    default = repr(asdict(PipelineParams()))
    ref = {v.study: r for v, r in zip(variants, rows)
           if not v.baseline and repr(asdict(apply_changes(v.changes))) == default}
    for v, r in zip(variants, rows):
        b = ref.get(v.study)
        for k in ("syn_f1", "real_f1"):
            r[f"d_{k}"] = (r.get(k, float("nan")) - b.get(k, float("nan"))) if b else float("nan")


def _f(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_table(rows: list[dict], study: str) -> None:
    """Print the results of one study."""
    print(f"\n== {study} ==")
    print(f"  {'variant':48} {'syn F1':>6} {'thick':>6} {'thin':>6} {'P':>6} {'MAE':>5} | "
          f"{'real F1':>7} {'thick':>6} {'thin':>6} {'P':>6} {'FPnear':>6} {'FPno':>5} | {'t [s]':>5}")
    for r in rows:
        print(f"  {r['variant'][:48]:48} {r.get('syn_f1', np.nan):6.3f} "
              f"{r.get('syn_recall_thick', np.nan):6.3f} {r.get('syn_recall_thin', np.nan):6.3f} "
              f"{r.get('syn_precision', np.nan):6.3f} {r.get('syn_vel_mae', np.nan):5.2f} | "
              f"{r.get('real_f1', np.nan):7.3f} {r.get('real_recall_thick', np.nan):6.3f} "
              f"{r.get('real_recall_thin', np.nan):6.3f} {r.get('real_precision', np.nan):6.3f} "
              f"{r.get('real_fp_near_label', 0):6d} {r.get('real_fp_no_label', 0):5d} | "
              f"{r['time_s']:5.1f}")


# ======================================================================================
# Figures
# ======================================================================================
def figure_bars(rows: list[dict], study: str, dpi: int) -> None:
    """Grouped horizontal bars: synthetic and real metrics of every variant."""
    names = [r["variant"] for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.42 * len(rows) + 2.2), sharey=True,
                             layout="constrained")
    for ax, pre, title in ((axes[0], "syn", "synthetic (ground truth)"),
                           (axes[1], "real", "real segments (manual labels, indicative)")):
        for k, (key, lab) in enumerate(((f"{pre}_recall_thick", "recall thick"),
                                        (f"{pre}_recall_thin", "recall thin"),
                                        (f"{pre}_precision", "precision"), (f"{pre}_f1", "F1"))):
            ax.barh(y + (k - 1.5) * 0.2, [r.get(key, np.nan) for r in rows], height=0.2,
                    label=lab)
        ref = next((r.get(f"{pre}_f1") for r in rows if r["variant"] == "final parameters"), None)
        if ref is not None:
            ax.axvline(ref, color="k", lw=0.8, ls="--")
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=4)
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(f"Stage 13 - {study}: object-level metrics (dashed = F1 of the final parameters)")
    save_figure(fig, FIG_DIR / f"{study}.png", dpi)


def figure_sweep(rows: list[dict], variants: list[Variant], dpi: int) -> None:
    """One panel per swept parameter: F1 (synthetic, real) and thin recall vs the value."""
    final = next(r for r in rows if r["variant"] == "final parameters")
    defaults = PipelineParams()
    by_field: dict[str, list] = {}
    for v, r in zip(variants, rows):
        if v.changes:
            by_field.setdefault(v.changes[0][0], []).append((v.changes[0][1], r))
    n = len(by_field)
    ncols = 3
    fig, axes = plt.subplots(-(-n // ncols), ncols, figsize=(15, 3.3 * -(-n // ncols)),
                             layout="constrained", squeeze=False)
    for ax, (field, pts) in zip(axes.ravel(), by_field.items()):
        sec, name = field.split(".")
        pts = pts + [(getattr(getattr(defaults, sec), name), final)]
        pts.sort(key=lambda p: p[0])
        xs = [p[0] for p in pts]
        for key, lab, st in (("syn_f1", "F1 synthetic", "o-"), ("syn_recall_thin", "thin recall synthetic", "s:"),
                             ("real_f1", "F1 real", "o-"), ("real_recall_thin", "thin recall real", "s:")):
            ax.plot(xs, [p[1].get(key, np.nan) for p in pts], st, label=lab, ms=4)
        ax.axvline(getattr(getattr(defaults, sec), name), color="k", lw=0.8, ls="--")
        ax.set_title(field + (" (+ lf_high)" if name == "lf_low" else ""), fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        tx = ax.twinx()
        times = [p[1]["time_s"] for p in pts]
        tx.plot(xs, times, "k.", alpha=0.4)
        tx.set_ylim(0, 1.2 * max(times))
        tx.set_ylabel("s / recording", fontsize=7)
        tx.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=7, loc="lower left")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Stage 13 - one-at-a-time sweep around the final parameters (dashed = final value, "
                 "dots = time per recording)")
    save_figure(fig, FIG_DIR / "sweep.png", dpi)


def _grid(rows, variants, key):
    xs = sorted({v.changes[0][1] for v in variants})
    ys = sorted({v.changes[1][1] for v in variants})
    g = np.full((len(ys), len(xs)), np.nan)
    for v, r in zip(variants, rows):
        g[ys.index(v.changes[1][1]), xs.index(v.changes[0][1])] = r.get(key, np.nan)
    return xs, ys, g


def figure_heatmaps(rows: list[dict], variants: list[Variant], study: str, xlabel: str,
                    ylabel: str, final: tuple, dpi: int, extra_ax: bool = False) -> plt.Figure:
    """Heat maps of the metrics over a two-parameter grid."""
    keys = (("syn_f1", "F1 synthetic"), ("syn_recall_thin", "thin recall synthetic"),
            ("real_f1", "F1 real"), ("real_precision", "precision real"))
    ncols = len(keys) + int(extra_ax)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.8), layout="constrained")
    for ax, (key, title) in zip(axes, keys):
        xs, ys, g = _grid(rows, variants, key)
        im = ax.imshow(g, origin="lower", cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(xs)), xs)
        ax.set_yticks(range(len(ys)), ys)
        for (i, j), val in np.ndenumerate(g):
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if val < np.nanmean(g) else "k")
        if final[0] in xs and final[1] in ys:
            ax.add_patch(plt.Rectangle((xs.index(final[0]) - 0.5, ys.index(final[1]) - 0.5), 1, 1,
                                       fill=False, ec="red", lw=2))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Stage 13 - {study} grid (red = final parameters)")
    if not extra_ax:
        save_figure(fig, FIG_DIR / f"{study}.png", dpi)
    return fig


def figure_hysteresis(rows: list[dict], variants: list[Variant], dpi: int) -> None:
    """Heat maps of the hysteresis grid (low x high threshold)."""
    grid = [(v, r) for v, r in zip(variants, rows) if v.changes[0][1] in (1.5, 2.0, 2.5, 3.0)
            and v.changes[1][1] in (3.0, 4.0, 5.0, 6.0)]
    d = PipelineParams().det
    figure_heatmaps([r for _, r in grid], [v for v, _ in grid], "hysteresis",
                    "low threshold [sigma]", "high threshold [sigma]",
                    (d.weak_low_sigma, d.weak_high_sigma), dpi)
    curve = sorted([(v.changes[1][1], r) for v, r in zip(variants, rows)
                    if abs(v.changes[0][1] - v.changes[1][1] / 2) < 1e-9], key=lambda p: p[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained", sharey=True)
    for ax, pre, title in ((axes[0], "syn", "synthetic"), (axes[1], "real", "real (manual labels)")):
        hs = [p[0] for p in curve]
        for key, lab in (("f1", "F1"), ("recall_thin", "recall thin"), ("recall_thick", "recall thick"),
                         ("precision", "precision")):
            ax.plot(hs, [p[1].get(f"{pre}_{key}", np.nan) for p in curve], "o-", label=lab, ms=4)
        ax.axvline(d.weak_high_sigma, color="k", ls="--", lw=0.8)
        ax.set_xlabel("weak-path hysteresis: high threshold [sigma] (low = high / 2)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage 13 - sensitivity to the weak-path threshold (dashed = final)")
    save_figure(fig, FIG_DIR / "hysteresis_sensitivity.png", dpi)


# ======================================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", nargs="*", default=list(STUDIES), choices=STUDIES)
    p.add_argument("--quick", action="store_true", help="2 synthetic samples + 1 real segment")
    p.add_argument("--jobs", type=int, default=0, help="parallel processes (0 = auto)")
    p.add_argument("--labels", default=str(config.ANNOT_FILE))
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = make_variants(args.study)
    unique = list({v.key: v for v in variants}.values())
    syn = dio.list_synthetic_ids()
    real = [s["name"] for s in config.SEGMENTS
            if segment_lines(load_labels(args.labels), s["name"])]
    if args.quick:
        syn, real = syn[:2], real[:1]
    specs = [("synthetic", s) for s in syn] + [("real", r) for r in real]
    jobs = args.jobs or max(1, min(len(specs), (os.cpu_count() or 2) - 2))
    print(f"Stage 13: {len(variants)} variants ({len(unique)} distinct) x {len(specs)} recordings "
          f"({len(syn)} synthetic, {len(real)} real with labels), {jobs} process(es)", flush=True)
    t0 = time.perf_counter()
    results: list[dict] = []
    if jobs == 1:
        for spec in specs:
            results += evaluate_recording(spec, unique, args.labels, False)
            print(f"  {spec[1]} done ({time.perf_counter() - t0:.0f} s)")
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(evaluate_recording, s, unique, args.labels, True): s for s in specs}
            for f in as_completed(futs):
                results += f.result()
                print(f"  {futs[f][1]} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    rows = [summarize_variant(v, results) for v in variants]
    add_deltas(rows, variants)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for study in args.study:
        sel = [(v, r) for v, r in zip(variants, rows) if v.study == study]
        sv, sr = [v for v, _ in sel], [r for _, r in sel]
        print_table(sr, study)
        with open(config.RESULTS_DIR / f"tuning_{study}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows([{k: _f(r.get(k, "")) for k in FIELDS} for r in sr])
        if study in ("ablation", "methods", "combined"):
            figure_bars(sr, study, args.dpi)
        elif study == "sweep":
            figure_sweep(sr, sv, args.dpi)
        elif study == "hysteresis":
            figure_hysteresis(sr, sv, args.dpi)
        elif study == "grouping":
            g = PipelineParams().grp
            figure_heatmaps(sr, sv, "grouping", "tol_time_s [s]", "tol_velocity_kmh [km/h]",
                            (g.tol_time_s, g.tol_velocity_kmh), args.dpi)
    summary = {"recordings": [s[1] for s in specs], "quick": args.quick,
               "elapsed_s": time.perf_counter() - t0,
               "final_parameters": asdict(PipelineParams()),
               "rows": [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                         for k, v in r.items()} for r in rows]}
    name = "tuning_quick.json" if args.quick else "tuning.json"
    old = config.RESULTS_DIR / name
    if old.exists() and set(args.study) != set(STUDIES):       # keep the other studies
        prev = json.loads(old.read_text(encoding="utf-8"))
        keep = [r for r in prev.get("rows", []) if r["study"] not in args.study]
        summary["rows"] = keep + summary["rows"]
    old.write_text(json.dumps(copy.deepcopy(summary), indent=1, default=str), encoding="utf-8")
    print(f"\n{time.perf_counter() - t0:.0f} s. Results: {config.RESULTS_DIR}/tuning_*.csv, "
          f"figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
