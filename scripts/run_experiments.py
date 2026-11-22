"""Stage 11: experiments - ablation, method comparison and parameter sweeps.

Every variant changes the final parameters (``config.PipelineParams``) in one place and
runs the whole pipeline (Stages 4-8) on

* the 3 real segments, scored against the manual labels (scored labels only), and
* the 6 synthetic samples, scored against the ground truth,

with the grade-4 metrics of ``evaluation.evaluate_objects`` (thick recall, precision,
F1, thick |v| error). Groups of variants:

* **ablation** - one step of the method switched off at a time;
* **method** - alternatives of single steps (another enhancer, another pass band) and
  a classical baseline (Canny edges + probabilistic Hough on the raw image);
* **sweep** - key parameters one at a time around the final value.

Selection rule (column ``decision``): the synthetic F1 (true ground truth) decides; one
object changes it by ~0.007, so a variant is better only if it gains more than
``SYN_GAIN`` = 0.01. Such a variant is *vetoed* when it loses more than 0.01 of real F1
(it helps only the simulator). Between synthetic ties (|dF1| <= 0.01) the real F1 (rough
manual labels) decides, but only for a gain of more than ``REAL_GAIN`` = 0.02 (one real
object changes it by ~0.01).

Runs in parallel, one process per recording (``--jobs``); inside a process the results
of Stages 4 and 5 are reused between variants with the same upstream parameters.

Outputs: ``outputs/results/experiments.csv`` / ``.json`` and
``outputs/figures/experiments/experiments.png``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das.annotation import load_labels, segment_lines  # noqa: E402
from das.config import PipelineParams  # noqa: E402
from das.detection import LineSegment, slope_bounds, slope_ok  # noqa: E402
from das.enhancement import enhance, robust_z  # noqa: E402
from das.evaluation import (evaluate_objects, gt_objects_from_mask,  # noqa: E402
                            references_from_annotations, split_references,
                            summarize_evaluations)
from das.grouping import build_objects  # noqa: E402
from das.pipeline import run_pipeline, to_jsonable  # noqa: E402
from das.preprocessing import preprocess, upsample_space  # noqa: E402
from das.visualization import save_figure  # noqa: E402

FIG_DIR = config.FIG_DIR / "experiments"
SYN_GAIN, REAL_GAIN = 0.01, 0.02          # significance of an F1 change (selection rule)


@dataclass
class Variant:
    """One experiment: parameter changes ("section.field": value) or a baseline."""
    group: str
    name: str
    changes: dict = field(default_factory=dict)
    baseline: str = ""                       # "" | "canny"


VARIANTS = [
    Variant("final", "final method"),
    # --- ablation: one step off -----------------------------------------------------------
    Variant("ablation", "no common-mode removal", {"pre.common_mode": "none"}),
    Variant("ablation", "no RMS envelope (plain decimation)", {"pre.envelope": "decimate"}),
    Variant("ablation", "no log compression (env / median)", {"pre.normalization": "ratio"}),
    Variant("ablation", "no f-k fan filter", {"enh.fk_filter": False}),
    Variant("ablation", "no directional filter bank", {"enh.enhancer": "none"}),
    Variant("ablation", "no 4-sigma floor (Otsu only)", {"det.min_sigma": 0.0}),
    Variant("ablation", "one RANSAC line per component", {"det.max_lines": 1}),
    Variant("ablation", "no linking of candidates (duplicate removal only)",
            {"grp.tol_velocity_kmh": 0.0, "grp.tol_velocity_rel": 0.0, "grp.cont_max_gap_m": -1e9}),
    # --- method comparison ------------------------------------------------------------------
    Variant("method", "Sato ridge filter instead of the filter bank", {"enh.enhancer": "sato"}),
    Variant("method", "fixed threshold 6 sigma instead of Otsu", {"det.threshold": "6"}),
    Variant("method", "band 1-8 Hz (best low-frequency band)", {"pre.band_low": 1.0,
                                                                 "pre.band_high": 8.0}),
    Variant("method", "band 20-312.5 Hz", {"pre.band_low": 20.0}),
    Variant("method", "broadband 0.2-312.5 Hz", {"pre.band_low": 0.2}),
    Variant("method", "baseline: Canny + HoughLinesP on the raw image", baseline="canny"),
    # --- sweeps -------------------------------------------------------------------------------
    *[Variant("sweep", f"threshold floor {s:g} sigma", {"det.min_sigma": s}) for s in (3.0, 5.0, 6.0)],
    *[Variant("sweep", f"kernel length {n} px", {"enh.kernel_length_px": n}) for n in (41, 121)],
    Variant("sweep", "12 kernel angles", {"enh.n_angles": 12}),
    Variant("sweep", "upsampling x4", {"enh.space_upsample": 4}),
    *[Variant("sweep", f"min. component area {a} px", {"det.min_area_px": a}) for a in (800, 3000)],
    *[Variant("sweep", f"grouping time tolerance {t:g} s", {"grp.tol_time_s": t}) for t in (1.0, 2.0)],
]


def make_params(changes: dict) -> PipelineParams:
    """Final parameters with ``changes`` ("section.field": value) applied."""
    pp = copy.deepcopy(PipelineParams())
    for key, value in changes.items():
        section, name = key.split(".")
        sub = getattr(pp, section)
        if not hasattr(sub, name):
            raise AttributeError(f"Unknown parameter {key}")
        setattr(sub, name, value)
    return pp


# ======================================================================================
# Classical baseline (unsuccessful approach): Canny + probabilistic Hough
# ======================================================================================
def canny_hough_objects(img: np.ndarray, pp: PipelineParams, row_dt: float, dx: float):
    """Edges (Canny with median-based thresholds after a 5x5 Gaussian blur) of ``img``
    upsampled like the response maps -> HoughLinesP -> segments whose slope is inside the
    velocity fan of one direction -> the same grouping as the method (Stage 7)."""
    up = pp.enh.space_upsample
    u = upsample_space(img, up)
    lo, hi = np.percentile(u, [1.0, 99.5])
    u8 = cv2.GaussianBlur(np.clip((u - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8),
                          (5, 5), 0)
    med = float(np.median(u8))
    edges = cv2.Canny(u8, int(max(0.0, 0.66 * med)), int(min(255.0, 1.33 * med)))
    lines = cv2.HoughLinesP(edges, 1, np.radians(0.5), 30, minLineLength=60, maxLineGap=20)
    z, cdx = robust_z(u), dx / up
    vr = (pp.enh.v_min_kmh, pp.enh.v_max_kmh)
    segs = []
    for c0, r0, c1, r1 in (np.asarray(lines).reshape(-1, 4).astype(float) if lines is not None else []):
        for d in (1, -1):
            s = LineSegment(c0, r0, c1, r1, 0.0, d, 0.0)
            if slope_ok(s.slope, slope_bounds(vr, d, row_dt, cdx), pp.det.slope_margin):
                p = np.rint(s.points()).astype(int)
                s.score = float(np.mean(z[np.clip(p[:, 1], 0, z.shape[0] - 1),
                                          np.clip(p[:, 0], 0, z.shape[1] - 1)]))
                segs.append(s)
                break
    segs = sorted(segs, key=lambda s: -s.length_px)[:1500]   # the link matrix is O(n^2)
    objs, _ = build_objects(segs, pp.grp, row_dt, cdx, 0.5 * img.shape[1] * dx, dx)
    return objs


# ======================================================================================
# Worker: all variants on one recording
# ======================================================================================
def load_recording(kind: str, name: str) -> dict:
    """Raw matrix, sampling, padding and reference objects of one recording."""
    if kind == "synthetic":
        sig, gt, meta = dio.load_synthetic(name)
        dt, dx = dio.synthetic_axes(meta)
        return {"x": sig, "fs": 1 / dt, "dx": dx, "crop": (0, 0),
                "refs": gt_objects_from_mask(gt, meta, dt, dx)}
    seg = dio.load_named_segment(name, pad_files=1)
    ev = PipelineParams().ev
    refs, _ = split_references(references_from_annotations(segment_lines(load_labels(), name)),
                               ev.ref_v_range_kmh, ev.ref_min_extent_m)
    return {"x": seg.data, "fs": 1 / seg.dt, "dx": seg.dx,
            "crop": (seg.pad_before, seg.pad_after), "refs": refs}


def run_recording(job: tuple[str, str]) -> list[dict]:
    """Every variant on one recording -> one row of metrics per variant."""
    kind, name = job
    rec = load_recording(kind, name)
    pre_cache, enh_cache, out = {}, {}, []
    for v in VARIANTS:
        pp = make_params(v.changes)
        t = time.perf_counter()
        kp = json.dumps(asdict(pp.pre), sort_keys=True)
        if kp not in pre_cache:
            pre_cache[kp] = preprocess(rec["x"], pp.pre, rec["fs"], rec["dx"], crop=rec["crop"])
        pre = pre_cache[kp]
        if v.baseline == "canny":
            objs = canny_hough_objects(np.log10(pre.stages["raw"] + 1e-12), pp, pre.row_dt,
                                       rec["dx"])
        else:
            ke = kp + json.dumps(asdict(pp.enh), sort_keys=True)
            if ke not in enh_cache:
                e = enhance(pre.image, pp.enh, pre.row_dt, pre.col_dx)
                e.stages = {}                    # only the response maps are needed
                enh_cache[ke] = e
            objs = run_pipeline(rec["x"], pp, rec["fs"], rec["dx"], pre=pre,
                                enhanced=enh_cache[ke]).objects
        ev = evaluate_objects(objs, rec["refs"], pp.ev, pre.image.shape[1], rec["dx"])
        out.append({"variant": v.name, "kind": kind, "recording": name,
                    "time_s": time.perf_counter() - t,
                    "ev": {k: val for k, val in ev.items() if k not in ("matches", "fp_kinds")}})
    print(f"  {kind} {name}: {len(VARIANTS)} variants done", flush=True)
    return out


# ======================================================================================
# Summary
# ======================================================================================
METRICS = ("recall_thick", "precision", "f1", "vel_mae_thick", "n_det")


def summarise(rows: list[dict]) -> list[dict]:
    """Pooled metrics per variant and dataset (synthetic / real)."""
    out = []
    for v in VARIANTS:
        r = {"group": v.group, "variant": v.name,
             "changes": "; ".join(f"{k}={val}" for k, val in v.changes.items()) or v.baseline}
        for kind, pre in (("synthetic", "syn"), ("real", "real")):
            evs = [x["ev"] for x in rows if x["variant"] == v.name and x["kind"] == kind]
            tot = summarize_evaluations(evs)
            r.update({f"{pre}_{m}": tot[m] for m in METRICS})
        r["time_s"] = float(np.mean([x["time_s"] for x in rows if x["variant"] == v.name]))
        out.append(r)
    for r in out:
        r["decision"] = decision(r, out[0])
    return out


def decision(r: dict, final: dict) -> str:
    """Outcome of the selection rule for one variant (see the module docstring)."""
    d_syn, d_real = r["syn_f1"] - final["syn_f1"], r["real_f1"] - final["real_f1"]
    if r is final:
        return "final"
    if d_syn > SYN_GAIN:
        return "vetoed: better only on synthetic" if d_real < -SYN_GAIN else "ADOPT"
    if abs(d_syn) <= SYN_GAIN and d_real > REAL_GAIN:
        return "ADOPT (real tie-break)"
    return "worse" if d_syn < -SYN_GAIN or d_real < -REAL_GAIN else "no significant change"


def figure(table: list[dict], dpi: int) -> None:
    """Horizontal bars of the synthetic and real F1 / thick recall of every variant."""
    n = len(table)
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.33 * n + 2), sharey=True, layout="constrained")
    y = np.arange(n)[::-1]
    names = [f"[{r['group']}] {r['variant']}" for r in table]
    for ax, (key, title) in zip(axes, (("f1", "F1 (thick recall x precision)"),
                                       ("recall_thick", "thick recall"),
                                       ("precision", "precision"))):
        ax.barh(y + 0.2, [r[f"syn_{key}"] for r in table], 0.4, label="synthetic (GT)")
        ax.barh(y - 0.2, [r[f"real_{key}"] for r in table], 0.4, label="real (manual labels)")
        ax.axvline(table[0][f"syn_{key}"], color="tab:blue", ls=":", lw=1)
        ax.axvline(table[0][f"real_{key}"], color="tab:orange", ls=":", lw=1)
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Experiments: every variant changes one step of the final method "
                 "(dotted = final method)")
    save_figure(fig, FIG_DIR / "experiments.png", dpi)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=min(9, os.cpu_count() or 1),
                   help="parallel processes (one recording each)")
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels()
    jobs = [("real", s["name"]) for s in config.SEGMENTS if segment_lines(labels, s["name"])]
    jobs += [("synthetic", sid) for sid in dio.list_synthetic_ids()]
    print(f"{len(VARIANTS)} variants x {len(jobs)} recordings, {args.jobs} processes")
    t = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = [r for part in ex.map(run_recording, jobs) for r in part]
    else:
        rows = [r for job in jobs for r in run_recording(job)]
    table = summarise(rows)
    print(f"done in {time.perf_counter() - t:.0f} s\n")
    print(f"  {'variant':52} {'syn R':>6} {'syn P':>6} {'syn F1':>6} {'syn |v|':>7} "
          f"{'real R':>6} {'real P':>6} {'real F1':>7} {'time':>6}  decision")
    for r in table:
        print(f"  {r['variant'][:52]:52} {r['syn_recall_thick']:6.3f} {r['syn_precision']:6.3f} "
              f"{r['syn_f1']:6.3f} {r['syn_vel_mae_thick']:7.2f} {r['real_recall_thick']:6.3f} "
              f"{r['real_precision']:6.3f} {r['real_f1']:7.3f} {r['time_s']:6.1f}  {r['decision']}")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "experiments.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in table])
    (config.RESULTS_DIR / "experiments.json").write_text(
        json.dumps(to_jsonable({"final_params": asdict(PipelineParams()), "table": table,
                                "per_recording": rows}), indent=1), encoding="utf-8")
    figure(table, args.dpi)
    print(f"\nResults: {config.RESULTS_DIR / 'experiments.csv'}\nFigure: {FIG_DIR / 'experiments.png'}")


if __name__ == "__main__":
    main()
