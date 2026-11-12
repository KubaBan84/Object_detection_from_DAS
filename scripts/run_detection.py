"""Stages 6-7: line detection (strong + weak paths) and grouping of the candidates.

1. Synthetic samples: Stages 4-5 -> detection -> grouping, matched against the ground
   truth (objects built from the marker mask) -> recall per vehicle type (thin = car/van,
   thick = tram/truck), precision, F1 and velocity error.
2. Real segments: the same chain; figures of every detection step (before / after) and
   "candidates -> groups -> mean line"; object table (CSV) per segment.
3. Variant study: detection / grouping options changed one at a time (Hough variants,
   masking of strong objects, bands, thresholds, tolerances), scored on the synthetic
   ground truth; the number of objects on the real segments is reported alongside.

Outputs: ``outputs/figures/detection/``, ``outputs/results/detection_*.csv`` and
``outputs/results/detection_summary.json``.

Plan: Stages 6 and 7.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das.config import (DetectionParams, EnhancementParams, GroupingParams,  # noqa: E402
                        PreprocessingParams)
from das.detection import detect_candidates  # noqa: E402
from das.enhancement import enhance  # noqa: E402
from das.evaluation import (HEAVY_TYPES, LIGHT_TYPES, detection_metrics,  # noqa: E402
                            gt_envelope_mask, gt_objects_from_mask, match_detections,
                            velocity_errors)
from das.grouping import build_objects  # noqa: E402
from das.preprocessing import preprocess, preprocess_segment  # noqa: E402
from das.velocity import write_objects_csv  # noqa: E402
from das.visualization import (plot_detections, plot_segments, plot_waterfall,  # noqa: E402
                               save_figure)

FIG_DIR = config.FIG_DIR / "detection"
MATCH_TOL_T_S = 1.5        # matching gate: time gap at the detection centre [s]
MATCH_TOL_V_KMH = 20.0     # matching gate: velocity difference [km/h]

# (group, name, detection overrides, grouping overrides) - relative to the defaults
VARIANTS = [
    ("baseline", "default", {}, {}),
    # Hough variants (plan 6B.5: compare HoughLinesP with the global hough_line)
    ("hough", "HoughLinesP only", {"use_hough_global": False}, {}),
    ("hough", "global hough_line only", {"use_hough_p": False}, {}),
    ("hough", "global Hough gap = HoughLinesP gap (20 px)", {"global_max_gap_px": 20}, {}),
    # processing paths
    ("paths", "strong path only (grade 3/4)", {"weak_bands": ()}, {}),
    ("paths", "no masking of strong objects", {"mask_strong": False}, {}),
    ("paths", "weak path HF only", {"weak_bands": ("hf",)}, {}),
    ("paths", "weak path LF only", {"weak_bands": ("lf",)}, {}),
    ("paths", "no local-orientation check", {"orientation_tol_deg": 0.0}, {}),
    # strong path (6A)
    ("strong", "one RANSAC line per component", {"strong_max_lines": 1}, {}),
    ("strong", "no ridge NMS (1 row)", {"strong_ridge_nms_rows": 1}, {}),
    ("strong", "strong floor 3 sigma", {"strong_min_sigma": 3.0}, {}),
    ("strong", "strong floor 6 sigma", {"strong_min_sigma": 6.0}, {}),
    ("strong", "strong floor 10 sigma", {"strong_min_sigma": 10.0}, {}),
    ("strong", "strong threshold 8 sigma (no Otsu)", {"strong_threshold": "8"}, {}),
    # weak path (6B) thresholds
    ("weak", "hysteresis 1.5 / 3 sigma", {"weak_low_sigma": 1.5, "weak_high_sigma": 3.0}, {}),
    ("weak", "hysteresis 2.5 / 5 sigma", {"weak_low_sigma": 2.5, "weak_high_sigma": 5.0}, {}),
    ("weak", "hysteresis 3 / 6 sigma", {"weak_low_sigma": 3.0, "weak_high_sigma": 6.0}, {}),
    # grouping and clean-up (Stage 7)
    ("grouping", "no weak acceptance rule (support 40 m)", {}, {"weak_min_support_m": 40.0}),
    ("grouping", "weak: support >= 60 m only (no score)", {}, {"weak_short_min_score": np.inf}),
    ("grouping", "weak: support 80 m or score 5", {}, {"weak_min_support_m": 80.0}),
    ("grouping", "weak: support 60 m or score 6", {}, {"weak_short_min_score": 6.0}),
    ("grouping", "no continuation link", {}, {"cont_max_gap_m": -1e9}),
    ("grouping", "tol_time 0.5 s", {}, {"tol_time_s": 0.5}),
    ("grouping", "tol_time 1.5 s", {}, {"tol_time_s": 1.5}),
    ("grouping", "tol_velocity 4 km/h / 8 %", {}, {"tol_velocity_kmh": 4.0, "tol_velocity_rel": 0.08}),
    ("grouping", "band weights LF = HF", {}, {"band_weights": (("hf", 1.0), ("lf", 1.0))}),
    ("grouping", "band weights LF x4", {}, {"band_weights": (("hf", 1.0), ("lf", 4.0))}),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", default="all", help="segment name from config.SEGMENTS or 'all'")
    p.add_argument("--synthetic", default="all", help="comma-separated synthetic ids or 'all'")
    p.add_argument("--skip-variants", action="store_true", help="no variant study")
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ======================================================================================
# Chain
# ======================================================================================
def enhance_bands(pre, ep: EnhancementParams) -> dict:
    return {b: enhance(img, ep, pre.row_dt, pre.col_dx) for b, img in pre.images.items()}


def detect_and_group(enh: dict, dp: DetectionParams, gp: GroupingParams,
                     ep: EnhancementParams, channel_dx: float) -> dict:
    """Stages 6 + 7 on the enhanced maps of one image. Returns a dict of results + timings."""
    e = next(iter(enh.values()))
    t = time.perf_counter()
    det = detect_candidates(enh, dp, (ep.v_min_kmh, ep.v_max_kmh))
    t_det = time.perf_counter() - t
    width = e.response[1].shape[1] * e.col_dx
    t = time.perf_counter()
    objs, groups = build_objects(det.segments, gp, e.row_dt, e.col_dx, 0.5 * width, channel_dx)
    return {"det": det, "objects": objs, "groups": groups,
            "timings": {"detect_s": t_det, "group_s": time.perf_counter() - t}}


def evaluate_synthetic(objs: list, refs: list[dict]) -> dict:
    """Detection metrics overall and per class (recall only) + velocity errors."""
    m = match_detections(objs, refs, MATCH_TOL_T_S, MATCH_TOL_V_KMH)
    out = detection_metrics(len(objs), len(refs), m)
    matched = {j for _, j in m}
    for name, types in (("light", LIGHT_TYPES), ("heavy", HEAVY_TYPES)):
        idx = [j for j, r in enumerate(refs) if r["type"] in types]
        out[f"n_{name}"] = len(idx)
        out[f"tp_{name}"] = sum(j in matched for j in idx)
        mm = [(i, j) for i, j in m if refs[j]["type"] in types]
        out[f"vel_mae_{name}"] = velocity_errors(objs, refs, mm)["mae"]
    out.update({f"vel_{k}": v for k, v in velocity_errors(objs, refs, m).items()})
    out["matches"] = m
    return out


# ======================================================================================
# Figures
# ======================================================================================
def _group_colours(groups) -> tuple[list, list]:
    cmap = plt.get_cmap("tab20")
    segs, cols = [], []
    for k, g in enumerate(groups):
        segs += g
        cols += [cmap(k % 20)] * len(g)
    return segs, cols


def figure_candidates(background: np.ndarray, res: dict, row_dt: float, col_dx: float, t0,
                      title: str, path, dpi: int, refs: list[dict] | None = None) -> None:
    """Three panels: candidates by source -> groups -> mean line per group (+ GT if given)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 9), sharey=True, layout="constrained")
    kw = dict(colorbar=False, vmin=0, vmax=8, cmap="magma")
    plot_waterfall(background, t0, row_dt, col_dx, ax=axes[0], title=
                   f"1. candidates ({len(res['det'].segments)})\nred strong, cyan HoughLinesP, "
                   f"yellow global Hough; dashed = LF", **kw)
    plot_segments(axes[0], res["det"].segments, row_dt, col_dx, lw=0.8)
    plot_waterfall(background, t0, row_dt, col_dx, ax=axes[1],
                   title=f"2. groups ({len(res['groups'])}), one colour per group", **kw)
    segs, cols = _group_colours(res["groups"])
    plot_segments(axes[1], segs, row_dt, col_dx, colours=cols, lw=1.0)
    n_s = sum(o.kind == "strong" for o in res["objects"])
    plot_detections(background, res["objects"], row_dt, col_dx, t0, ax=axes[2],
                    title=f"3. mean line per group after clean-up\n{n_s} strong (solid), "
                          f"{len(res['objects']) - n_s} weak (dashed)", vmin=0, vmax=8, cmap="gray")
    if refs is not None:
        for r in refs:
            axes[2].plot(r["track"][:, 0], r["track"][:, 1], color="lime", lw=0.6, alpha=0.8)
        axes[2].set_title(axes[2].get_title() + "\nlime = ground truth", fontsize=10)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle(title)
    save_figure(fig, path, dpi)


def figure_paths(res: dict, enh: dict, band: str, direction: int, row_dt: float,
                 col_dx: float, t0, title: str, path, dpi: int) -> None:
    """Before / after of every step of the strong (6A) and weak (6B) path for one map."""
    dn = "pos" if direction > 0 else "neg"
    st = res["det"].stages
    thr = res["det"].thresholds.get(f"{band}_{dn}", {})
    panels = [("response (Stage 5)", enh[band].response[direction], "magma", (0, 8))]
    if f"{band}_{dn}_strong_binary" in st:
        panels += [(f"6A threshold > {thr['strong']:.1f}", st[f"{band}_{dn}_strong_binary"], "gray", (0, 1)),
                   ("6A oriented opening", st[f"{band}_{dn}_strong_opened"], "gray", (0, 1)),
                   ("6A strong mask (kept)", st[f"{band}_{dn}_strong_mask"], "gray", (0, 1))]
    panels += [("6B masked input", st[f"{band}_{dn}_weak_input"], "magma", (0, 8)),
               (f"6B hysteresis {thr.get('weak_low', 0):.1f}/{thr.get('weak_high', 0):.1f}",
                st[f"{band}_{dn}_weak_hysteresis"], "gray", (0, 1)),
               ("6B skeleton + segments", st[f"{band}_{dn}_weak_skeleton"], "gray", (0, 1))]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 8.5), sharey=True,
                             layout="constrained")
    for ax, (name, img, cmap, (lo, hi)) in zip(axes, panels):
        plot_waterfall(np.asarray(img, dtype=np.float32), t0, row_dt, col_dx, ax=ax,
                       colorbar=False, cmap=cmap, vmin=lo, vmax=hi, title=name)
        ax.set_xlabel("x [m]")
        if ax is not axes[0]:
            ax.set_ylabel("")
    segs = [s for s in res["det"].segments if s.direction == direction and s.band == band]
    plot_segments(axes[-1], [s for s in segs if s.source != "strong"], row_dt, col_dx, lw=0.8)
    if len(panels) > 4:
        plot_segments(axes[3], [s for s in segs if s.source == "strong"], row_dt, col_dx, lw=1.2)
    fig.suptitle(title)
    save_figure(fig, path, dpi)


# ======================================================================================
# Runs
# ======================================================================================
def run_synthetic(ids, pp, ep, dp, gp, dpi, cache) -> list[dict]:
    """Stages 4-7 on the synthetic samples: candidates, objects, metrics, figures."""
    rows = []
    for sid in ids:
        sig, gt, meta = dio.load_synthetic(sid)
        dt, dx = dio.synthetic_axes(meta)
        t = time.perf_counter()
        pre = preprocess(sig, pp, 1.0 / dt, dx)
        enh = enhance_bands(pre, ep)
        t_enh = time.perf_counter() - t
        refs = gt_objects_from_mask(gt, meta, dt, dx)
        cache["synthetic"][sid] = (enh, refs, dx)
        res = detect_and_group(enh, dp, gp, ep, dx)
        ev = evaluate_synthetic(res["objects"], refs)
        e = enh["hf"]
        bg = np.maximum(enh["hf"].combined, enh["lf"].combined)
        figure_candidates(bg, res, e.row_dt, e.col_dx, None,
                          f"Synthetic {sid}: Stage 6 candidates -> Stage 7 groups -> objects "
                          f"(recall light {ev['tp_light']}/{ev['n_light']}, heavy "
                          f"{ev['tp_heavy']}/{ev['n_heavy']}, {ev['fp']} FP)",
                          FIG_DIR / f"synthetic_{sid}.png", dpi, refs)
        rows.append({"sample": sid, "n_candidates": len(res["det"].segments),
                     "n_groups": len(res["groups"]), "n_objects": len(res["objects"]),
                     **{k: v for k, v in ev.items() if k != "matches"},
                     "enhance_s": t_enh, **res["timings"]})
        r = rows[-1]
        print(f"  {sid}: light {r['tp_light']}/{r['n_light']}  heavy {r['tp_heavy']}/{r['n_heavy']}  "
              f"FP {r['fp']}  P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f}  "
              f"|v| MAE {r['vel_mae']:.1f} km/h  ({r['detect_s']:.1f}+{r['group_s']:.2f} s)")
    return rows


def run_real(specs, pp, ep, dp, gp, dpi, cache) -> list[dict]:
    """Stages 4-7 on the real segments: candidates, objects and figures."""
    rows = []
    for spec in specs:
        seg = dio.load_named_segment(spec["name"], pad_files=1)
        pre = preprocess_segment(seg, pp)
        enh = enhance_bands(pre, ep)
        cache["real"][spec["name"]] = (enh, seg.dx)
        res = detect_and_group(enh, dp, gp, ep, seg.dx)
        e = enh["hf"]
        out_dir = FIG_DIR / spec["name"]
        title = f"{spec['name']} {seg.start:%H:%M:%S}-{seg.end:%H:%M:%S}"
        bg = np.maximum(enh["hf"].combined, enh["lf"].combined)
        figure_candidates(bg, res, e.row_dt, e.col_dx, seg.start,
                          f"{title}: Stage 6 candidates -> Stage 7 groups -> objects",
                          out_dir / "candidates_groups_objects.png", dpi)
        for band in ("hf", "lf"):
            for d in (1, -1):
                figure_paths(res, enh, band, d, e.row_dt, e.col_dx, seg.start,
                             f"{title} - Stage 6 on {band.upper()} v {'>' if d > 0 else '<'} 0: "
                             f"strong path (6A) and weak path (6B)",
                             out_dir / f"stage6_{band}_{'pos' if d > 0 else 'neg'}.png", dpi)
        raw = np.log10(pre.common["raw"] + 1e-12)
        fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True, layout="constrained")
        plot_detections(raw, res["objects"], pre.row_dt, pre.col_dx, seg.start, ax=axes[0],
                        vmin=float(np.percentile(raw, 1)), vmax=float(np.percentile(raw, 99.5)),
                        title="raw log RMS + detected objects")
        plot_detections(bg, res["objects"], e.row_dt, e.col_dx, seg.start, ax=axes[1],
                        vmin=0, vmax=8, cmap="magma", title="enhanced (max HF/LF, both directions)")
        axes[1].set_ylabel("")
        fig.suptitle(f"{title}: detected objects (solid = strong/thick, dashed = weak/thin)")
        save_figure(fig, out_dir / "objects.png", dpi)
        write_objects_csv(res["objects"], config.RESULTS_DIR / f"detection_objects_{spec['name']}.csv",
                          seg.start)
        objs = res["objects"]
        rows.append({"segment": spec["name"], "n_candidates": len(res["det"].segments),
                     "candidates_by_source": dict(Counter(f"{s.source}/{s.band}" for s in res["det"].segments)),
                     "n_groups": len(res["groups"]), "n_objects": len(objs),
                     "n_strong": sum(o.kind == "strong" for o in objs),
                     "n_weak": sum(o.kind == "weak" for o in objs),
                     "n_pos": sum(o.direction > 0 for o in objs),
                     "n_neg": sum(o.direction < 0 for o in objs),
                     "thresholds": res["det"].thresholds, **res["timings"]})
        r = rows[-1]
        print(f"  {spec['name']}: {r['n_candidates']} candidates -> {r['n_groups']} groups -> "
              f"{r['n_objects']} objects ({r['n_strong']} strong, {r['n_weak']} weak; "
              f"{r['n_pos']} v>0, {r['n_neg']} v<0)  ({r['detect_s']:.1f}+{r['group_s']:.2f} s)")
    return rows


def variant_study(cache, ep, dp0, gp0) -> list[dict]:
    """Rerun detection + grouping for every variant on the cached response maps."""
    rows = []
    for group, name, d_over, g_over in VARIANTS:
        dp, gp = replace(dp0, **d_over), replace(gp0, **g_over)
        tot = Counter()
        mae = []
        t = time.perf_counter()
        for enh, refs, dx in cache["synthetic"].values():
            res = detect_and_group(enh, dp, gp, ep, dx)
            ev = evaluate_synthetic(res["objects"], refs)
            for k in ("tp", "fp", "fn", "tp_light", "n_light", "tp_heavy", "n_heavy"):
                tot[k] += ev[k]
            mae += [abs(abs(res["objects"][i].velocity_kmh) - abs(refs[j]["velocity_kmh"]))
                    for i, j in ev["matches"]]
        n_real = Counter()
        for enh, dx in cache["real"].values():
            objs = detect_and_group(enh, dp, gp, ep, dx)["objects"]
            n_real["strong"] += sum(o.kind == "strong" for o in objs)
            n_real["weak"] += sum(o.kind == "weak" for o in objs)
        prec = tot["tp"] / max(tot["tp"] + tot["fp"], 1)
        rec = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
        row = {"group": group, "variant": name,
               "recall_light": tot["tp_light"] / max(tot["n_light"], 1),
               "recall_heavy": tot["tp_heavy"] / max(tot["n_heavy"], 1),
               "precision": prec, "recall": rec,
               "f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
               "fp": tot["fp"], "vel_mae_kmh": float(np.mean(mae)) if mae else float("nan"),
               "real_strong": n_real["strong"], "real_weak": n_real["weak"],
               "time_s": time.perf_counter() - t}
        rows.append(row)
        print(f"  {name:<42} R light {row['recall_light']:.3f} heavy {row['recall_heavy']:.3f}  "
              f"P {row['precision']:.3f}  F1 {row['f1']:.3f}  FP {row['fp']:>3}  MAE "
              f"{row['vel_mae_kmh']:.2f}  real {row['real_strong']}+{row['real_weak']}")
    return rows


def plot_variants(rows: list[dict], dpi: int) -> None:
    """Recall / precision / F1 of the detection variants."""
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.36 * len(rows) + 2), sharey=True,
                             layout="constrained")
    for k, (key, mk) in enumerate((("recall_light", "o"), ("recall_heavy", "s"),
                                   ("precision", "^"), ("f1", "*"))):
        axes[0].scatter([r[key] for r in rows], y + (k - 1.5) * 0.15, marker=mk, label=key)
    axes[0].axvline(rows[0]["f1"], color="gray", ls=":")
    axes[0].set_title("synthetic (6 samples): recall / precision / F1")
    axes[0].legend(fontsize=8)
    axes[1].scatter([r["vel_mae_kmh"] for r in rows], y, color="C4")
    axes[1].set_title("|v| MAE of matched objects [km/h]")
    axes[2].barh(y - 0.2, [r["real_strong"] for r in rows], height=0.4, label="strong")
    axes[2].barh(y + 0.2, [r["real_weak"] for r in rows], height=0.4, label="weak")
    axes[2].set_title("objects on the 3 real segments")
    axes[2].legend(fontsize=8)
    axes[0].set_yticks(y, [r["variant"] for r in rows], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Stages 6-7 variant study (one change at a time vs the default; dotted = default F1)")
    save_figure(fig, FIG_DIR / "variant_study.png", dpi)


# ======================================================================================
def main() -> None:
    args = parse_args()
    pp, ep, dp, gp = PreprocessingParams(), EnhancementParams(), DetectionParams(), GroupingParams()
    ids = dio.list_synthetic_ids() if args.synthetic == "all" else args.synthetic.split(",")
    specs = config.SEGMENTS if args.segment == "all" else [dio.get_segment_spec(args.segment)]
    cache = {"synthetic": {}, "real": {}}

    print(f"[1/3] synthetic samples {ids}")
    syn = run_synthetic(ids, pp, ep, dp, gp, args.dpi, cache)
    tot = {k: sum(r[k] for r in syn) for k in ("tp", "fp", "fn", "tp_light", "n_light",
                                               "tp_heavy", "n_heavy")}
    print(f"  TOTAL: recall light {tot['tp_light']}/{tot['n_light']}, heavy "
          f"{tot['tp_heavy']}/{tot['n_heavy']}, FP {tot['fp']}")
    print(f"[2/3] real segments {[s['name'] for s in specs]}")
    real = run_real(specs, pp, ep, dp, gp, args.dpi, cache)

    summary = {"detection_params": asdict(dp), "grouping_params": asdict(gp),
               "match_tolerances": {"t_s": MATCH_TOL_T_S, "v_kmh": MATCH_TOL_V_KMH},
               "synthetic": syn, "synthetic_total": tot, "real": real}
    path = config.RESULTS_DIR / "detection_synthetic.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(syn[0].keys()))
        w.writeheader()
        w.writerows(syn)
    if not args.skip_variants:
        print(f"[3/3] variant study ({len(VARIANTS)} variants)")
        rows = variant_study(cache, ep, dp, gp)
        with open(config.RESULTS_DIR / "detection_variants.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        plot_variants(rows, args.dpi)
        summary["variants"] = rows
    (config.RESULTS_DIR / "detection_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=1), encoding="utf-8")
    print(f"\nFigures: {FIG_DIR}\nResults: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
