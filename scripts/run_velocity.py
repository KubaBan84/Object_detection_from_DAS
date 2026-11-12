"""Stages 8-9: velocity of every object and velocity over time (v(t), extra point 1).

Runs the pipeline (Stages 4-9, ``das.pipeline.run_pipeline``) and

1. synthetic samples: matches the objects with the ground truth and compares
   - the line velocity (Stage 7) and the track velocity (Stage 8, LSQ x(t) on the
     ridge track) with the true mean velocity, per vehicle class;
   - the velocity profile v(t) with the GT profile, the acceleration of the quadratic
     fit and the constant / varying speed label;
2. real segments: figures (velocities as in the brief, curved tracks, combined v(t),
   per-object v(t) gallery) and the object table with R^2 and acceleration;
3. variant study of the tracking options (energy map, band, window, ...), re-tracking the
   same objects.

Outputs: ``outputs/figures/velocity/``, ``outputs/results/velocity_*.csv`` and
``outputs/results/velocity_summary.json``.

Plan: Stages 8 and 9.
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
from das.config import PipelineParams, TrackingParams  # noqa: E402
from das.evaluation import (HEAVY_TYPES, LIGHT_TYPES, gt_objects_from_mask,  # noqa: E402
                            match_detections, profile_errors)
from das.pipeline import retrack, run_pipeline, run_segment  # noqa: E402
from das.tracking import velocity_profile  # noqa: E402
from das.velocity import write_objects_csv  # noqa: E402
from das.visualization import (plot_detections, plot_profile_gallery,  # noqa: E402
                               plot_velocity_profiles, save_figure)

FIG_DIR = config.FIG_DIR / "velocity"
MATCH_TOL_T_S, MATCH_TOL_V_KMH = 1.5, 20.0     # same matching as Stages 6-7
REF_WINDOW = 3                                 # GT v(t): 3-channel window (~instantaneous)

# (group, name, tracking overrides) - relative to the defaults
VARIANTS = [
    ("baseline", "default", {}),
    ("map", "track on the filter-bank response", {"track_map": "response"}),
    ("map", "band HF only", {"track_band": "hf"}),
    ("map", "band LF only", {"track_band": "lf"}),
    ("map", "band auto (LF if detected in LF)", {"track_band": "auto"}),
    ("map", "no time smoothing", {"smooth_rows": 0.0}),
    ("map", "time smoothing 2 rows", {"smooth_rows": 2.0}),
    ("search", "peak: centroid of the upper half", {"peak_method": "centroid"}),
    ("search", "pass-1 window +-2 s", {"search_halfwidth_s": 2.0}),
    ("search", "pass-2 window +-0.3 s", {"refine_halfwidth_s": 0.3}),
    ("search", "min peak z 1.0", {"min_peak_z": 1.0}),
    ("search", "min peak z 2.5", {"min_peak_z": 2.5}),
    ("search", "RANSAC line (order 1)", {"poly_order": 1}),
    ("search", "RANSAC cubic (order 3)", {"poly_order": 3}),
    ("profile", "window 5 channels", {"velocity_window_channels": 5}),
    ("profile", "window 7 channels", {"velocity_window_channels": 7}),
    ("profile", "window 11 channels", {"velocity_window_channels": 11}),
    ("profile", "window 15 channels", {"velocity_window_channels": 15}),
    ("class", "varying: dv 10 %", {"varying_dv_rel": 0.10}),
    ("class", "varying: dv 20 %", {"varying_dv_rel": 0.20}),
    ("class", "varying: RMS gain 20 %", {"varying_rms_gain": 0.2}),
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
# Evaluation against the synthetic ground truth
# ======================================================================================
def evaluate(objs: list, refs: list[dict], tp: TrackingParams) -> dict:
    """Per matched object: velocity errors (line / track), v(t) error, class labels."""
    rows = []
    for i, j in match_detections(objs, refs, MATCH_TOL_T_S, MATCH_TOL_V_KMH):
        o, r = objs[i], refs[j]
        pe = profile_errors(o, r, tp, REF_WINDOW)
        rows.append({"type": r["type"], "heavy": r["type"] in HEAVY_TYPES,
                     "v_true": abs(r["velocity_kmh"]),
                     "err_line": abs(abs(o.velocity_kmh) - abs(r["velocity_kmh"])),
                     "err_track": abs(abs(o.v_track_kmh) - abs(r["velocity_kmh"])),
                     "r2": o.r2, **pe, "obj": i, "ref": j})
    return {"rows": rows}


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-object rows over samples (NaN-safe means; class confusion)."""
    def mean(key, sel=None):
        v = np.array([r[key] for r in rows if sel is None or sel(r)], dtype=float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    tracked = [r for r in rows if np.isfinite(r["err_track"])]
    conf = Counter((r["class_gt"], r["class_det"]) for r in tracked)
    out = {"n_matched": len(rows), "n_tracked": len(tracked),
           "vel_mae_line": mean("err_line"), "vel_mae_track": mean("err_track"),
           "vel_mae_line_tracked": mean("err_line", lambda r: np.isfinite(r["err_track"])),
           "vt_mae": mean("vt_mae_kmh"), "accel_err_med": float(np.nanmedian(
               [r["accel_err_ms2"] for r in tracked])) if tracked else float("nan"),
           "coverage": mean("coverage", lambda r: np.isfinite(r["err_track"])),
           "class_agree": (conf[("constant", "constant")] + conf[("varying", "varying")])
           / max(sum(conf.values()), 1),
           "confusion": {f"gt_{a}->det_{b}": n for (a, b), n in sorted(conf.items())}}
    for name, heavy in (("light", False), ("heavy", True)):
        out[f"vel_mae_line_{name}"] = mean("err_line", lambda r: r["heavy"] == heavy)
        out[f"vel_mae_track_{name}"] = mean("err_track", lambda r: r["heavy"] == heavy)
        out[f"vt_mae_{name}"] = mean("vt_mae_kmh", lambda r: r["heavy"] == heavy)
    return out


# ======================================================================================
# Figures
# ======================================================================================
def _bg(res) -> np.ndarray:
    return np.maximum.reduce([e.combined for e in res.enhanced.values()])


def figure_tracks(res, t0, title, path, dpi, refs=None) -> None:
    """Enhanced map with the straight averaged lines and the measured curved tracks."""
    bg = _bg(res)
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True, layout="constrained")
    plot_detections(bg, res.objects, res.row_dt, res.col_dx, t0, ax=axes[0], vmin=0, vmax=8,
                    cmap="gray", title="Stage 7 averaged lines (red) - velocity label = track fit")
    plot_detections(bg, res.objects, res.row_dt, res.col_dx, t0, ax=axes[1], vmin=0, vmax=8,
                    cmap="gray", tracks=True, lines=False, label=False,
                    title="Stage 9 tracks: blue constant, orange varying speed")
    if refs is not None:
        for r in refs:
            axes[1].plot(r["track"][:, 0], r["track"][:, 1], color="lime", lw=0.6, alpha=0.7)
        axes[1].set_title(axes[1].get_title() + "\nlime = ground truth", fontsize=10)
    axes[1].set_ylabel("")
    fig.suptitle(title)
    save_figure(fig, path, dpi)


def figure_velocities(res, t0, title, path, dpi) -> None:
    """The result as in the brief: raw waterfall + red lines + velocity labels."""
    raw = np.log10(res.pre.common["raw"] + 1e-12)
    fig, ax = plt.subplots(figsize=(7.5, 10), layout="constrained")
    plot_detections(raw, res.objects, res.pre.row_dt, res.pre.col_dx, t0, ax=ax,
                    vmin=float(np.percentile(raw, 1)), vmax=float(np.percentile(raw, 99.5)),
                    title=title, label_fontsize=8)
    save_figure(fig, path, dpi)


def gt_profiles(objs, refs, matches) -> dict:
    """{obj_id: (t, v)} GT profiles of the matched objects (for the gallery)."""
    out = {}
    for i, j in matches:
        t, v = velocity_profile(refs[j]["track"], REF_WINDOW)
        out[objs[i].obj_id] = (t, v)
    return out


# ======================================================================================
# Runs
# ======================================================================================
def run_synthetic(ids, pp: PipelineParams, dpi: int, cache: dict) -> tuple[list, dict]:
    """Stages 4-9 on the synthetic samples: velocities and v(t) vs the GT."""
    rows, per_sample = [], {}
    for sid in ids:
        sig, gt, meta = dio.load_synthetic(sid)
        dt, dx = dio.synthetic_axes(meta)
        res = run_pipeline(sig, pp, 1.0 / dt, dx)
        refs = gt_objects_from_mask(gt, meta, dt, dx)
        cache["synthetic"][sid] = (res, refs)
        ev = evaluate(res.objects, refs, pp.trk)
        for r in ev["rows"]:
            r["sample"] = sid
        rows += ev["rows"]
        s = summarise(ev["rows"])
        per_sample[sid] = {k: v for k, v in s.items() if k != "confusion"}
        print(f"  {sid}: tracked {s['n_tracked']}/{s['n_matched']}  |v| MAE line {s['vel_mae_line']:.2f}"
              f" -> track {s['vel_mae_track']:.2f} km/h  v(t) MAE {s['vt_mae']:.2f} km/h  "
              f"class agree {s['class_agree']:.2f}  (track {res.timings['track_s']:.2f} s)")
        figure_tracks(res, None, f"Synthetic {sid}: Stage 9 tracks vs ground truth",
                      FIG_DIR / f"synthetic_{sid}_tracks.png", dpi, refs)
        matches = [(r["obj"], r["ref"]) for r in ev["rows"]]
        fig = plot_profile_gallery([res.objects[i] for i, _ in matches], None, ncols=7,
                                   references=gt_profiles(res.objects, refs, matches),
                                   title=f"Synthetic {sid}: v(t) of matched objects (dots), "
                                         f"quadratic fit (line), constant fit (dotted), GT (black)")
        save_figure(fig, FIG_DIR / f"synthetic_{sid}_profiles.png", dpi)
    return rows, per_sample


def run_real(specs, pp: PipelineParams, dpi: int, cache: dict) -> list[dict]:
    """Stages 4-9 on the real segments: tracks, v(t) and figures."""
    out = []
    for spec in specs:
        seg = dio.load_named_segment(spec["name"], pad_files=1)
        res = run_segment(seg, pp)
        cache["real"][spec["name"]] = res
        d = FIG_DIR / spec["name"]
        title = f"{spec['name']} {seg.start:%H:%M:%S}-{seg.end:%H:%M:%S}"
        figure_velocities(res, seg.start, f"{title}: detected objects and velocities "
                          f"(solid = strong, dashed = weak)", d / "velocities.png", dpi)
        figure_tracks(res, seg.start, f"{title}: averaged lines vs measured tracks",
                      d / "tracks.png", dpi)
        ax = plot_velocity_profiles(res.objects, seg.start,
                                    title=f"{title}: velocity over time of every object")
        save_figure(ax.figure, d / "velocity_profiles.png", dpi)
        fig = plot_profile_gallery(res.objects, seg.start, ncols=6,
                                   title=f"{title}: v(t) per object - local profile (dots), "
                                         f"quadratic fit (line), constant fit (dotted)")
        save_figure(fig, d / "profile_gallery.png", dpi)
        write_objects_csv(res.objects, config.RESULTS_DIR / f"velocity_objects_{spec['name']}.csv",
                          seg.start)
        objs = res.objects
        tracked = [o for o in objs if np.isfinite(o.v_track_kmh)]
        row = {"segment": spec["name"], "n_objects": len(objs), "n_tracked": len(tracked),
               "n_varying": sum(o.speed_class == "varying" for o in objs),
               "median_r2": float(np.median([o.r2 for o in tracked])) if tracked else float("nan"),
               "median_abs_v_kmh": float(np.median([abs(o.v_track_kmh) for o in tracked]))
               if tracked else float("nan"),
               "max_abs_accel_ms2": float(np.max([abs(o.accel_ms2) for o in tracked]))
               if tracked else float("nan"),
               "timings_s": res.timings}
        out.append(row)
        print(f"  {spec['name']}: {row['n_objects']} objects, {row['n_tracked']} tracked, "
              f"{row['n_varying']} varying speed, median R^2 {row['median_r2']:.3f}, median |v| "
              f"{row['median_abs_v_kmh']:.1f} km/h (track {res.timings['track_s']:.2f} s)")
    return out


def variant_study(cache: dict, tp0: TrackingParams) -> list[dict]:
    """Rerun the tracking for every variant on the cached objects."""
    rows = []
    for group, name, over in VARIANTS:
        tp = replace(tp0, **over)
        t = time.perf_counter()
        ev_rows = []
        for res, refs in cache["synthetic"].values():
            ev_rows += evaluate(retrack(res, tp), refs, tp)["rows"]
        s = summarise(ev_rows)
        real = Counter()
        for res in cache["real"].values():
            objs = retrack(res, tp)
            real["tracked"] += sum(np.isfinite(o.v_track_kmh) for o in objs)
            real["varying"] += sum(o.speed_class == "varying" for o in objs)
        row = {"group": group, "variant": name, **{k: v for k, v in s.items() if k != "confusion"},
               "real_tracked": real["tracked"], "real_varying": real["varying"],
               "time_s": time.perf_counter() - t}
        rows.append(row)
        print(f"  {name:<36} tracked {s['n_tracked']}/{s['n_matched']}  |v| MAE track "
              f"{s['vel_mae_track']:.2f}  v(t) MAE {s['vt_mae']:.2f}  accel err {s['accel_err_med']:.3f}"
              f"  class {s['class_agree']:.2f}  cov {s['coverage']:.2f}  real {real['tracked']} "
              f"tracked / {real['varying']} varying")
    return rows


def plot_variants(rows: list[dict], dpi: int) -> None:
    """|v| and v(t) errors of the tracking variants."""
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.36 * len(rows) + 2), sharey=True,
                             layout="constrained")
    axes[0].scatter([r["vel_mae_track"] for r in rows], y, label="|v| MAE (track fit)")
    axes[0].scatter([r["vt_mae"] for r in rows], y, marker="s", label="v(t) MAE")
    axes[0].axvline(rows[0]["vt_mae"], color="gray", ls=":")
    axes[0].set_title("synthetic: velocity errors [km/h]")
    axes[0].legend(fontsize=8)
    axes[1].scatter([r["class_agree"] for r in rows], y, color="C2", label="class agreement")
    axes[1].scatter([r["coverage"] for r in rows], y, marker="^", color="C3", label="GT coverage")
    axes[1].set_title("synthetic: speed class / track coverage")
    axes[1].legend(fontsize=8)
    axes[2].barh(y - 0.2, [r["real_tracked"] for r in rows], height=0.4, label="tracked")
    axes[2].barh(y + 0.2, [r["real_varying"] for r in rows], height=0.4, label="varying")
    axes[2].set_title("real segments: objects")
    axes[2].legend(fontsize=8)
    axes[0].set_yticks(y, [r["variant"] for r in rows], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Stage 9 variant study (one change at a time vs the default; dotted = default v(t) MAE)")
    save_figure(fig, FIG_DIR / "variant_study.png", dpi)


# ======================================================================================
def main() -> None:
    args = parse_args()
    pp = PipelineParams()
    ids = dio.list_synthetic_ids() if args.synthetic == "all" else args.synthetic.split(",")
    specs = config.SEGMENTS if args.segment == "all" else [dio.get_segment_spec(args.segment)]
    cache = {"synthetic": {}, "real": {}}

    print(f"[1/3] synthetic samples {ids}")
    rows, per_sample = run_synthetic(ids, pp, args.dpi, cache)
    total = summarise(rows)
    print(f"  TOTAL: tracked {total['n_tracked']}/{total['n_matched']}, |v| MAE line "
          f"{total['vel_mae_line']:.2f} -> track {total['vel_mae_track']:.2f} km/h (light "
          f"{total['vel_mae_track_light']:.2f}, heavy {total['vel_mae_track_heavy']:.2f}), v(t) MAE "
          f"{total['vt_mae']:.2f} km/h, accel error (median) {total['accel_err_med']:.3f} m/s^2, "
          f"class agreement {total['class_agree']:.2f} {total['confusion']}")
    with open(config.RESULTS_DIR / "velocity_synthetic_objects.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[2/3] real segments {[s['name'] for s in specs]}")
    real = run_real(specs, pp, args.dpi, cache)
    summary = {"tracking_params": asdict(pp.trk), "ref_window_channels": REF_WINDOW,
               "synthetic_total": total, "synthetic_per_sample": per_sample, "real": real}
    if not args.skip_variants:
        print(f"[3/3] variant study ({len(VARIANTS)} variants)")
        vrows = variant_study(cache, pp.trk)
        with open(config.RESULTS_DIR / "velocity_variants.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(vrows[0].keys()))
            w.writeheader()
            w.writerows(vrows)
        plot_variants(vrows, args.dpi)
        summary["variants"] = vrows
    (config.RESULTS_DIR / "velocity_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=1), encoding="utf-8")
    print(f"\nFigures: {FIG_DIR}\nResults: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
