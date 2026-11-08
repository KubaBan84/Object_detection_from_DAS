"""Stages 4-5: preprocessing (energy images) and enhancement (moving objects only).

1. For every configured real segment: run the Stage-4 chain (HF and LF band) and the
   Stage-5 chain (f-k fan filter per direction, upsampling, directional filter bank) and
   save before/after figures of every step, including the removed components.
2. Variant study: every preprocessing / enhancement option is changed one at a time
   against the default configuration and scored with
   - pixel AUC against the synthetic ground truth (light = car/van, heavy = tram/truck),
   - line SNR (p99 of the robust z-score) on the real segments.

Outputs: figures in ``outputs/figures/enhancement/``, ``outputs/results/enhancement_variants.csv``
and ``outputs/results/enhancement_summary.json``.

Plan: Stages 4 and 5.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das.config import EnhancementParams, PreprocessingParams  # noqa: E402
from das.enhancement import enhance, line_snr  # noqa: E402
from das.evaluation import enhancement_auc, gt_envelope_mask  # noqa: E402
from das.preprocessing import preprocess, preprocess_segment  # noqa: E402
from das.visualization import plot_stages, plot_waterfall, save_figure  # noqa: E402

FIG_DIR = config.FIG_DIR / "enhancement"
SYNTH_IDS = ("40", "42", "44")

# The study is defined relative to the INITIAL configuration (before the study), so that
# it stays reproducible after the winning settings became the defaults in config.py.
INITIAL_PP = PreprocessingParams()
INITIAL_EP = replace(EnhancementParams(), v_min_kmh=10.0, fk_pad_mode="reflect",
                     kernel_length_px=41)
_BEST = {"fk_pad_mode": "zero", "kernel_length_px": 81, "v_min_kmh": 5.0}

# (group, name, preprocessing overrides, enhancement overrides) - relative to INITIAL_*
VARIANTS = [
    ("baseline", "initial configuration", {}, {}),
    ("stage4", "envelope = abs_mean", {"envelope": "abs_mean"}, {}),
    ("stage4", "envelope = decimate (no envelope)", {"envelope": "decimate"}, {}),
    ("stage4", "normalization = robust_z", {"normalization": "robust_z"}, {}),
    ("stage4", "no lower clip", {"clip_low": False}, {}),
    ("stage4", "Gaussian smoothing (2 rows, 0.5 ch)", {"smooth_sigma_t": 2.0, "smooth_sigma_x": 0.5}, {}),
    ("stage4", "no common-mode removal", {"common_mode": "none"}, {}),
    ("stage4", "no detrend", {"detrend": False}, {}),
    ("stage4", "CLAHE", {"clahe": True}, {}),
    ("stage5", "no f-k filter", {}, {"fk_filter": False}),
    ("stage5", "f-k pad = zero", {}, {"fk_pad_mode": "zero"}),
    ("stage5", "f-k pad = tukey", {}, {"fk_pad_mode": "tukey"}),
    ("stage5", "white top-hat background", {}, {"background": True}),
    ("stage5", "enhancer = none (f-k only)", {}, {"enhancer": "none"}),
    ("stage5", "enhancer = sato", {}, {"enhancer": "sato"}),
    ("stage5", "enhancer = meijering", {}, {"enhancer": "meijering"}),
    ("stage5", "enhancer = gabor", {}, {"enhancer": "gabor"}),
    ("stage5", "kernel length 21 px", {}, {"kernel_length_px": 21}),
    ("stage5", "kernel length 81 px", {}, {"kernel_length_px": 81}),
    ("stage5", "kernel width sigma 1", {}, {"kernel_width_sigma": 1.0}),
    ("stage5", "kernel width sigma 4", {}, {"kernel_width_sigma": 4.0}),
    ("stage5", "fan 5-200 km/h", {}, {"v_min_kmh": 5.0, "v_max_kmh": 200.0}),
    # round 2: combinations of the best single changes
    ("round2", "A: pad zero + L81", {}, {"fk_pad_mode": "zero", "kernel_length_px": 81}),
    ("round2", "B: A + fan 5-150 (FINAL)", {}, dict(_BEST)),
    ("round2", "C: A + fan 5-200", {}, {**_BEST, "v_max_kmh": 200.0}),
    ("round2", "D: B + L121", {}, {**_BEST, "kernel_length_px": 121}),
    ("round2", "E: B + L61", {}, {**_BEST, "kernel_length_px": 61}),
    ("round2", "F: B + robust_z", {"normalization": "robust_z"}, dict(_BEST)),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", default="all", help="segment name from config.SEGMENTS or 'all'")
    p.add_argument("--skip-variants", action="store_true", help="only the figures")
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
# 1. Figures for the real segments
# ======================================================================================
def segment_figures(spec: dict, pp: PreprocessingParams, ep: EnhancementParams, dpi: int) -> dict:
    seg = dio.load_named_segment(spec["name"], pad_files=1)
    out_dir = FIG_DIR / spec["name"]
    title = f"{spec['name']} {seg.start:%H:%M:%S}-{seg.end:%H:%M:%S}"
    t = time.perf_counter()
    pre = preprocess_segment(seg, pp)
    timings = {"preprocess_s": time.perf_counter() - t}
    enh = {}
    for band, img in pre.images.items():
        t = time.perf_counter()
        enh[band] = enhance(img, ep, pre.row_dt, pre.col_dx)
        timings[f"enhance_{band}_s"] = time.perf_counter() - t
    log_env = lambda e: np.log10(e + 1e-12)  # noqa: E731

    for band, st in pre.bands.items():
        lo, hi = (pp.hf_low, pp.hf_high) if band == "hf" else (pp.lf_low, pp.lf_high)
        stages = {
            "1 raw (log RMS)": log_env(pre.common["raw"]),
            "removed: common mode": log_env(pre.common["common_mode"]),
            "2 after common mode": log_env(pre.common["after_common_mode"]),
            f"removed: outside {lo:g}-{hi:g} Hz": log_env(st["removed_by_bandpass"]),
            f"3 band {lo:g}-{hi:g} Hz": log_env(st["envelope"]),
            "4 normalised + log + clip": st["image"],
        }
        fig = plot_stages(stages, pre.row_dt, pre.col_dx, seg.start,
                          title=f"{title} - Stage 4 ({band.upper()} path): before / after / removed")
        save_figure(fig, out_dir / f"stage4_{band}.png", dpi)

        e = enh[band]
        stages5 = {
            "input (Stage 4)": e.stages["input"],
            "removed by f-k fan": e.stages["fk_removed"],
            "f-k, v > 0": e.stages["fk_pos"],
            "f-k, v < 0": e.stages["fk_neg"],
            "response v > 0": e.response[1],
            "response v < 0": e.response[-1],
        }
        dxs = {k: (e.col_dx if k.startswith("response") else pre.col_dx) for k in stages5}
        fig = plot_stages(stages5, pre.row_dt, dxs, seg.start,
                          symmetric=("removed by f-k fan", "f-k, v > 0", "f-k, v < 0"),
                          title=f"{title} - Stage 5 ({band.upper()} path): f-k fan filter per "
                                f"direction + directional filter bank (robust z)")
        save_figure(fig, out_dir / f"stage5_{band}.png", dpi)

    # final maps: only moving objects visible
    fig, axes = plt.subplots(1, 5, figsize=(16, 8), sharey=True, layout="constrained")
    raw = log_env(pre.common["raw"])
    plot_waterfall(raw, seg.start, pre.row_dt, pre.col_dx, ax=axes[0], colorbar=False,
                   vmin=float(np.percentile(raw, 1)), vmax=float(np.percentile(raw, 99.5)),
                   title="raw (log RMS)")
    for ax, (band, d, name) in zip(axes[1:], [("hf", 1, "HF, v > 0"), ("hf", -1, "HF, v < 0"),
                                              ("lf", 1, "LF, v > 0"), ("lf", -1, "LF, v < 0")]):
        plot_waterfall(enh[band].response[d], seg.start, pre.row_dt, enh[band].col_dx, ax=ax,
                       colorbar=False, vmin=0, vmax=8, cmap="magma", title=f"{name} (z 0-8)")
        ax.set_ylabel("")
    fig.suptitle(f"{title}: enhanced maps - only moving objects remain")
    save_figure(fig, out_dir / "final_maps.png", dpi)

    return {"name": spec["name"], "timings_s": timings,
            "line_snr": {f"{b}_{'pos' if d > 0 else 'neg'}": line_snr(e.response[d])
                         for b, e in enh.items() for d in (1, -1)}}


def synthetic_figure(pp, ep, dpi) -> dict:
    """Enhanced maps of one synthetic sample next to its ground truth."""
    sig, gt, meta = dio.load_synthetic(SYNTH_IDS[0])
    pre = preprocess(sig, pp, config.FS, meta["axes"]["dx"])
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True, layout="constrained")
    res = {}
    for ax, band in zip(axes[:2], ("hf", "lf")):
        e = enhance(pre.images[band], ep, pre.row_dt, pre.col_dx)
        res[band] = enhancement_auc(e.combined, gt, meta, pp.time_decimation, ep.space_upsample)
        plot_waterfall(e.combined, None, pre.row_dt, e.col_dx, ax=ax, colorbar=False, vmin=0,
                       vmax=8, cmap="magma",
                       title=f"{band.upper()} response\nAUC light {res[band]['light']:.3f}, "
                             f"heavy {res[band]['heavy']:.3f}")
    img = pre.images["lf"]
    plot_waterfall(img, None, pre.row_dt, pre.col_dx, ax=axes[2], colorbar=False,
                   vmin=0, vmax=float(np.percentile(img, 99.5)), title="LF Stage-4 image")
    types = {v["marker_value"]: v["type"] for v in meta["ground_truth"]["vehicles"]}
    heavy = gt_envelope_mask(gt, pp.time_decimation,
                             [m for m, t in types.items() if t in ("tram", "truck")])
    light = gt_envelope_mask(gt, pp.time_decimation,
                             [m for m, t in types.items() if t in ("car", "van")])
    plot_waterfall(heavy * 2.0 + light, None, pre.row_dt, pre.col_dx, ax=axes[3],
                   colorbar=False, cmap="viridis", title="ground truth\n(yellow heavy, teal light)")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle(f"Synthetic sample {SYNTH_IDS[0]}: enhancement vs ground truth")
    save_figure(fig, FIG_DIR / f"synthetic_{SYNTH_IDS[0]}.png", dpi)
    return res


# ======================================================================================
# 2. Variant study
# ======================================================================================
def evaluate(pp: PreprocessingParams, ep: EnhancementParams, synth: list, real: list) -> dict:
    t0 = time.perf_counter()
    auc = {b: {"light": [], "heavy": []} for b in ("hf", "lf", "fused")}
    for sig, gt, meta in synth:
        pre = preprocess(sig, pp, config.FS, meta["axes"]["dx"])
        maps = {b: enhance(img, ep, pre.row_dt, pre.col_dx).combined
                for b, img in pre.images.items()}
        maps["fused"] = np.maximum(maps["hf"], maps["lf"])
        for b, m in maps.items():
            a = enhancement_auc(m, gt, meta, pp.time_decimation, ep.space_upsample)
            auc[b]["light"].append(a["light"])
            auc[b]["heavy"].append(a["heavy"])
    snr = {"hf": [], "lf": []}
    for seg in real:
        pre = preprocess_segment(seg, pp)
        for b, img in pre.images.items():
            snr[b].append(line_snr(enhance(img, ep, pre.row_dt, pre.col_dx).combined)[0])
    row = {f"auc_{c}_{b}": float(np.nanmean(auc[b][c])) for b in auc for c in ("light", "heavy")}
    row.update({f"snr_real_{b}": float(np.mean(v)) for b, v in snr.items()})
    row["time_s"] = time.perf_counter() - t0
    return row


def variant_study(pp0, ep0) -> list[dict]:
    synth = [dio.load_synthetic(s) for s in SYNTH_IDS]
    real = [dio.load_named_segment(s["name"], pad_files=1) for s in config.SEGMENTS]
    rows = []
    for group, name, p_over, e_over in VARIANTS:
        r = evaluate(replace(pp0, **p_over), replace(ep0, **e_over), synth, real)
        rows.append({"group": group, "variant": name, **r})
        print(f"  {name:<38} AUC light hf/lf/fused {r['auc_light_hf']:.3f}/{r['auc_light_lf']:.3f}/"
              f"{r['auc_light_fused']:.3f}  heavy {r['auc_heavy_hf']:.3f}/{r['auc_heavy_lf']:.3f}/"
              f"{r['auc_heavy_fused']:.3f}  SNR real hf/lf {r['snr_real_hf']:.2f}/{r['snr_real_lf']:.2f}"
              f"  ({r['time_s']:.0f} s)")
    return rows


def plot_variants(rows: list[dict], dpi: int) -> None:
    names = [r["variant"] for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.38 * len(rows) + 2), sharey=True,
                             layout="constrained")
    for ax, key, title in ((axes[0], "auc_light", "AUC light (car/van) - synthetic"),
                           (axes[1], "auc_heavy", "AUC heavy (tram/truck) - synthetic")):
        for k, (b, mk) in enumerate((("hf", "o"), ("lf", "s"), ("fused", "*"))):
            ax.scatter([r[f"{key}_{b}"] for r in rows], y + (k - 1) * 0.2, marker=mk, label=b)
        ax.axvline(rows[0][f"{key}_fused"], color="gray", ls=":")
        ax.set_title(title)
        ax.legend(fontsize=8)
    for k, (b, mk) in enumerate((("hf", "o"), ("lf", "s"))):
        axes[2].scatter([r[f"snr_real_{b}"] for r in rows], y + (k - 0.5) * 0.2, marker=mk, label=b)
    axes[2].set_title("line SNR - real segments (p99 robust z)")
    axes[2].legend(fontsize=8)
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Stages 4-5 variant study (one change at a time vs the initial configuration; "
                 "round 2 = combinations; dotted = initial fused)")
    save_figure(fig, FIG_DIR / "variant_study.png", dpi)


# ======================================================================================
def main() -> None:
    args = parse_args()
    pp, ep = PreprocessingParams(), EnhancementParams()
    specs = config.SEGMENTS if args.segment == "all" else [dio.get_segment_spec(args.segment)]
    print(f"[1/3] figures for {[s['name'] for s in specs]}")
    segs = []
    for spec in specs:
        r = segment_figures(spec, pp, ep, args.dpi)
        segs.append(r)
        print(f"  {spec['name']:<12} times " +
              ", ".join(f"{k} {v:.2f}" for k, v in r["timings_s"].items()) + "  line SNR " +
              ", ".join(f"{k} {v[0]:.1f}" for k, v in r["line_snr"].items()))
    print("[2/3] synthetic figure")
    syn = synthetic_figure(pp, ep, args.dpi)
    print("  ", {b: {k: round(v, 3) for k, v in a.items()} for b, a in syn.items()})
    summary = {"preprocessing_params": asdict(pp), "enhancement_params": asdict(ep),
               "segments": segs, "synthetic_example_auc": syn}
    if not args.skip_variants:
        print(f"[3/3] variant study ({len(VARIANTS)} variants, synthetic {SYNTH_IDS} + "
              f"{len(config.SEGMENTS)} real segments)")
        rows = variant_study(INITIAL_PP, INITIAL_EP)
        path = config.RESULTS_DIR / "enhancement_variants.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        plot_variants(rows, args.dpi)
        summary["variants"] = rows
    out = config.RESULTS_DIR / "enhancement_summary.json"
    out.write_text(json.dumps(to_jsonable(summary), indent=1), encoding="utf-8")
    print(f"\nFigures: {FIG_DIR}\nResults: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
