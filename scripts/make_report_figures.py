"""Collect (and optionally regenerate) every figure of the report in ``report/figures/``.

1. ``--regenerate``: runs all analysis scripts in order with ``python -W error``
   (EDA, spectral analysis, Stages 4-12, evaluation, Stage-13 studies), so that every
   figure and table reflects the final parameters. This takes long (``--skip-studies``
   skips the slow variant studies and the Stage-13 tuning).
2. Draws the algorithm block diagram (``flowchart.png``, matplotlib - the same blocks as
   the mermaid source in ``docs/flowchart.md``).
3. Copies the figures listed in :data:`FIGURES` from ``outputs/figures/`` downscaled
   with ``cv2.INTER_AREA`` to at most ``--max-width`` pixels; large raster-like figures
   (waterfalls) are stored as JPEG so that the HTML report stays small.

Usage::

    uv run python scripts/make_report_figures.py                 # collect + flowchart
    uv run python scripts/make_report_figures.py --regenerate    # rerun everything first

Plan: Stage 14.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from das import config  # noqa: E402

REPORT_FIG_DIR = config.ROOT / "report" / "figures"
SCRIPTS = config.ROOT / "scripts"

# (script, arguments, part of the slow studies)
PIPELINE_RUNS: list[tuple[str, list[str], bool]] = [
    ("run_eda.py", [], False),
    ("run_spectral.py", [], False),
    ("run_enhancement.py", ["--skip-variants"], False),
    ("run_enhancement.py", [], True),
    ("run_detection.py", ["--skip-variants"], False),
    ("run_detection.py", [], True),
    ("run_velocity.py", ["--skip-variants"], False),
    ("run_velocity.py", [], True),
    ("run_clustering.py", [], False),
    ("evaluate_synthetic.py", [], False),
    ("evaluate_real.py", [], False),
    ("run_pipeline.py", ["--segment", "all"], False),
    ("tune_parameters.py", [], True),
]

# report name -> source in outputs/figures (all three segments where the report shows them)
FIGURES: dict[str, str] = {
    # 1-2 data and EDA
    "eda_global_overview": "eda/global_overview.png",
    "eda_global_histograms": "eda/global_histograms.png",
    "eda_global_channel_profile": "eda/global_channel_profile.png",
    "eda_global_file_channel_rms": "eda/global_file_channel_rms.png",
    "eda_global_activity_timeline": "eda/global_activity_timeline.png",
    "eda_seg_kuba_waterfall": "eda/seg_kuba/waterfall.png",
    "eda_seg_own_raw_zoom": "eda/seg_own/raw_zoom.png",
    "eda_seg_kuba_variability": "eda/seg_kuba/variability.png",
    "eda_seg_kuba_histogram": "eda/seg_kuba/histogram.png",
    "eda_seg_kuba_correlation": "eda/seg_kuba/correlation.png",
    # 3 frequency analysis
    "spectral_global_psd": "spectral/global_psd.png",
    "spectral_seg_kuba_fft": "spectral/seg_kuba/fft.png",
    "spectral_seg_kuba_stft": "spectral/seg_kuba/stft.png",
    "spectral_seg_kuba_fk": "spectral/seg_kuba/fk.png",
    "spectral_band_contrast": "spectral/band_contrast.png",
    "spectral_band_selection": "spectral/band_selection.png",
    "spectral_seg_kuba_band_comparison": "spectral/seg_kuba/band_comparison.png",
    # 5-6 preprocessing and enhancement
    "stage4_seg_kuba_hf": "enhancement/seg_kuba/stage4_hf.png",
    "stage4_seg_kuba_lf": "enhancement/seg_kuba/stage4_lf.png",
    "stage5_seg_kuba_hf": "enhancement/seg_kuba/stage5_hf.png",
    "stage5_seg_kuba_lf": "enhancement/seg_kuba/stage5_lf.png",
    "stage5_seg_own_final_maps": "enhancement/seg_own/final_maps.png",
    "stage5_synthetic_40": "enhancement/synthetic_40.png",
    "stage5_variant_study": "enhancement/variant_study.png",
    # 7 detection and grouping
    "stage6_seg_own_hf_neg": "detection/seg_own/stage6_hf_neg.png",
    "stage6_seg_own_lf_neg": "detection/seg_own/stage6_lf_neg.png",
    "stage7_seg_own_candidates": "detection/seg_own/candidates_groups_objects.png",
    "stage6_synthetic_42": "detection/synthetic_42.png",
    "stage6_variant_study": "detection/variant_study.png",
    # 8-9 velocity and v(t)
    "stage8_seg_kuba_velocities": "velocity/seg_kuba/velocities.png",
    "stage9_seg_own_tracks": "velocity/seg_own/tracks.png",
    "stage9_seg_own_profiles": "velocity/seg_own/velocity_profiles.png",
    "stage9_seg_own_gallery": "velocity/seg_own/profile_gallery.png",
    "stage9_synthetic_42_profiles": "velocity/synthetic_42_profiles.png",
    "stage9_variant_study": "velocity/variant_study.png",
    # 10 clustering
    "stage10_feature_separability": "clustering/feature_separability.png",
    "stage10_synthetic_pca": "clustering/synthetic/pca.png",
    "stage10_real_pca": "clustering/real/pca.png",
    "stage10_real_k_selection": "clustering/real/k_selection.png",
    "stage10_real_features": "clustering/real/features.png",
    "stage10_real_gallery": "clustering/real/gallery.png",
    "stage10_seg_own_clusters": "clustering/real/seg_own_clusters.png",
    # 11 results for the three segments
    **{f"result_{s}_{n}": f"pipeline/{s}/{f}" for s in ("seg_kuba", "seg_mateusz", "seg_own")
       for n, f in (("raw", "01_raw.png"), ("result", "05_result.png"),
                    ("velocity", "06_velocity.png"))},
    "result_seg_kuba_enhancement": "pipeline/seg_kuba/03_enhancement.png",
    "result_seg_kuba_candidates": "pipeline/seg_kuba/04_candidates.png",
    "result_timings": "pipeline/timings.png",
    # 12 evaluation and experiments
    "eval_synthetic_summary": "evaluation/synthetic_summary.png",
    "eval_synthetic_42": "evaluation/synthetic_42.png",
    "eval_real_summary": "evaluation/real_summary.png",
    **{f"eval_real_{s}": f"evaluation/real_{s}.png" for s in ("seg_kuba", "seg_mateusz", "seg_own")},
    "tuning_ablation": "tuning/ablation.png",
    "tuning_methods": "tuning/methods.png",
    "tuning_sweep": "tuning/sweep.png",
    "tuning_hysteresis": "tuning/hysteresis.png",
    "tuning_hysteresis_sensitivity": "tuning/hysteresis_sensitivity.png",
    "tuning_grouping": "tuning/grouping.png",
}


# ======================================================================================
# Block diagram
# ======================================================================================
# (key, column 0..2 or "wide", row, text, colour); column "L"/"R" = HF / LF path
_C = {"io": "#d9e8fb", "pre": "#e3f4e1", "enh": "#fdf0d5", "det": "#fbe0e0", "post": "#ece3f7",
      "out": "#e8e8e8"}
NODES = [
    ("load", "W", 0, "Load 12 consecutive 10-s files (2 min) + 1 file padding on each side\n"
     "raw strain rate: 87 500 x 52 samples (dt 1.6 ms, dx 5.1 m)", "io"),
    ("cm", "W", 1, "Stage 4 - per-channel detrend + common-mode removal\n"
     "(median over channels at every time sample)", "pre"),
    ("hf_bp", "L", 2, "HF band-pass 60-312.5 Hz\n(vibration energy: trams, cars)", "pre"),
    ("lf_bp", "R", 2, "LF band-pass 1-8 Hz\n(quasi-static load: sharp car lines)", "pre"),
    ("hf_env", "L", 3, "RMS envelope, x25 decimation (0.04 s rows),\n"
     "crop padding, log10(env / channel median), clip", "pre"),
    ("lf_env", "R", 3, "RMS envelope, x25 decimation (0.04 s rows),\n"
     "crop padding, log10(env / channel median), clip", "pre"),
    ("hf_fk", "L", 4, "Stage 5 - f-k fan filter 5-150 km/h\nseparately for v > 0 and v < 0", "enh"),
    ("lf_fk", "R", 4, "Stage 5 - f-k fan filter 5-150 km/h\nseparately for v > 0 and v < 0", "enh"),
    ("hf_fb", "L", 5, "x8 upsampling along the cable +\nmatched line-filter bank (24 angles, 81 px)"
     "\n-> robust z maps HF+ / HF-", "enh"),
    ("lf_fb", "R", 5, "x8 upsampling along the cable +\nmatched line-filter bank (24 angles, 81 px)"
     "\n-> robust z maps LF+ / LF-", "enh"),
    ("strong", "L", 6, "Stage 6A - thick lines (HF): Otsu >= 4 sigma,\noriented opening, components,"
     " ridge +\nsequential RANSAC -> strong lines + mask", "det"),
    ("weak", "R", 6, "Stage 6B - thin lines (HF + LF): mask strong,\nhysteresis 2 / 4 sigma, skeleton,"
     "\nHoughLinesP + global Hough, orientation check", "det"),
    ("group", "W", 7, "Stage 7 - grouping (same line / continuation links, connected components),\n"
     "robust weighted mean line per group, duplicates + plausibility (5-150 km/h, >= 40 m)", "post"),
    ("track", "W", 8, "Stages 8-9 - ridge tracking on the f-k maps (2 passes, RANSAC quadratic t(x))\n"
     "-> velocity (LSQ x = v t + b), v(t) (9-channel window), acceleration, constant / varying", "post"),
    ("feat", "W", 9, "Stage 10 - object features (line energy, HF SNR, thickness, spectrum, speed change)\n"
     "-> factorized k-means (cv2.kmeans): thick / thin x constant / varying speed", "post"),
    ("out", "W", 10, "Outputs: waterfall with velocities [km/h], object table, v(t) plots, clusters,\n"
     "timings; Stage 11: evaluation vs synthetic GT / manual labels", "out"),
]
EDGES = [("load", "cm"), ("cm", "hf_bp"), ("cm", "lf_bp"), ("hf_bp", "hf_env"), ("lf_bp", "lf_env"),
         ("hf_env", "hf_fk"), ("lf_env", "lf_fk"), ("hf_fk", "hf_fb"), ("lf_fk", "lf_fb"),
         ("hf_fb", "strong"), ("hf_fb", "weak"), ("lf_fb", "weak"), ("strong", "weak"),
         ("strong", "group"), ("weak", "group"), ("group", "track"), ("track", "feat"),
         ("feat", "out")]
REMOVED = {"cm": "removed: common mode\n(cable-wide events)",
           "hf_env": "removed: out-of-band part,\nchannel gain differences",
           "hf_fk": "removed: |v| outside the fan\n(horizontal / vertical stripes,\nhotspot dashes)",
           "strong": "strong mask excluded\nfrom the weak path",
           "group": "removed: fragments,\nduplicates, short / faint"}


def flowchart_figure(path: Path, dpi: int) -> None:
    """Draw the block diagram of the method (nodes, arrows, removed components)."""
    fig, ax = plt.subplots(figsize=(11.5, 15))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-11.0, 0.6)
    ax.axis("off")
    xs = {"W": (0.4, 8.6), "L": (0.4, 4.2), "R": (4.8, 8.6)}
    box = {}
    for key, col, row, text, kind in NODES:
        x0, x1 = xs[col]
        y = -row * 1.05
        h = 0.78 if row in (5, 6) else 0.66
        ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0.04",
                                    fc=_C[kind], ec="0.3", lw=1))
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=8.2)
        box[key] = (x0, x1, y, h)
    for a, b in EDGES:
        ax0, ax1, ay, ah = box[a]
        bx0, bx1, by, bh = box[b]
        if abs(ay - by) < 1e-9:                        # same row: strong -> weak (mask)
            p, q = (ax1, ay), (bx0, by)
        else:
            xa = np.clip((bx0 + bx1) / 2, ax0 + 0.3, ax1 - 0.3)
            xb = np.clip((ax0 + ax1) / 2, bx0 + 0.3, bx1 - 0.3)
            p, q = (xa, ay - ah / 2), (xb, by + bh / 2)
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color="0.25",
                                     lw=1.0, shrinkA=1, shrinkB=1))
    for key, text in REMOVED.items():
        x0, x1, y, h = box[key]
        ax.text(9.0, y, text, ha="left", va="center", fontsize=7.4, color="#a33",
                style="italic")
        ax.plot([8.65, 8.95], [y, y], color="#a33", lw=0.8, ls=":")
    ax.text(0.45, -1.62, "HF path", ha="left", fontsize=9, weight="bold", color="0.35")
    ax.text(8.55, -1.62, "LF path", ha="right", fontsize=9, weight="bold", color="0.35")
    ax.set_title("Detection of moving objects in DAS recordings - block diagram", fontsize=12)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ======================================================================================
# Collect figures
# ======================================================================================
def downscale(src: Path, dst_stem: Path, max_width: int, jpeg_above_kb: int) -> Path:
    """Copy a figure resized to ``max_width`` (INTER_AREA); large files become JPEG."""
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Cannot read {src}")
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(round(h * max_width / w))),
                         interpolation=cv2.INTER_AREA)
    if src.stat().st_size > jpeg_above_kb * 1024:
        dst = dst_stem.with_suffix(".jpg")
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    else:
        dst = dst_stem.with_suffix(".png")
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return dst


def regenerate(skip_studies: bool) -> None:
    """Rerun the result scripts with ``-W error`` so the figures are up to date."""
    for script, extra, slow in PIPELINE_RUNS:
        if slow and skip_studies:
            continue
        if not slow and not skip_studies and any(s == script and sl for s, _, sl in PIPELINE_RUNS):
            continue                                   # the full run follows
        cmd = [sys.executable, "-W", "error", str(SCRIPTS / script), *extra]
        t = time.perf_counter()
        print(f"  running {script} {' '.join(extra)}", flush=True)
        subprocess.run(cmd, check=True, cwd=config.ROOT, stdout=subprocess.DEVNULL)
        print(f"    done in {time.perf_counter() - t:.0f} s", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regenerate", action="store_true", help="rerun all analysis scripts first")
    p.add_argument("--skip-studies", action="store_true",
                   help="with --regenerate: skip the variant studies and Stage-13 tuning")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--max-width", type=int, default=1300)
    p.add_argument("--jpeg-above-kb", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.regenerate:
        print("Regenerating all results (python -W error):")
        regenerate(args.skip_studies)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(REPORT_FIG_DIR.glob("*.png")) + list(REPORT_FIG_DIR.glob("*.jpg")):
        old.unlink()
    flowchart_figure(REPORT_FIG_DIR / "flowchart.png", args.dpi)
    missing, total = [], (REPORT_FIG_DIR / "flowchart.png").stat().st_size
    for name, rel in FIGURES.items():
        src = config.FIG_DIR / rel
        if not src.exists():
            missing.append(rel)
            continue
        total += downscale(src, REPORT_FIG_DIR / name, args.max_width, args.jpeg_above_kb).stat().st_size
    print(f"{len(FIGURES) - len(missing) + 1} figures in {REPORT_FIG_DIR} ({total / 2 ** 20:.1f} MB)")
    if missing:
        print("Missing (run the corresponding script or --regenerate):\n  " + "\n  ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    main()
