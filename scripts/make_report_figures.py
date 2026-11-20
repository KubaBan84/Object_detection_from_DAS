"""Stage 12: collect (and optionally regenerate) every figure of the report in ``report/figures/``.

1. ``--regenerate``: runs all analysis scripts in order with ``python -W error`` (EDA,
   frequency analysis, the pipeline on the 3 segments, evaluation, experiments), so that
   every figure and table reflects the final parameters (``--skip-experiments`` skips the
   slowest step).
2. Draws the algorithm block diagram (``flowchart.png``, matplotlib - the same blocks as
   the mermaid source in ``docs/flowchart.md``).
3. Copies the figures listed in :data:`FIGURES` from ``outputs/figures/`` downscaled
   with ``cv2.INTER_AREA`` to at most ``--max-width`` pixels; large raster-like figures
   (waterfalls) are stored as JPEG so that the HTML report stays small.

Usage::

    uv run python scripts/make_report_figures.py                 # collect + flowchart
    uv run python scripts/make_report_figures.py --regenerate    # rerun everything first
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from das import config  # noqa: E402

REPORT_FIG_DIR = config.ROOT / "report" / "figures"
SCRIPTS = config.ROOT / "scripts"
SEGS = ("seg_kuba", "seg_mateusz", "seg_own")

# (script, arguments, slow)
RUNS: list[tuple[str, list[str], bool]] = [
    ("run_eda.py", [], False),
    ("run_spectral.py", [], False),
    ("run_pipeline.py", ["--segment", "all"], False),
    ("evaluate_real.py", [], False),
    ("evaluate_synthetic.py", [], False),
    ("run_experiments.py", [], True),
]

# report name -> source in outputs/figures
FIGURES: dict[str, str] = {
    # data and EDA
    "eda_global_overview": "eda/global_overview.png",
    "eda_global_histograms": "eda/global_histograms.png",
    "eda_global_channel_profile": "eda/global_channel_profile.png",
    "eda_global_file_channel_rms": "eda/global_file_channel_rms.png",
    "eda_global_activity_timeline": "eda/global_activity_timeline.png",
    "eda_seg_kuba_waterfall": "eda/seg_kuba/waterfall.png",
    "eda_seg_own_raw_zoom": "eda/seg_own/raw_zoom.png",
    "eda_seg_kuba_variability": "eda/seg_kuba/variability.png",
    "eda_seg_kuba_channel_boxplots": "eda/seg_kuba/channel_boxplots.png",
    "eda_seg_kuba_histogram": "eda/seg_kuba/histogram.png",
    "eda_seg_kuba_correlation": "eda/seg_kuba/correlation.png",
    # frequency analysis
    "spectral_global_psd": "spectral/global_psd.png",
    "spectral_seg_kuba_fft": "spectral/seg_kuba/fft.png",
    "spectral_seg_kuba_stft": "spectral/seg_kuba/stft.png",
    "spectral_seg_kuba_fk": "spectral/seg_kuba/fk.png",
    "spectral_band_contrast": "spectral/band_contrast.png",
    "spectral_band_selection": "spectral/band_selection.png",
    "spectral_seg_kuba_band_comparison": "spectral/seg_kuba/band_comparison.png",
    # processing steps (seg_kuba) and the grouping of seg_own
    "step_seg_kuba_preprocessing": "pipeline/seg_kuba/02_preprocessing.png",
    "step_seg_kuba_enhancement": "pipeline/seg_kuba/03_enhancement.png",
    "step_seg_kuba_detection_neg": "pipeline/seg_kuba/04_detection_neg.png",
    "step_seg_kuba_detection_pos": "pipeline/seg_kuba/04_detection_pos.png",
    "step_seg_kuba_grouping": "pipeline/seg_kuba/05_grouping.png",
    "step_seg_own_grouping": "pipeline/seg_own/05_grouping.png",
    # results for the three segments
    **{f"result_{s}_{n}": f"pipeline/{s}/{f}" for s in SEGS
       for n, f in (("raw", "01_raw.png"), ("result", "06_result.png"))},
    "result_timings": "pipeline/timings.png",
    # evaluation and experiments
    "eval_real_summary": "evaluation/real_summary.png",
    **{f"eval_real_{s}": f"evaluation/real_{s}.png" for s in SEGS},
    "eval_synthetic_summary": "evaluation/synthetic_summary.png",
    "eval_synthetic_42": "evaluation/synthetic_42.png",
    "experiments": "experiments/experiments.png",
}


# ======================================================================================
# Block diagram
# ======================================================================================
_C = {"io": "#d9e8fb", "pre": "#e3f4e1", "enh": "#fdf0d5", "det": "#fbe0e0", "post": "#ece3f7",
      "out": "#e8e8e8"}
# (key, text, colour)
NODES = [
    ("load", "Load 12 consecutive 10-s files (2 min) + 1 file of padding on each side\n"
     "raw strain rate: 87 500 x 52 samples (dt 1.6 ms, dx 5.1 m)", "io"),
    ("cm", "Stage 4 - per-channel detrend + common-mode removal\n"
     "(median over the channels at every time sample)", "pre"),
    ("bp", "Stage 4 - zero-phase FFT band-pass 60-312.5 Hz\n(vibration energy of the vehicles)", "pre"),
    ("env", "Stage 4 - RMS envelope of 25-sample blocks (0.04 s rows), crop the padding,\n"
     "log10(envelope / channel median), clip -> energy image 3000 x 52", "pre"),
    ("fk", "Stage 5 - f-k fan filter 5-150 km/h (2-D FFT),\n"
     "separately for v > 0 and v < 0", "enh"),
    ("fb", "Stage 5 - x8 upsampling along the cable + matched line-filter bank\n"
     "(24 angles, 81 px) -> robust z response map per direction", "enh"),
    ("thr", "Stage 6 - threshold: Otsu, at least median + 4 sigma (robust)\n"
     "-> binary mask of the strong line pixels", "det"),
    ("ransac", "Stage 6 - connected components (area >= 1500 px, extent >= 40 m),\n"
     "ridge along time + sequential RANSAC -> line candidates", "det"),
    ("group", "Stage 7 - grouping (same-line / continuation links, connected components),\n"
     "robust weighted mean line per group, duplicates, plausibility (5-150 km/h, >= 8 ch)", "post"),
    ("vel", "Stage 8 - velocity of each object from the slope of its mean line:\n"
     "v = dx / dt [km/h], sign = direction", "post"),
    ("out", "Outputs: waterfall with lines and velocities [km/h], object table, timings;\n"
     "Stage 9: evaluation vs the manual labels (real) and the ground truth (synthetic)", "out"),
]
REMOVED = {"cm": "removed: cable-wide events\n(common mode)",
           "bp": "removed: < 60 Hz (seismic,\nquasi-static load, drift)",
           "env": "removed: oscillation (envelope),\nchannel gain differences",
           "fk": "removed: |v| outside the fan\n(horizontal / vertical stripes,\n"
                 "other direction)",
           "thr": "removed: noise texture\n(below the threshold)",
           "ransac": "removed: specks, small /\nshort components",
           "group": "merged: fragments,\nduplicates"}


def flowchart_figure(path: Path, dpi: int) -> None:
    """Draw the block diagram of the method (boxes, arrows, removed components)."""
    fig, ax = plt.subplots(figsize=(11.5, 13))
    n = len(NODES)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-(n - 1) * 1.05 - 0.6, 0.6)
    ax.axis("off")
    x0, x1, h = 0.3, 8.6, 0.7
    ys = {}
    for row, (key, text, kind) in enumerate(NODES):
        y = -row * 1.05
        ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0.04",
                                    fc=_C[kind], ec="0.3", lw=1))
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=8.4)
        ys[key] = y
    keys = [k for k, _, _ in NODES]
    for a, b in zip(keys[:-1], keys[1:]):
        ax.add_patch(FancyArrowPatch(((x0 + x1) / 2, ys[a] - h / 2), ((x0 + x1) / 2, ys[b] + h / 2),
                                     arrowstyle="-|>", mutation_scale=11, color="0.25", lw=1.0,
                                     shrinkA=1, shrinkB=1))
    for key, text in REMOVED.items():
        ax.text(9.0, ys[key], text, ha="left", va="center", fontsize=7.6, color="#a33",
                style="italic")
        ax.plot([8.65, 8.95], [ys[key], ys[key]], color="#a33", lw=0.8, ls=":")
    ax.set_title("Detection of thick moving-object lines in DAS recordings - block diagram",
                 fontsize=12)
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


def regenerate(skip_slow: bool) -> None:
    """Rerun the result scripts with ``-W error`` so the figures are up to date."""
    for script, extra, slow in RUNS:
        if slow and skip_slow:
            continue
        t = time.perf_counter()
        print(f"  running {script} {' '.join(extra)}", flush=True)
        subprocess.run([sys.executable, "-W", "error", str(SCRIPTS / script), *extra],
                       check=True, cwd=config.ROOT, stdout=subprocess.DEVNULL)
        print(f"    done in {time.perf_counter() - t:.0f} s", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regenerate", action="store_true", help="rerun all analysis scripts first")
    p.add_argument("--skip-experiments", action="store_true",
                   help="with --regenerate: do not rerun run_experiments.py (slowest step)")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--max-width", type=int, default=1300)
    p.add_argument("--jpeg-above-kb", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.regenerate:
        print("Regenerating all results (python -W error):")
        regenerate(args.skip_experiments)
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
