"""Stage 10: the whole method in one command - raw DAS recording -> thick lines with
velocities, with the figures and tables of every processing step.

Usage::

    uv run python -W error scripts/run_pipeline.py --segment all       # the 3 segments
    uv run python scripts/run_pipeline.py --segment seg_kuba
    uv run python scripts/run_pipeline.py --start 092002 --n-files 12  # any 2-min window
    uv run python scripts/run_pipeline.py --synthetic 42               # synthetic sample
    uv run python scripts/run_pipeline.py --synthetic all --no-figures

For every recording ``outputs/figures/pipeline/<name>/`` gets:

- ``01_raw.png``            raw waterfall (log10 RMS per 0.04 s, as in the brief);
- ``02_preprocessing.png``  raw -> common mode removed -> band-pass -> envelope ->
  log-ratio image, plus what was removed (common mode, out-of-band part);
- ``03_enhancement.png``    f-k fan filter (both directions, what it removed, per
  direction) and the directional filter-bank response per direction;
- ``04_detection_<pos|neg>.png``  threshold -> components ->
  ridge -> RANSAC lines on the response map of one direction;
- ``05_grouping.png``       candidates -> groups -> one averaged line per vehicle;
- ``06_result.png``         detected objects with velocities on the raw waterfall
  (synthetic: ground truth in lime);

and ``outputs/results/pipeline_<name>_objects.csv`` / ``pipeline_<name>.json``. The run
also writes ``pipeline_summary.csv`` (objects per direction and timings of every
recording) and ``outputs/figures/pipeline/timings.png``. ``--cache`` stores the
intermediate images as ``.npy`` in ``outputs/cache/<name>/`` (``das.pipeline.load_cache``).

Nothing is specific to the 3 segments: every parameter comes from ``config.PipelineParams``
and the axes (dt, dx) from the data / metadata.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das.config import PipelineParams  # noqa: E402
from das.evaluation import evaluate_objects, gt_objects_from_mask  # noqa: E402
from das.pipeline import STAGE_TIMERS, PipelineResult, run_pipeline, save_result  # noqa: E402
from das.visualization import (plot_detections, plot_segments, plot_stages,  # noqa: E402
                               plot_waterfall, save_figure)

FIG_DIR = config.FIG_DIR / "pipeline"
SUMMARY_FIELDS = ["name", "kind", "start", "duration_s", "n_channels", "n_objects", "n_pos",
                  "n_neg", "load_s", *STAGE_TIMERS, "total_s", "realtime_factor",
                  "recall_thick", "precision", "f1", "vel_mae_thick"]


@dataclass
class Job:
    """One recording to process."""
    name: str
    kind: str                                  # "segment", "window" or "synthetic"
    t0: datetime | None = None
    result: PipelineResult | None = None
    refs: list = field(default_factory=list)   # synthetic ground truth
    info: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", default=None,
                   help="segment name from config.SEGMENTS or 'all' (default if nothing else is given)")
    p.add_argument("--start", default=None, help="HHMMSS of an ad-hoc segment of --n-files files")
    p.add_argument("--n-files", type=int, default=config.FILES_PER_SEGMENT,
                   help="number of 10-s files of an ad-hoc segment (12 = 2 min)")
    p.add_argument("--synthetic", default=None, help="synthetic sample id(s), comma-separated or 'all'")
    p.add_argument("--no-figures", action="store_true", help="tables only")
    p.add_argument("--cache", action="store_true", help="store the stage images as .npy")
    p.add_argument("--dpi", type=int, default=100)
    args = p.parse_args()
    if args.segment is None and args.start is None and args.synthetic is None:
        args.segment = "all"
    return args


# ======================================================================================
# Running
# ======================================================================================
def load_and_run(job: Job, spec, pp: PipelineParams) -> None:
    """Load the recording of a job and run the pipeline on it."""
    t = time.perf_counter()
    if job.kind == "synthetic":
        sig, gt, meta = dio.load_synthetic(spec)
        dt, dx = dio.synthetic_axes(meta)
        load_s = time.perf_counter() - t
        job.result = run_pipeline(sig, pp, 1.0 / dt, dx)
        job.refs = gt_objects_from_mask(gt, meta, dt, dx)
        n_samples, n_ch = sig.shape
    else:
        seg = (dio.load_named_segment(spec, pad_files=1) if job.kind == "segment"
               else dio.load_segment(spec, job.info.pop("n_files"), pad_files=1, name=job.name))
        load_s = time.perf_counter() - t
        job.t0, dt = seg.start, seg.dt
        job.result = run_pipeline(seg.data, pp, 1.0 / seg.dt, seg.dx,
                                  crop=(seg.pad_before, seg.pad_after))
        n_samples, n_ch = seg.n_samples, seg.n_channels
    job.result.timings["load_s"] = load_s
    job.info.update({"duration_s": n_samples * dt, "n_channels": n_ch,
                     "start": job.t0.strftime("%H:%M:%S") if job.t0 else ""})


def make_jobs(args) -> list[tuple[Job, object]]:
    """Recordings selected on the command line (segments, windows, synthetic samples)."""
    jobs = []
    if args.segment:
        names = [s["name"] for s in config.SEGMENTS] if args.segment == "all" else [args.segment]
        jobs += [(Job(n, "segment"), n) for n in names]
    if args.start:
        start = dio._stem(args.start)
        jobs.append((Job(f"start_{start}", "window", info={"n_files": args.n_files}), start))
    if args.synthetic:
        ids = (dio.list_synthetic_ids() if args.synthetic == "all"
               else [s.strip() for s in args.synthetic.split(",")])
        jobs += [(Job(f"synthetic_{sid}", "synthetic"), sid) for sid in ids]
    return jobs


# ======================================================================================
# Figures
# ======================================================================================
def _log(img: np.ndarray) -> np.ndarray:
    return np.log10(img + 1e-12)


def _limits(img: np.ndarray) -> dict:
    return {"vmin": float(np.percentile(img, 1)), "vmax": float(np.percentile(img, 99.5))}


def figure_raw(job: Job, dpi: int) -> None:
    """01: raw waterfall (log10 RMS)."""
    res = job.result
    raw = _log(res.pre.stages["raw"])
    fig, ax = plt.subplots(figsize=(6.5, 9), layout="constrained")
    plot_waterfall(raw, job.t0, res.pre.row_dt, res.pre.col_dx, ax=ax, cbar_label="log10 RMS",
                   title=f"{job.name}: raw waterfall (RMS per {res.pre.row_dt:g} s)", **_limits(raw))
    save_figure(fig, FIG_DIR / job.name / "01_raw.png", dpi)


def figure_preprocessing(job: Job, pp, dpi: int) -> None:
    """02: every Stage-4 step and the removed components (log10 RMS except the image)."""
    res, st = job.result, job.result.pre.stages
    band = f"{pp.band_low:g}-{pp.band_high:g} Hz"
    stages = {"1 raw": _log(st["raw"]),
              "removed:\ncommon mode": _log(st["common_mode"]),
              "2 common mode\nremoved": _log(st["after_common_mode"]),
              f"removed:\noutside {band}": _log(st["removed_by_bandpass"]),
              f"3 band-pass {band}\n(RMS envelope)": _log(st["envelope"]),
              "4 log10(env / median)\n= Stage-4 image": st["image"]}
    fig = plot_stages(stages, res.pre.row_dt, res.pre.col_dx, job.t0, panel_width=2.6,
                      title=f"{job.name}: Stage 4 - preprocessing (panels 1-3: log10 RMS of the "
                            f"signal; 'removed' = what the step took away)")
    save_figure(fig, FIG_DIR / job.name / "02_preprocessing.png", dpi)


def figure_enhancement(job: Job, dpi: int) -> None:
    """03: f-k fan filter (what it keeps and removes) and the filter-bank response."""
    res, e = job.result, job.result.enhanced
    stages = {"input (Stage 4)": e.stages["input"], "f-k, both directions": e.stages["fk_both"],
              "removed by the f-k fan": e.stages["fk_removed"], "f-k, v > 0": e.stages["fk_pos"],
              "f-k, v < 0": e.stages["fk_neg"], "response v > 0": e.response[1],
              "response v < 0": e.response[-1]}
    dxs = {k: (res.col_dx if k.startswith("response") else res.pre.col_dx) for k in stages}
    sym = tuple(k for k in stages if k.startswith(("f-k", "removed")))
    fig = plot_stages(stages, res.row_dt, dxs, job.t0, symmetric=sym, panel_width=2.3,
                      cmaps={k: "magma" for k in stages if k.startswith("response")},
                      title=f"{job.name}: Stage 5 - f-k fan filter (5-150 km/h, per direction) "
                            f"and directional filter-bank response (robust z)")
    save_figure(fig, FIG_DIR / job.name / "03_enhancement.png", dpi)


def figure_detection(job: Job, direction: int, dpi: int) -> None:
    """04: every step of the thick-line detection on the response map of one direction."""
    res = job.result
    dn = "pos" if direction > 0 else "neg"
    st, thr = res.detections.stages, res.detections.thresholds[dn]
    panels = [("response (Stage 5)", res.enhanced.response[direction], "magma", (0, 8)),
              (f"1 threshold z > {thr:.1f}", st[f"{dn}_binary"], "gray", (0, 1)),
              ("2 kept components", st[f"{dn}_mask"], "gray", (0, 1)),
              ("3 ridge + RANSAC lines", st[f"{dn}_ridge"] & st[f"{dn}_mask"], "gray", (0, 1))]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 8.5), sharey=True,
                             layout="constrained")
    for ax, (name, img, cmap, (lo, hi)) in zip(axes, panels):
        plot_waterfall(np.asarray(img, dtype=np.float32), job.t0, res.row_dt, res.col_dx, ax=ax,
                       colorbar=False, cmap=cmap, vmin=lo, vmax=hi, title=name)
        ax.set_xlabel("x [m]")
        if ax is not axes[0]:
            ax.set_ylabel("")
    segs = [s for s in res.detections.segments if s.direction == direction]
    plot_segments(axes[-1], segs, res.row_dt, res.col_dx, lw=1.2)
    sign = ">" if direction > 0 else "<"
    fig.suptitle(f"{job.name}: Stage 6 - detection of thick lines, v {sign} 0 "
                 f"({len(segs)} line candidates)")
    save_figure(fig, FIG_DIR / job.name / f"04_detection_{dn}.png", dpi)


def figure_grouping(job: Job, dpi: int) -> None:
    """05: candidates -> groups -> one averaged line per vehicle."""
    res = job.result
    bg = res.enhanced.combined
    fig, axes = plt.subplots(1, 3, figsize=(14, 9), sharey=True, layout="constrained")
    kw = dict(colorbar=False, vmin=0, vmax=8, cmap="magma")
    plot_waterfall(bg, job.t0, res.row_dt, res.col_dx, ax=axes[0], **kw,
                   title=f"Stage 6: {len(res.detections.segments)} line candidates")
    plot_segments(axes[0], res.detections.segments, res.row_dt, res.col_dx, lw=0.9)
    plot_waterfall(bg, job.t0, res.row_dt, res.col_dx, ax=axes[1], **kw,
                   title=f"Stage 7: {len(res.groups)} groups (one colour each)")
    cmap = plt.get_cmap("tab20")
    segs = [s for g in res.groups for s in g]
    cols = [cmap(k % 20) for k, g in enumerate(res.groups) for _ in g]
    plot_segments(axes[1], segs, res.row_dt, res.col_dx, colours=cols, lw=1.2)
    plot_detections(bg, res.objects, res.row_dt, res.col_dx, job.t0, ax=axes[2], vmin=0,
                    vmax=8, cmap="gray", label_fontsize=6.5,
                    title=f"Stage 7: {len(res.objects)} objects (averaged line per group)")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle(f"{job.name}: candidates -> groups -> one line per vehicle")
    save_figure(fig, FIG_DIR / job.name / "05_grouping.png", dpi)


def figure_result(job: Job, dpi: int) -> None:
    """06: the result as in the brief - red lines with velocities on the raw waterfall."""
    res = job.result
    raw = _log(res.pre.stages["raw"])
    c = res.counts
    fig, ax = plt.subplots(figsize=(7.5, 10.5), layout="constrained")
    plot_detections(raw, res.objects, res.pre.row_dt, res.pre.col_dx, job.t0, ax=ax,
                    cmap="viridis", label_fontsize=8, **_limits(raw),
                    title=f"{job.name}: {c['n_objects']} detected thick lines "
                          f"(v > 0: {c['n_pos']}, v < 0: {c['n_neg']}), velocities [km/h]")
    for r in job.refs:
        ax.plot(r["track"][:, 0], r["track"][:, 1], color="lime", lw=0.6, alpha=0.7)
    if job.refs:
        ax.set_title(ax.get_title() + "\nlime = ground truth (all vehicles)", fontsize=10)
    save_figure(fig, FIG_DIR / job.name / "06_result.png", dpi)


def timing_figure(rows: list[dict], dpi: int) -> None:
    """Stacked bars of the processing time per stage and recording."""
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(rows) + 3), 4.5), layout="constrained")
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for key in ("load_s", *STAGE_TIMERS):
        v = np.array([r.get(key, 0.0) or 0.0 for r in rows])
        ax.bar(x, v, bottom=bottom, label=key.removesuffix("_s"))
        bottom += v
    ax.set_xticks(x, [r["name"] for r in rows], rotation=20, fontsize=8)
    ax.set_ylabel("time [s]")
    ax.set_title("pipeline time per recording")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, FIG_DIR / "timings.png", dpi)


# ======================================================================================
# Main
# ======================================================================================
def print_objects(job: Job) -> None:
    """Print the object table of one recording."""
    print(f"    {'id':>3} {'dir':>3} {'v [km/h]':>8} {'t_ref':>11} {'x [m]':>11} {'score':>6}")
    for o in job.result.objects:
        clock = (datetime.fromtimestamp(job.t0.timestamp() + o.t_ref_s).strftime("%H:%M:%S")
                 if job.t0 else f"{o.t_ref_s:7.1f} s")
        print(f"    {o.obj_id:>3} {o.direction:>3} {o.velocity_kmh:8.1f} {clock:>11} "
              f"{o.x_start_m:5.0f}-{o.x_end_m:<5.0f} {o.score:6.1f}")


def main() -> None:
    args = parse_args()
    pp = PipelineParams()
    jobs = make_jobs(args)
    print(f"Pipeline (Stages 4-8) on {len(jobs)} recording(s)")
    rows = []
    for job, spec in jobs:
        load_and_run(job, spec, pp)
        res, tm = job.result, job.result.timings
        extra = {"kind": job.kind, **job.info}
        ev = {}
        if job.refs:
            ev = evaluate_objects(res.objects, job.refs, pp.ev, res.n_channels, res.channel_dx)
            extra["evaluation"] = {k: v for k, v in ev.items()
                                   if k not in ("matches", "pairs", "fp_kinds")}
        save_result(res, job.name, job.t0, pp,
                    cache_dir=config.CACHE_DIR if args.cache else None, extra=extra)
        if not args.no_figures:
            figure_raw(job, args.dpi)
            figure_preprocessing(job, pp.pre, args.dpi)
            figure_enhancement(job, args.dpi)
            for d in (1, -1):
                figure_detection(job, d, args.dpi)
            figure_grouping(job, args.dpi)
            figure_result(job, args.dpi)
        c = res.counts
        rows.append({"name": job.name, "kind": job.kind, **job.info, **c,
                     **{k: tm.get(k, float("nan")) for k in ("load_s", *STAGE_TIMERS, "total_s")},
                     "realtime_factor": job.info["duration_s"] / tm["total_s"],
                     **{k: ev.get(k, "") for k in ("recall_thick", "precision", "f1",
                                                   "vel_mae_thick")}})
        print(f"  {job.name}: {c['n_objects']} objects (v > 0: {c['n_pos']}, v < 0: {c['n_neg']}) "
              f"in {tm['total_s']:.1f} s (+ load {tm['load_s']:.1f} s)"
              + (f" | vs GT: thick recall {ev['tp_thick']}/{ev['n_ref_thick']}, precision "
                 f"{ev['precision']:.3f}, thick |v| MAE {ev['vel_mae_thick']:.2f} km/h" if ev else ""))
        print_objects(job)
        job.result = None                       # free the intermediate images

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "pipeline_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows])
    if not args.no_figures:
        timing_figure(rows, args.dpi)
    keys = ("load_s", *STAGE_TIMERS, "total_s")
    print("\n  stage timings [s]:")
    print(f"    {'recording':18} {'length':>6} " + " ".join(f"{k.removesuffix('_s'):>10}" for k in keys))
    for r in rows:
        print(f"    {r['name']:18} {r['duration_s']:6.0f} " + " ".join(f"{r[k]:10.2f}" for k in keys))
    if not args.no_figures:
        print(f"\nFigures: {FIG_DIR}")
    print(f"Results: {config.RESULTS_DIR}/pipeline_*")


if __name__ == "__main__":
    main()
