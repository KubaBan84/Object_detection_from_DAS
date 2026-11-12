"""Stage 12: the whole method in one command - raw DAS recording -> objects, velocities,
v(t), features and clusters, with the figures and tables of every stage.

Usage::

    uv run python -W error scripts/run_pipeline.py --segment all       # the 3 segments
    uv run python scripts/run_pipeline.py --segment seg_kuba
    uv run python scripts/run_pipeline.py --start 095012 --n-files 12  # any 2-min window
    uv run python scripts/run_pipeline.py --synthetic 42               # synthetic sample
    uv run python scripts/run_pipeline.py --synthetic all --no-figures

For every recording ``outputs/figures/pipeline/<name>/`` gets:

- ``01_raw.png``           raw waterfall (log10 RMS per 0.04 s, as in the brief);
- ``02_preprocessed.png``  raw vs the Stage-4 images of the HF and LF paths;
- ``03_enhancement.png``   f-k fan filter per direction + directional filter response;
- ``04_candidates.png``    Stage-6 candidates -> Stage-7 groups -> one line per vehicle;
- ``05_result.png``        detected objects with velocities on the raw waterfall + the
  measured tracks coloured by cluster (synthetic: ground truth in lime);
- ``06_velocity.png``      v(t) of every object, coloured by cluster;

and ``outputs/results/pipeline_<name>_objects.csv`` / ``pipeline_<name>.json`` (object
table with velocities, v(t) summary, features, cluster; timings, counts, parameters).
The run writes ``pipeline_summary.csv`` (timings and strong / weak objects per direction
of every recording) and ``outputs/figures/pipeline/timings.png``.

Clustering (Stage 10) is done over all objects of the run (one recording alone has too
few objects); ``--no-cluster`` skips it. ``--cache`` also stores the intermediate images
as ``.npy`` in ``outputs/cache/<name>/`` (``das.pipeline.load_cache``).

Nothing is specific to the 3 segments: every parameter comes from ``config.PipelineParams``
and the axes (dt, dx) from the data / metadata.

Plan: Stage 12.
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
from matplotlib.lines import Line2D  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das.clustering import cluster_objects  # noqa: E402
from das.config import PipelineParams  # noqa: E402
from das.evaluation import evaluate_objects, gt_objects_from_mask  # noqa: E402
from das.pipeline import STAGE_TIMERS, PipelineResult, run_pipeline, save_result  # noqa: E402
from das.velocity import reported_velocity_kmh  # noqa: E402
from das.visualization import (cluster_colour, plot_detections, plot_segments,  # noqa: E402
                               plot_stages, plot_velocity_profiles, plot_waterfall,
                               save_figure)

FIG_DIR = config.FIG_DIR / "pipeline"
SUMMARY_FIELDS = ["name", "kind", "start", "duration_s", "n_channels", "n_objects", "strong_pos",
                  "strong_neg", "weak_pos", "weak_neg", "speed_constant", "speed_varying",
                  "load_s", *STAGE_TIMERS, "total_s", "realtime_factor", "f1", "recall",
                  "precision", "vel_mae"]


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
    p.add_argument("--no-cluster", action="store_true", help="skip the clustering of the objects")
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
               else dio.load_segment(spec, job.info.pop("n_files"), pad_files=1,
                                     name=job.name))
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
def _raw(res: PipelineResult) -> np.ndarray:
    return np.log10(res.pre.common["raw"] + 1e-12)


def _limits(img: np.ndarray) -> dict:
    return {"vmin": float(np.percentile(img, 1)), "vmax": float(np.percentile(img, 99.5))}


def stage_figures(job: Job, dpi: int) -> None:
    """Figures 01-04: raw, preprocessing, enhancement, candidates (need the stage images)."""
    res, d, t0 = job.result, FIG_DIR / job.name, job.t0
    raw = _raw(res)
    fig, ax = plt.subplots(figsize=(6.5, 9), layout="constrained")
    plot_waterfall(raw, t0, res.pre.row_dt, res.pre.col_dx, ax=ax, cbar_label="log10 RMS",
                   title=f"{job.name}: raw waterfall (RMS per {res.pre.row_dt:g} s)", **_limits(raw))
    save_figure(fig, d / "01_raw.png", dpi)

    stages = {"raw (log RMS)": raw}
    stages.update({f"{b.upper()} path (Stage 4)": img for b, img in res.pre.images.items()})
    fig = plot_stages(stages, res.pre.row_dt, res.pre.col_dx, t0, panel_width=3.2,
                      title=f"{job.name}: preprocessing - common mode removed, band-pass, "
                            f"envelope, per-channel normalisation, log")
    save_figure(fig, d / "02_preprocessed.png", dpi)

    stages, dxs, sym = {}, {}, []
    for b, e in res.enhanced.items():
        for name, key in (("f-k, v > 0", "fk_pos"), ("f-k, v < 0", "fk_neg")):
            if key in e.stages:
                stages[f"{b.upper()} {name}"] = e.stages[key]
                dxs[f"{b.upper()} {name}"] = res.pre.col_dx
                sym.append(f"{b.upper()} {name}")
        for dr, name in ((1, "response v > 0"), (-1, "response v < 0")):
            stages[f"{b.upper()} {name}"] = e.response[dr]
            dxs[f"{b.upper()} {name}"] = e.col_dx
    fig = plot_stages(stages, res.row_dt, dxs, t0, symmetric=tuple(sym), panel_width=2.2,
                      cmaps={k: "magma" for k in stages if "response" in k},
                      title=f"{job.name}: Stage 5 - f-k fan filter per direction and "
                            f"directional filter-bank response (robust z)")
    save_figure(fig, d / "03_enhancement.png", dpi)

    bg = np.maximum.reduce([e.combined for e in res.enhanced.values()])
    fig, axes = plt.subplots(1, 3, figsize=(14, 9), sharey=True, layout="constrained")
    kw = dict(colorbar=False, vmin=0, vmax=8, cmap="magma")
    plot_waterfall(bg, t0, res.row_dt, res.col_dx, ax=axes[0], **kw,
                   title=f"Stage 6: {len(res.detections.segments)} candidates\nred strong (6A), "
                         f"cyan / yellow weak (6B); dashed = LF")
    plot_segments(axes[0], res.detections.segments, res.row_dt, res.col_dx, lw=0.8)
    plot_waterfall(bg, t0, res.row_dt, res.col_dx, ax=axes[1], **kw,
                   title=f"Stage 7: {len(res.groups)} groups (one colour each)")
    cmap = plt.get_cmap("tab20")
    segs = [s for g in res.groups for s in g]
    cols = [cmap(k % 20) for k, g in enumerate(res.groups) for _ in g]
    plot_segments(axes[1], segs, res.row_dt, res.col_dx, colours=cols, lw=1.0)
    c = res.counts
    plot_detections(bg, res.objects, res.row_dt, res.col_dx, t0, ax=axes[2], vmin=0, vmax=8,
                    cmap="gray", label_fontsize=6.5,
                    title=f"Stage 7: {c['n_objects']} objects\nstrong (solid) "
                          f"{c['strong_pos'] + c['strong_neg']}, weak (dashed) "
                          f"{c['weak_pos'] + c['weak_neg']}")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle(f"{job.name}: candidates -> groups -> one line per vehicle")
    save_figure(fig, d / "04_candidates.png", dpi)


def _cluster_colours(res: PipelineResult, names: list[str]) -> list | None:
    if not res.clusters:
        return None
    return [cluster_colour(names.index(c)) for c in res.clusters]


def _cluster_legend(ax, res: PipelineResult, names: list[str]) -> None:
    present = [n for n in names if n in res.clusters]
    if present:
        ax.legend([Line2D([], [], color=cluster_colour(names.index(n)), lw=2) for n in present],
                  [f"{n} ({res.clusters.count(n)})" for n in present], fontsize=7,
                  loc="lower left", framealpha=0.8)


def result_figures(job: Job, names: list[str], dpi: int) -> None:
    """Figures 05-06: the result with velocities (+ tracks by cluster) and v(t)."""
    res, d, t0 = job.result, FIG_DIR / job.name, job.t0
    c = res.counts
    raw = _raw(res)
    colours = _cluster_colours(res, names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True, layout="constrained")
    plot_detections(raw, res.objects, res.pre.row_dt, res.pre.col_dx, t0, ax=axes[0],
                    cmap="gray", label_fontsize=7.5, **_limits(raw),
                    title=f"detected objects and velocities [km/h]\n"
                          f"solid = strong / thick, dashed = weak / thin")
    bg = np.maximum.reduce([e.combined for e in res.enhanced.values()])
    if colours:      # measured tracks (or the line if not tracked), one colour per cluster
        plot_waterfall(bg, t0, res.row_dt, res.col_dx, ax=axes[1], colorbar=False, vmin=0,
                       vmax=8, cmap="gray",
                       title="measured tracks (Stage 9) coloured by cluster (Stage 10)")
        for o, col in zip(res.objects, colours):
            xt = (o.track if o.track is not None and len(o.track) >= 2 else
                  np.array([[o.x_start_m, o.t_at(o.x_start_m)], [o.x_end_m, o.t_at(o.x_end_m)]]))
            axes[1].plot(xt[:, 0], xt[:, 1], color=col, lw=1.6,
                         ls="-" if o.kind == "strong" else "--")
        _cluster_legend(axes[1], res, names)
    else:
        plot_detections(bg, res.objects, res.row_dt, res.col_dx, t0, ax=axes[1], vmin=0,
                        vmax=8, cmap="gray", tracks=True, lines=False, label=False,
                        title="measured tracks (Stage 9: blue constant, orange varying speed)")
    if job.refs:
        for r in job.refs:
            axes[1].plot(r["track"][:, 0], r["track"][:, 1], color="lime", lw=0.6, alpha=0.7)
        axes[1].set_title(axes[1].get_title() + "; lime = ground truth", fontsize=10)
    axes[1].set_ylabel("")
    fig.suptitle(f"{job.name}: {c['n_objects']} objects - strong v>0 {c['strong_pos']}, "
                 f"v<0 {c['strong_neg']}; weak v>0 {c['weak_pos']}, v<0 {c['weak_neg']}; "
                 f"{c['speed_varying']} with varying speed")
    save_figure(fig, d / "05_result.png", dpi)

    ax = plot_velocity_profiles(res.objects, t0, colours=colours,
                                title=f"{job.name}: velocity over time v(t) of every object"
                                      + (" (colour = cluster)" if colours else ""))
    if colours:
        _cluster_legend(ax, res, names)
    save_figure(ax.figure, d / "06_velocity.png", dpi)


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
    ax.set_title("pipeline time per recording (2-min segment target: <= 20-30 s)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, FIG_DIR / "timings.png", dpi)


# ======================================================================================
# Main
# ======================================================================================
def print_objects(job: Job) -> None:
    """Print the object table of one recording."""
    res = job.result
    print(f"    {'id':>3} {'kind':6} {'dir':>3} {'v [km/h]':>8} {'class':8} {'t_ref':>9} "
          f"{'x [m]':>11}  cluster")
    for k, o in enumerate(res.objects):
        t_ref = (job.t0.timestamp() if job.t0 else 0.0) + o.t_ref_s
        clock = (datetime.fromtimestamp(t_ref).strftime("%H:%M:%S") if job.t0
                 else f"{o.t_ref_s:7.1f} s")
        cl = res.clusters[k] if k < len(res.clusters) else ""
        print(f"    {o.obj_id:>3} {o.kind:6} {o.direction:>3} {reported_velocity_kmh(o):8.1f} "
              f"{o.speed_class or '-':8} {clock:>9} {o.x_start_m:5.0f}-{o.x_end_m:<5.0f}  {cl}")


def main() -> None:
    args = parse_args()
    pp = PipelineParams()
    jobs = make_jobs(args)
    print(f"[1/3] pipeline (Stages 4-10) on {len(jobs)} recording(s)")
    for job, spec in jobs:
        load_and_run(job, spec, pp)
        tm = job.result.timings
        print(f"  {job.name}: {job.result.counts['n_objects']} objects in {tm['total_s']:.1f} s "
              f"(+ load {tm['load_s']:.1f} s) - "
              + ", ".join(f"{k.removesuffix('_s')} {tm[k]:.1f}" for k in STAGE_TIMERS if k in tm))
        if not args.no_figures:
            stage_figures(job, args.dpi)

    names: list[str] = []
    feats = [f for job, _ in jobs for f in job.result.features]
    if not args.no_cluster:
        cl = cluster_objects(feats, pp.clu)
        names = list(cl["names"])
        k = 0
        for job, _ in jobs:
            n = len(job.result.objects)
            job.result.clusters = cl["object_names"][k:k + n]
            k += n
        print(f"[2/3] clustering of {len(feats)} objects ({pp.clu.mode}): k = {cl['k']}, "
              f"silhouette {cl['silhouette']:.2f}")
        for c, nm in enumerate(names):
            print(f"    {c} {nm or '(not clustered)'}: {int(np.sum(cl['labels'] == c))}")
    else:
        print("[2/3] clustering skipped")

    print("[3/3] figures and tables")
    rows = []
    for job, _ in jobs:
        res = job.result
        extra = {"kind": job.kind, **job.info}
        if job.refs:
            ev = evaluate_objects(res.objects, job.refs, pp.ev, res.pre.images["hf"].shape,
                                  res.row_dt, res.channel_dx)
            extra["evaluation"] = {k: v for k, v in ev.items()
                                   if k not in ("matches", "pairs", "fp_kinds")}
        save_result(res, job.name, job.t0, pp,
                    cache_dir=config.CACHE_DIR if args.cache else None, extra=extra)
        if not args.no_figures:
            result_figures(job, names, args.dpi)
        tm = res.timings
        ev = extra.get("evaluation", {})
        rows.append({"name": job.name, "kind": job.kind, **job.info, **res.counts,
                     **{k: tm.get(k, float("nan")) for k in ("load_s", *STAGE_TIMERS, "total_s")},
                     "realtime_factor": job.info["duration_s"] / tm["total_s"],
                     **{k: ev.get(k, "") for k in ("f1", "recall", "precision", "vel_mae")}})
        c = res.counts
        print(f"  {job.name}: strong v>0 {c['strong_pos']}, v<0 {c['strong_neg']} | weak v>0 "
              f"{c['weak_pos']}, v<0 {c['weak_neg']} | varying speed {c['speed_varying']}"
              + (f" | vs GT: F1 {ev['f1']:.3f}, |v| MAE {ev['vel_mae']:.2f} km/h" if ev else ""))
        print_objects(job)

    with open(config.RESULTS_DIR / "pipeline_summary.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows])
    if not args.no_figures:
        timing_figure(rows, args.dpi)
    print("\n  stage timings [s]:")
    print(f"    {'recording':18} {'length':>6} " + " ".join(f"{k.removesuffix('_s'):>10}"
                                                       for k in ("load_s", *STAGE_TIMERS, "total_s")))
    for r in rows:
        print(f"    {r['name']:18} {r['duration_s']:6.0f} "
              + " ".join(f"{r[k]:10.2f}" for k in ("load_s", *STAGE_TIMERS, "total_s")))
    if not args.no_figures:
        print(f"\nFigures: {FIG_DIR}")
    print(f"Results: {config.RESULTS_DIR}/pipeline_*")


if __name__ == "__main__":
    main()
