"""Exploratory data analysis of the real DAS recordings.

1. Global streaming pass over all 259 files: per-channel moments, histograms, per-file and
   per-second RMS, neighbour-channel correlation, bad-channel detection, overview image.
2. Per-segment analysis for the segments in ``config.SEGMENTS``: descriptive statistics,
   variability in time and space, distributions, spatial coherence, figures.
3. Ranking of candidate 2-minute segments for the own (third) segment ``seg_own``.

Outputs: ``outputs/results/eda_*.json`` / ``eda_candidates.csv`` and figures in
``outputs/figures/eda/``.

Plan: Stage 2.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das import stats  # noqa: E402
from das.visualization import plot_overview_panels, plot_waterfall, save_figure  # noqa: E402

FIG_DIR = config.FIG_DIR / "eda"
WIN_1S = int(round(1.0 / config.DT))        # 625 samples = 1 s
ENV_BLOCK = 50                              # 0.08 s rows in the overview image
DISPLAY_BLOCK = 25                          # 0.04 s rows in segment waterfalls
SEG_COLOURS = {"assigned": "red", "own": "orange"}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", default="all", help="segment name from config.SEGMENTS or 'all'")
    p.add_argument("--top", type=int, default=8, help="number of candidate segments to list")
    p.add_argument("--dpi", type=int, default=110)
    return p.parse_args()


def to_jsonable(obj):
    """Recursively convert numpy / datetime / Path objects for json.dump."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.name
    return obj


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=1), encoding="utf-8")


def segment_interval(spec: dict) -> tuple[datetime, datetime]:
    start = dio.parse_timestamp(spec["start"])
    return start, start + timedelta(seconds=spec["n_files"] * config.FILE_DURATION_S)


# ======================================================================================
# 1. Global streaming pass
# ======================================================================================
def global_pass(files: list[Path]) -> dict:
    """One pass over all files; nothing but small summaries is kept in memory."""
    first = dio.load_real_file(files[0])
    n_ch = first.shape[1]
    lim = 40.0 * float(first.std())                         # data-driven histogram range
    rs = stats.RunningStats(n_ch, bins=np.linspace(-lim, lim, 801),
                            abs_bins=np.logspace(-11, -3, 161))
    nf = len(files)
    rows_env = config.SAMPLES_PER_FILE // ENV_BLOCK
    out = {
        "file_rms": np.empty((nf, n_ch)),
        "file_max_abs": np.empty(nf),
        "rms_1s": np.empty((nf, config.SAMPLES_PER_FILE // WIN_1S, n_ch)),
        "env": np.empty((nf, rows_env, n_ch), dtype=np.float32),
        "neigh_corr": np.empty((nf, n_ch - 1)),
    }
    for i, f in enumerate(files):
        x = first if i == 0 else dio.load_real_file(f)
        rs.update(x)
        out["file_rms"][i] = np.sqrt(np.mean(np.square(x, dtype=np.float64), axis=0))
        out["file_max_abs"][i] = float(np.abs(x).max())
        out["rms_1s"][i] = stats.block_rms(x, WIN_1S)
        out["env"][i] = stats.block_rms(x, ENV_BLOCK)
        out["neigh_corr"][i] = stats.neighbour_correlation(x)
        if (i + 1) % 50 == 0 or i + 1 == nf:
            print(f"  global pass: {i + 1}/{nf} files")
    out["running"] = rs.result()
    out["files"] = files
    out["times"] = [dio.parse_timestamp(f) for f in files]
    # Per-channel reference level of the 1 s RMS (median over the whole dataset).
    out["ref_1s"] = np.median(out["rms_1s"].reshape(-1, n_ch), axis=0)
    out["env_ref"] = np.median(out["env"].reshape(-1, n_ch), axis=0)
    return out


def global_figures(g: dict, bad: np.ndarray, z: np.ndarray, runs: list[list[Path]],
                   dpi: int) -> None:
    """Whole-dataset figures: activity timeline, channel profile, per-file RMS, histograms."""
    res = g["running"]
    glob = res["global"]
    n_ch = g["file_rms"].shape[1]
    x_m = np.arange(n_ch) * config.DX

    # G1 - value distributions -------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    edges, counts = res["hist"]["edges"], res["hist"]["counts"]
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    density = counts / (counts.sum() * width)
    a1.semilogy(centres, density, lw=1.2, label="data")
    for sigma, lab in ((glob["std"], "Gaussian, σ = std"),
                       (stats.MAD_TO_SIGMA * glob["abs_p50"], "Gaussian, σ = 1.4826·median|x|")):
        a1.semilogy(centres, np.exp(-0.5 * (centres / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
                    "--", lw=1, label=lab)
    a1.set_ylim(density[density > 0].min(), density.max() * 3)
    a1.set_xlabel("strain rate")
    a1.set_ylabel("probability density")
    a1.set_title(f"All values (skewness {glob['skewness']:.2f}, "
                 f"excess kurtosis {glob['kurtosis']:.1f})")
    a1.legend(fontsize=8)
    ae, ac = res["abs_hist"]["edges"], res["abs_hist"]["counts"]
    a2.loglog(np.sqrt(ae[1:] * ae[:-1]), ac / np.diff(ae) / ac.sum(), lw=1.2)
    for q in (50, 99, 99.9):
        a2.axvline(glob[f"abs_p{q:g}"], ls=":", color="gray")
        a2.text(glob[f"abs_p{q:g}"], 0.02, f" p{q:g}", fontsize=8, rotation=90,
                va="bottom", transform=a2.get_xaxis_transform())
    a2.set_xlabel("|strain rate| (log bins)")
    a2.set_ylabel("density")
    a2.set_title("Distribution of |x|")
    save_figure(fig, FIG_DIR / "global_histograms.png", dpi)

    # G2 - profile along the cable ----------------------------------------------------
    pc = res["per_channel"]
    corr = g["neigh_corr"].mean(axis=0)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    axes[0].semilogy(x_m, pc["rms"], "o-", ms=3, label="RMS (all files)")
    axes[0].semilogy(x_m, np.median(g["file_rms"], axis=0), "s-", ms=3,
                     label="median of per-file RMS")
    if bad.any():
        axes[0].semilogy(x_m[bad], pc["rms"][bad], "rx", ms=10, mew=2, label="bad channel")
    axes[0].set_ylabel("RMS")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Channel statistics along the fibre (all 259 files)")
    axes[1].plot(x_m, z, "o-", ms=3)
    axes[1].axhspan(-3.5, 3.5, color="green", alpha=0.1, label="|z| < 3.5 (normal)")
    axes[1].set_ylabel("robust z of log RMS")
    axes[1].legend(fontsize=8)
    axes[2].semilogy(x_m, np.maximum(pc["kurtosis"], 1e-2), "o-", ms=3)
    axes[2].set_ylabel("excess kurtosis")
    axes[3].plot(x_m[:-1] + config.DX / 2, corr, "o-", ms=3)
    axes[3].set_ylabel("corr(ch, ch+1)")
    axes[3].set_xlabel("distance along fibre [m]")
    save_figure(fig, FIG_DIR / "global_channel_profile.png", dpi)

    # G3 - per-file RMS heatmap (file x channel) --------------------------------------
    fig, ax = plt.subplots(figsize=(9, 11), layout="constrained")
    img = np.log10(g["file_rms"] / np.median(g["file_rms"], axis=0) + EPS)
    im = ax.imshow(img, aspect="auto", cmap="magma", vmin=-0.3, vmax=float(np.percentile(img, 99.5)),
                   extent=(0, n_ch * config.DX, len(g["files"]), 0), interpolation="nearest")
    step = 12
    ax.set_yticks(np.arange(0, len(g["files"]), step) + 0.5)
    ax.set_yticklabels([g["times"][i].strftime("%H:%M:%S") for i in range(0, len(g["files"]), step)],
                       fontsize=7)
    ax.axhline(len(runs[0]), color="cyan", lw=1.5, ls="--")
    ax.text(n_ch * config.DX, len(runs[0]), " gap", color="cyan", va="center", fontsize=8)
    for spec in config.SEGMENTS:
        i0 = [f.stem for f in g["files"]].index(spec["start"])
        ax.add_patch(plt.Rectangle((0, i0), n_ch * config.DX, spec["n_files"], fill=False,
                                   edgecolor=SEG_COLOURS[spec["kind"]], lw=2))
        ax.text(1, i0 + 1, spec["name"], color=SEG_COLOURS[spec["kind"]], fontsize=8, va="top")
    ax.set_xlabel("distance along fibre [m]")
    ax.set_ylabel("file (start time)")
    ax.set_title("log10(per-file RMS / channel median) - variability over 40 min")
    fig.colorbar(im, ax=ax, fraction=0.04)
    save_figure(fig, FIG_DIR / "global_file_channel_rms.png", dpi)

    # G4 - activity timeline ------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, layout="constrained")
    t_mid = [t + timedelta(seconds=config.FILE_DURATION_S / 2) for t in g["times"]]
    med_rms = np.median(g["file_rms"], axis=1)
    t_1s = [t + timedelta(seconds=s + 0.5) for t in g["times"] for s in range(g["rms_1s"].shape[1])]
    ratio_1s = (g["rms_1s"] / g["ref_1s"]).reshape(-1, n_ch)
    for run in runs:  # plot runs separately so that the gap is not bridged
        idx = [g["files"].index(f) for f in run]
        a1.plot([t_mid[i] for i in idx], med_rms[idx], "o-", ms=2, color="C0")
        a2.plot([t_1s[j] for i in idx for j in range(i * 10, i * 10 + 10)],
                np.mean(ratio_1s > 3.0, axis=1)[[j for i in idx for j in range(i * 10, i * 10 + 10)]],
                lw=0.6, color="C1")
    for ax in (a1, a2):
        for spec in config.SEGMENTS:
            s, e = segment_interval(spec)
            ax.axvspan(s, e, color=SEG_COLOURS[spec["kind"]], alpha=0.15)
            ax.text(s, 0.97, spec["name"], transform=ax.get_xaxis_transform(), fontsize=8,
                    va="top", color=SEG_COLOURS[spec["kind"]])
    a1.set_ylabel("median channel RMS\nper 10 s file")
    a2.set_ylabel("fraction of channels\nwith 1 s RMS > 3× ref")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a2.set_xlabel("time (2024-05-07)")
    a1.set_title("Traffic activity over the whole recording (shaded: analysed segments)")
    save_figure(fig, FIG_DIR / "global_activity_timeline.png", dpi)

    # G5 - overview of the whole recording (2-minute panels) --------------------------
    env_norm = np.log10(g["env"] / g["env_ref"] + EPS)
    panels = []
    seg_iv = [(spec, *segment_interval(spec)) for spec in config.SEGMENTS]
    for run in runs:
        idx = [g["files"].index(f) for f in run]
        for c in range(0, len(idx), config.FILES_PER_SEGMENT):
            chunk = idx[c:c + config.FILES_PER_SEGMENT]
            t0 = g["times"][chunk[0]]
            t1 = t0 + timedelta(seconds=len(chunk) * config.FILE_DURATION_S)
            highlights = []
            for spec, s, e in seg_iv:
                lo, hi = max(s, t0), min(e, t1)
                if lo < hi:
                    highlights.append(((lo - t0).total_seconds(), (hi - t0).total_seconds(),
                                       spec["name"], SEG_COLOURS[spec["kind"]]))
            panels.append({"img": env_norm[chunk].reshape(-1, n_ch), "t0": t0,
                           "title": f"{t0:%H:%M:%S} ({len(chunk)} f)", "highlights": highlights})
    fig = plot_overview_panels(panels, ENV_BLOCK * config.DT, config.DX, ncols=12,
                               vmin=-0.2, vmax=float(np.percentile(env_norm, 99.7)),
                               cbar_label="log10(RMS / channel median), 0.08 s blocks",
                               suptitle="Whole recording in 2-minute panels (time downwards)")
    save_figure(fig, FIG_DIR / "global_overview.png", dpi=90)


# ======================================================================================
# 2. Per-segment analysis
# ======================================================================================
def analyse_segment(spec: dict, g: dict, dpi: int) -> dict:
    """Statistics and figures of one 2-minute segment (histograms, box plots, correlation)."""
    seg = dio.load_named_segment(spec["name"])
    x = seg.core
    n_ch = x.shape[1]
    x_m = seg.distance_axis()
    out_dir = FIG_DIR / spec["name"]
    title = f"{spec['name']} ({spec.get('owner', '')}) {seg.start:%H:%M:%S}-{seg.end:%H:%M:%S}"

    desc = stats.describe(x)
    pch = stats.per_channel_stats(x)
    pts = stats.per_time_stats(x, WIN_1S)
    ncorr = stats.neighbour_correlation(x)
    cmat = stats.channel_correlation_matrix(x)
    bad, z = stats.detect_bad_channels(pch["rms"])
    act = stats.activity_metrics(pts["rms_map"], g["ref_1s"])

    # S1 - waterfall (RMS envelope), linear and log ------------------------------------
    env = stats.block_rms(x, DISPLAY_BLOCK)
    row_dt = DISPLAY_BLOCK * seg.dt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 9), layout="constrained")
    plot_waterfall(env, seg.start, row_dt, seg.dx, ax=a1, vmin=0,
                   vmax=float(np.percentile(env, 99.5)),
                   title="RMS in 0.04 s blocks (linear, clip p99.5)", cbar_label="RMS strain rate")
    lg = np.log10(env / np.median(env, axis=0) + EPS)
    plot_waterfall(lg, seg.start, row_dt, seg.dx, ax=a2, vmin=0,
                   vmax=float(np.percentile(lg, 99.5)),
                   title="log10(RMS / channel median)", cbar_label="log10 ratio")
    fig.suptitle(title)
    save_figure(fig, out_dir / "waterfall.png", dpi)

    # S2 - raw signal zoom: the busiest 10 s -------------------------------------------
    per_s = pts["rms_median"]
    k = int(np.argmax(np.convolve(per_s, np.ones(10), mode="valid")))
    raw = x[k * WIN_1S:(k + 10) * WIN_1S]
    lim = float(np.percentile(np.abs(raw), 99.5))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 7), layout="constrained",
                                 gridspec_kw={"width_ratios": [1, 1.4]})
    plot_waterfall(raw, seg.start, seg.dt, seg.dx, ax=a1, cmap="seismic", vmin=-lim, vmax=lim,
                   t_offset_s=k, title="raw signed strain rate (busiest 10 s)",
                   cbar_label="strain rate")
    t = k + np.arange(raw.shape[0]) * seg.dt
    chans = np.linspace(0, n_ch - 1, 5).astype(int)
    offset = 4 * float(np.std(raw))
    for j, c in enumerate(chans):
        a2.plot(t, raw[:, c] + j * offset, lw=0.4)
        a2.text(t[-1], j * offset, f" {x_m[c]:.0f} m", fontsize=8, va="center")
    a2.set_xlabel("time from segment start [s]")
    a2.set_yticks([])
    a2.set_title("raw traces of 5 channels (vertically offset)")
    fig.suptitle(title)
    save_figure(fig, out_dir / "raw_zoom.png", dpi)

    # S3 - variability in time and space -----------------------------------------------
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1], sharey=ax_map)
    ax_x = fig.add_subplot(gs[1, 0], sharex=ax_map)
    rmap = np.log10(pts["rms_map"] + EPS)
    plot_waterfall(rmap, seg.start, 1.0, seg.dx, ax=ax_map, cmap="magma",
                   vmin=float(np.percentile(rmap, 1)), vmax=float(np.percentile(rmap, 99.5)),
                   title="log10 RMS in 1 s windows", cbar_label="log10 RMS")
    ts = np.arange(len(per_s)) + 0.5
    ax_t.semilogx(pts["rms_mean"], ts, label="mean over channels")
    ax_t.semilogx(pts["rms_median"], ts, label="median over channels")
    ax_t.set_xlabel("RMS (1 s)")
    ax_t.legend(fontsize=7)
    ax_t.tick_params(labelleft=False)
    centres = x_m + seg.dx / 2
    ax_x.semilogy(centres, pch["rms"], "o-", ms=3, label="RMS")
    ax_x.semilogy(centres, stats.MAD_TO_SIGMA * pch["mad"], "s-", ms=3, label="1.4826·MAD")
    ax_x.semilogy(centres, pch["max_abs"], "^-", ms=3, label="max |x|")
    if bad.any():
        ax_x.semilogy(centres[bad], pch["rms"][bad], "rx", ms=10, mew=2, label="bad")
    ax_x.set_xlabel("distance along fibre [m]")
    ax_x.legend(fontsize=7, ncols=4)
    fig.suptitle(title + " - variability in time and space")
    save_figure(fig, out_dir / "variability.png", dpi)

    # S4 - per-channel boxplots ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4.5), layout="constrained")
    ax.boxplot(x[::5], showfliers=False, whis=(0.5, 99.5), widths=0.6)
    ax.set_xticks(np.arange(1, n_ch + 1, 4), [f"{x_m[i]:.0f}" for i in range(0, n_ch, 4)])
    ax.set_xlabel("distance along fibre [m]")
    ax.set_ylabel("strain rate")
    ax.set_title(f"{title} - per-channel distribution (box: IQR, whiskers: p0.5-p99.5)")
    save_figure(fig, out_dir / "channel_boxplots.png", dpi)

    # S5 - histogram vs global -----------------------------------------------------------
    edges = g["running"]["hist"]["edges"]
    gc = g["running"]["hist"]["counts"]
    centres_h = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]
    sc = np.histogram(x, edges)[0]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.semilogy(centres_h, gc / (gc.sum() * w), label="all 259 files")
    ax.semilogy(centres_h, sc / (sc.sum() * w), label=spec["name"])
    sig = desc["robust_sigma"]
    ax.semilogy(centres_h, np.exp(-0.5 * (centres_h / sig) ** 2) / (sig * np.sqrt(2 * np.pi)),
                "--", label="Gaussian, σ = 1.4826·MAD")
    dens = gc / (gc.sum() * w)
    ax.set_ylim(dens[dens > 0].min(), dens.max() * 3)
    ax.set_xlabel("strain rate")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title(f"{spec['name']}: kurtosis {desc['kurtosis']:.1f}, skewness {desc['skewness']:.2f}")
    save_figure(fig, out_dir / "histogram.png", dpi)

    # S6 - spatial coherence -------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    ext = (0, n_ch * seg.dx, n_ch * seg.dx, 0)
    im = a1.imshow(cmat, cmap="RdBu_r", vmin=-1, vmax=1, extent=ext)
    a1.set_xlabel("distance [m]")
    a1.set_ylabel("distance [m]")
    a1.set_title("channel correlation matrix (raw signal)")
    fig.colorbar(im, ax=a1, fraction=0.046)
    a2.plot(x_m[:-1] + seg.dx / 2, ncorr, "o-", ms=3, label=spec["name"])
    a2.plot(x_m[:-1] + seg.dx / 2, g["neigh_corr"].mean(axis=0), "--", label="all files (mean)")
    a2.set_xlabel("distance along fibre [m]")
    a2.set_ylabel("corr(ch, ch+1)")
    a2.legend(fontsize=8)
    a2.set_title("neighbour-channel correlation")
    save_figure(fig, out_dir / "correlation.png", dpi)

    result = {
        "name": spec["name"], "owner": spec.get("owner"), "kind": spec["kind"],
        "start": seg.start, "end": seg.end, "files": seg.files, "shape": x.shape,
        "describe": desc, "activity": act,
        "per_channel": {k: v for k, v in pch.items()},
        "per_second": {"rms_mean": pts["rms_mean"], "rms_median": pts["rms_median"],
                       "max_abs": pts["max_abs"]},
        "neighbour_correlation": ncorr,
        "bad_channels": np.flatnonzero(bad), "bad_channel_z": z,
        "busiest_10s_offset_s": k,
    }
    save_json(result, config.RESULTS_DIR / f"eda_{spec['name']}.json")
    return result


# ======================================================================================
# 3. Candidates for the own segment
# ======================================================================================
def rank_candidates(g: dict, runs: list[list[Path]]) -> list[dict]:
    """Score every 12-file window that does not overlap the assigned segments."""
    n_ch = g["rms_1s"].shape[2]
    stems = [f.stem for f in g["files"]]
    taken = [segment_interval(s) for s in config.SEGMENTS]
    rows = []
    for run in runs:
        for i in range(len(run) - config.FILES_PER_SEGMENT + 1):
            idx = [stems.index(f.stem) for f in run[i:i + config.FILES_PER_SEGMENT]]
            start = g["times"][idx[0]]
            end = start + timedelta(seconds=config.FILES_PER_SEGMENT * config.FILE_DURATION_S)
            if any(start < e and s < end for s, e in taken):
                continue
            m = stats.activity_metrics(g["rms_1s"][idx].reshape(-1, n_ch), g["ref_1s"])
            rows.append({"start": run[i].stem, "end_file": run[i + config.FILES_PER_SEGMENT - 1].stem,
                         "has_margin": 0 < i < len(run) - config.FILES_PER_SEGMENT, **m})
    rows.sort(key=lambda r: r["active_fraction"], reverse=True)
    return rows


def non_overlapping(rows: list[dict], n: int) -> list[dict]:
    """Pick the first ``n`` rows whose 2-minute windows do not overlap."""
    chosen: list[dict] = []
    for r in rows:
        s = dio.parse_timestamp(r["start"])
        e = s + timedelta(seconds=config.FILES_PER_SEGMENT * config.FILE_DURATION_S)
        if all(not (s < ce and cs < e) for cs, ce in
               ((dio.parse_timestamp(c["start"]),
                 dio.parse_timestamp(c["start"]) + timedelta(seconds=120)) for c in chosen)):
            chosen.append(r)
        if len(chosen) == n:
            break
    return chosen


# ======================================================================================
def main() -> None:
    args = parse_args()
    files = dio.list_real_files()
    runs = dio.find_continuous_runs(files)
    print(f"[1/3] global pass over {len(files)} files ...")
    g = global_pass(files)
    res = g["running"]
    bad, z = stats.detect_bad_channels(np.median(g["file_rms"], axis=0))
    global_figures(g, bad, z, runs, args.dpi)
    save_json({
        "n_files": len(files),
        "runs": [{"start": r[0].stem, "end": r[-1].stem, "n_files": len(r)} for r in runs],
        "global": res["global"],
        "per_channel": res["per_channel"],
        "bad_channels": np.flatnonzero(bad), "bad_channel_z": z,
        "neighbour_correlation_mean": g["neigh_corr"].mean(axis=0),
        "ref_rms_1s": g["ref_1s"],
        "per_file": [{"file": f.stem, "median_rms": float(np.median(r)), "max_abs": float(m)}
                     for f, r, m in zip(files, g["file_rms"], g["file_max_abs"])],
        "hist": res["hist"], "abs_hist": res["abs_hist"],
    }, config.RESULTS_DIR / "eda_global.json")
    gl = res["global"]
    print(f"  global: mean={gl['mean']:.3e} std={gl['std']:.3e} max|x|={gl['max_abs']:.3e} "
          f"skew={gl['skewness']:.2f} kurt={gl['kurtosis']:.1f}")
    print(f"  |x| percentiles: p50={gl['abs_p50']:.2e} p99={gl['abs_p99']:.2e} "
          f"p99.9={gl['abs_p99.9']:.2e}")
    print(f"  bad channels (|z|>3.5): {np.flatnonzero(bad).tolist()}  "
          f"max |z| = {np.abs(z).max():.2f} (channel {int(np.argmax(np.abs(z)))})")

    specs = config.SEGMENTS if args.segment == "all" else [dio.get_segment_spec(args.segment)]
    print(f"[2/3] segment analysis: {[s['name'] for s in specs]}")
    for spec in specs:
        r = analyse_segment(spec, g, args.dpi)
        d, a = r["describe"], r["activity"]
        print(f"  {spec['name']:<12} std={d['std']:.3e} robust_sigma={d['robust_sigma']:.3e} "
              f"kurt={d['kurtosis']:.1f} active={a['active_fraction']:.3f} "
              f"strong={a['strong_fraction']:.4f} events={a['n_events']} "
              f"bad={r['bad_channels'].tolist()}")

    print("[3/3] candidates for seg_own (no overlap with assigned segments)")
    rows = rank_candidates(g, runs)
    path = config.RESULTS_DIR / "eda_candidates.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows)} candidate windows -> {path.name}; top non-overlapping:")
    for r in non_overlapping(rows, args.top):
        print(f"  {r['start']}-{r['end_file']} margin={str(r['has_margin']):<5} "
              f"active={r['active_fraction']:.3f} strong={r['strong_fraction']:.4f} "
              f"busy={r['busy_fraction']:.3f} events={r['n_events']}")
    print(f"\nFigures: {FIG_DIR}\nResults: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
