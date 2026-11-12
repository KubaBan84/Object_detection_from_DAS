"""Frequency analysis of the real DAS recordings and the choice of the pass bands.

1. Global Welch PSD over all 259 files (per channel), detection of tonal lines
   (on the channel median and on every channel separately).
2. Band selection: for a grid of bands [f_low, f_high) the signal of every analysed
   segment is common-mode filtered, band-passed and converted to the log energy envelope
   image; the fraction of its f-k energy inside the vehicle velocity fan (10-150 km/h,
   both directions) is the label-free criterion. The best low-frequency (LF, quasi-static
   load -> thin lines) and high-frequency (HF, vibrations -> thick lines) bands are chosen.
   The vehicle-vs-background spectral contrast (and the rejected d' criterion) is plotted.
3. Per segment: per-channel FFT, STFT of selected channels, band-limited envelopes,
   f-k spectra of the raw signal and of the LF / HF envelopes with the velocity fan.

Outputs: ``outputs/results/spectral_summary.json`` and figures in ``outputs/figures/spectral/``.

Plan: Stage 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das import spectral, stats  # noqa: E402
from das.enhancement import fk_velocity_filter, line_snr  # noqa: E402
from das.preprocessing import bandpass_fft, bandpass_response, remove_common_mode  # noqa: E402
from das.visualization import plot_waterfall, save_figure  # noqa: E402

FIG_DIR = config.FIG_DIR / "spectral"
FS = config.FS
NYQ = FS / 2
ENV_BLOCK = 25                     # 0.04 s rows of the envelope image
ROW_DT = ENV_BLOCK * config.DT
CELL_WIN = 1250                    # 2 s cells for the vehicle/background contrast
PLAN_BANDS = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, NYQ)]
COMPARE_BANDS = [(0.5, 5), (5, 20), (20, 50), (50, 120), (120, 300)]
D_LOWS = np.array([0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60])
D_HIGHS = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 312])
FAN_LOWS = np.array([0.2, 0.3, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80])
FAN_HIGHS = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, NYQ])
LF_MAX_HIGH = 10.0                 # LF bands: f_high <= 10 Hz
HF_MIN_LOW = 10.0                  # HF bands: f_low >= 10 Hz
V_FAN_KMH = (10.0, 150.0)
V_LINES_KMH = (20, 50, 100)
EPS = 1e-30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", default="all", help="segment name from config.SEGMENTS or 'all'")
    p.add_argument("--channels", type=int, nargs="*", default=None,
                   help="channels for the STFT (default: 3 spread along the fibre incl. ~128 m)")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--recompute", action="store_true", help="ignore the cached band grid")
    return p.parse_args()


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def band_label(band: tuple[float, float]) -> str:
    return f"{band[0]:g}-{band[1]:g} Hz"


def taper_for(f_low: float) -> float:
    """Transition width: 1 Hz, but at most half of f_low (narrow low-frequency bands)."""
    return min(1.0, 0.5 * f_low) if f_low > 0 else 1.0


def log_norm_envelope(x: np.ndarray) -> np.ndarray:
    """log10(block RMS / channel median) - the image used throughout the project."""
    env = stats.block_rms(x, ENV_BLOCK)
    return np.log10(env / np.median(env, axis=0) + 1e-12)


def band_envelope(seg: dio.Segment, xcm: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Common-mode-free signal -> band-pass -> log envelope image cropped to the core."""
    xb = bandpass_fft(xcm, FS, band[0], band[1], taper_hz=taper_for(band[0]))
    return seg.crop(log_norm_envelope(xb), ENV_BLOCK)


# ======================================================================================
# 1. Global PSD and tonal lines
# ======================================================================================
def global_psd(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Mean Welch PSD per channel over many files."""
    acc = None
    for i, f in enumerate(files):
        freqs, psd = spectral.welch_psd(dio.load_real_file(f), FS, nperseg=2048)
        acc = psd if acc is None else acc + psd
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f"  global PSD: {i + 1}/{len(files)} files")
    return freqs, acc / len(files)


def channel_tonal_lines(freqs: np.ndarray, psd: np.ndarray, prominence_db: float = 10.0
                        ) -> list[dict]:
    """Tonal lines found on individual channels, grouped by frequency (0.5 Hz bins)."""
    found: dict[float, list[int]] = {}
    for c in range(psd.shape[1]):
        for p in spectral.find_tonal_peaks(freqs, psd[:, c], prominence_db):
            found.setdefault(round(p["freq_hz"] * 2) / 2, []).append(c)
    return [{"freq_hz": f, "n_channels": len(ch), "channels": ch}
            for f, ch in sorted(found.items())]


def plot_global_psd(freqs, psd, peaks, ch_lines, dpi) -> None:
    """Channel x frequency PSD map and the mean spectrum with the found peaks."""
    n_ch = psd.shape[1]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 9), layout="constrained",
                                 gridspec_kw={"height_ratios": [1.2, 1]})
    img = 10 * np.log10(psd.T + EPS)
    im = a1.imshow(img, aspect="auto", origin="lower", cmap="magma",
                   extent=(freqs[0], freqs[-1], 0, n_ch * config.DX),
                   vmin=float(np.percentile(img, 2)), vmax=float(np.percentile(img, 99.8)))
    a1.set_xscale("symlog", linthresh=1.0)
    a1.set_xlim(0, NYQ)
    a1.set_xlabel("frequency [Hz]")
    a1.set_ylabel("distance along fibre [m]")
    a1.set_title("Welch PSD of every channel, mean over all 259 files")
    fig.colorbar(im, ax=a1, fraction=0.03, label="dB re 1 (strain rate)²/Hz")
    a2.loglog(freqs[1:], np.median(psd, axis=1)[1:], label="median over channels")
    a2.loglog(freqs[1:], np.percentile(psd, 90, axis=1)[1:], lw=0.8, label="90th percentile")
    a2.loglog(freqs[1:], np.percentile(psd, 10, axis=1)[1:], lw=0.8, label="10th percentile")
    for p in peaks:
        a2.axvline(p["freq_hz"], color="red", ls=":", lw=0.8)
    for line in ch_lines:
        a2.axvline(line["freq_hz"], color="gray", ls=":", lw=0.6)
    a2.set_xlabel("frequency [Hz]")
    a2.set_ylabel("PSD")
    a2.legend(fontsize=8)
    a2.set_title(f"PSD across channels (tonal lines > 10 dB: {len(peaks)} on the median, "
                 f"grey: {len(ch_lines)} on single channels)")
    save_figure(fig, FIG_DIR / "global_psd.png", dpi)


# ======================================================================================
# 2. Band selection
# ======================================================================================
def fan_grid(specs: list[dict]) -> np.ndarray:
    """Band metrics for every band and segment -> (S, nL, nH, 4).

    Metrics: [fan energy fraction v>0, fan energy fraction v<0, line SNR (p99 of the robust
    z-score of the fan-filtered envelope image), fraction of fan-filtered pixels > 4 sigma].
    """
    v_min, v_max = (v / 3.6 for v in V_FAN_KMH)
    grid = np.full((len(specs), len(FAN_LOWS), len(FAN_HIGHS), 4), np.nan)
    for s, spec in enumerate(specs):
        seg = dio.load_named_segment(spec["name"], pad_files=1)
        xcm = remove_common_mode(seg.data).astype(np.float64)
        spec_x = np.fft.rfft(xcm, axis=0)
        fr = np.fft.rfftfreq(xcm.shape[0], 1.0 / FS)
        for i, lo in enumerate(FAN_LOWS):
            for j, hi in enumerate(FAN_HIGHS):
                if hi <= 1.5 * lo:
                    continue
                h = bandpass_response(fr, lo, hi, taper_hz=taper_for(lo))
                xb = np.fft.irfft(spec_x * h[:, None], n=xcm.shape[0], axis=0)
                img = seg.crop(log_norm_envelope(xb), ENV_BLOCK)
                grid[s, i, j, :2] = spectral.fk_fan_fraction(img, ROW_DT, seg.dx, v_min, v_max)
                filt = fk_velocity_filter(img, ROW_DT, seg.dx, v_min, v_max, direction=0)
                grid[s, i, j, 2:] = line_snr(filt)
        print(f"  fan-energy grid: {spec['name']} done")
    return grid


def choose_bands(score: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Best LF band (f_high <= LF_MAX_HIGH) and HF band (f_low >= HF_MIN_LOW) by ``score``."""
    lf = np.where(FAN_HIGHS[None, :] <= LF_MAX_HIGH, score, np.nan)
    hf = np.where(FAN_LOWS[:, None] >= HF_MIN_LOW, score, np.nan)
    i, j = np.unravel_index(np.nanargmax(lf), lf.shape)
    a, b = np.unravel_index(np.nanargmax(hf), hf.shape)
    return (float(FAN_LOWS[i]), float(FAN_HIGHS[j])), (float(FAN_LOWS[a]), float(FAN_HIGHS[b]))


def plot_band_grids(snr: np.ndarray, fan_total: np.ndarray, lf, hf, dpi) -> None:
    """Line SNR and fan energy fraction over the grid of candidate bands."""
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.5), layout="constrained")
    for ax, grid, title, label in (
            (axes[0], snr, "PRIMARY: line SNR after the f-k fan filter\n"
                           "p99 of robust z-score of the fan-filtered log envelope", "line SNR"),
            (axes[1], fan_total, "secondary: fraction of envelope f-k energy inside the fan\n"
                                 "(energy-weighted, penalised by short horizontal streaks)",
             "fan energy fraction")):
        im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(FAN_HIGHS)), [f"{v:g}" for v in FAN_HIGHS], rotation=60)
        ax.set_yticks(range(len(FAN_LOWS)), [f"{v:g}" for v in FAN_LOWS])
        for band, name, colour in ((lf, "LF", "red"), (hf, "HF", "white")):
            i, j = list(FAN_LOWS).index(band[0]), list(FAN_HIGHS).index(band[1])
            ax.plot(j, i, "*", color=colour, ms=16, mec="black")
            ax.text(j, i + 0.45, f"{name} {band_label(band)}", color=colour, ha="center",
                    fontsize=9, fontweight="bold")
        ax.axvline(list(FAN_HIGHS).index(LF_MAX_HIGH) + 0.5, color="red", ls=":", lw=1)
        ax.axhline(list(FAN_LOWS).index(HF_MIN_LOW) - 0.5, color="white", ls=":", lw=1)
        ax.set_xlabel("f_high [Hz]")
        ax.set_ylabel("f_low [Hz]")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, label=label)
    fig.suptitle(f"Band selection, mean of the 3 segments (common-mode removed, fan "
                 f"{V_FAN_KMH[0]:g}-{V_FAN_KMH[1]:g} km/h); stars = chosen LF / HF bands")
    save_figure(fig, FIG_DIR / "band_selection.png", dpi)


def contrast_analysis(specs: list[dict], lf, hf, dpi: int) -> dict:
    """Vehicle vs background cell spectra; d' grid (biased - documented, not used)."""
    powers, vehs, bgs = [], [], []
    freqs = None
    for spec in specs:
        x = dio.load_named_segment(spec["name"]).core
        freqs, power, rms = spectral.cell_spectra(x, FS, CELL_WIN)
        veh, bg = spectral.select_cells(rms, 0.9, 0.3)
        powers.append(power)
        vehs.append(veh)
        bgs.append(bg)
    power, veh, bg = np.concatenate(powers), np.concatenate(vehs), np.concatenate(bgs)
    mean_v, mean_b = power[veh].mean(axis=0), power[bg].mean(axis=0)
    ratio_db = 10 * np.log10((mean_v + EPS) / (mean_b + EPS))
    sm = np.convolve(ratio_db, np.ones(5) / 5, mode="same")
    d = spectral.band_separability(freqs, power, veh, bg, D_LOWS, D_HIGHS)
    i, j = np.unravel_index(np.nanargmax(d), d.shape)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    axes[0].loglog(freqs[1:], mean_v[1:], label=f"vehicle cells (top 10% RMS, n={veh.sum()})")
    axes[0].loglog(freqs[1:], mean_b[1:], label=f"background cells (bottom 30%, n={bg.sum()})")
    axes[0].set_xlabel("frequency [Hz]")
    axes[0].set_ylabel("mean PSD of 2 s cells")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Spectra of vehicle vs background cells (3 segments)")
    axes[1].semilogx(freqs[1:], ratio_db[1:], lw=0.5, alpha=0.5)
    axes[1].semilogx(freqs[1:], sm[1:], lw=1.5)
    axes[1].axvspan(*lf, color="red", alpha=0.15, label=f"LF {band_label(lf)}")
    axes[1].axvspan(*hf, color="green", alpha=0.15, label=f"HF {band_label(hf)}")
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("vehicle / background [dB]")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Spectral contrast of traffic")
    im = axes[2].imshow(d, origin="lower", cmap="viridis", aspect="auto")
    axes[2].set_xticks(range(len(D_HIGHS)), [f"{v:g}" for v in D_HIGHS], rotation=60, fontsize=8)
    axes[2].set_yticks(range(len(D_LOWS)), [f"{v:g}" for v in D_LOWS], fontsize=8)
    axes[2].plot(j, i, "r*", ms=14)
    axes[2].set_xlabel("f_high [Hz]")
    axes[2].set_ylabel("f_low [Hz]")
    axes[2].set_title("d' of log band power - REJECTED criterion\n"
                      "(cells chosen by broadband RMS -> favours the widest band)")
    fig.colorbar(im, ax=axes[2], fraction=0.05, label="d'")
    save_figure(fig, FIG_DIR / "band_contrast.png", dpi)
    return {
        "n_vehicle_cells": int(veh.sum()), "n_background_cells": int(bg.sum()),
        "ratio_db_at": {f"{f:g} Hz": float(np.interp(f, freqs, sm)) for f in
                        (1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300)},
        "ratio_db_peak_hz": float(freqs[np.argmax(sm)]),
        "dprime_rejected": {"best_band_hz": (float(D_LOWS[i]), float(D_HIGHS[j])),
                            "best": float(d[i, j]), "f_lows": D_LOWS, "f_highs": D_HIGHS, "d": d},
    }


# ======================================================================================
# 3. Per-segment analysis
# ======================================================================================
def plot_fk(ax, f, k, s, fmax, name):
    """f-k amplitude map [dB] of one image on ``ax``; returns the image artist."""
    keep = (f >= 0) & (f <= fmax)
    sdb = 20 * np.log10(s[keep] + EPS)
    im = ax.imshow(sdb, aspect="auto", origin="lower", cmap="magma",
                   extent=(k[0], k[-1], 0, f[keep][-1]),
                   vmin=float(np.percentile(sdb, 30)), vmax=float(np.percentile(sdb, 99.95)))
    kk = np.linspace(k[0], k[-1], 200)
    for v_kmh in V_LINES_KMH:
        for sign, style in ((1, "w--"), (-1, "c--")):
            ff = -sign * v_kmh / 3.6 * kk        # numpy convention: f = -v k
            m = (ff >= 0) & (ff <= f[keep][-1])
            ax.plot(kk[m], ff[m], style, lw=0.8)
            ax.text(kk[m][-1 if sign < 0 else 0], ff[m][-1 if sign < 0 else 0],
                    f"{sign * v_kmh:+d}", color=style[0], fontsize=7)
    ax.set_xlabel("wavenumber k [cycles/m]")
    ax.set_ylabel("frequency f [Hz]")
    ax.set_title(f"{name}\nwhite: v > 0 (towards +x), cyan: v < 0 [km/h]", fontsize=10)
    return im


def analyse_segment(spec: dict, lf, hf, channels: list[int], dpi: int) -> dict:
    """FFT per channel, STFT and f-k analysis of one segment."""
    seg = dio.load_named_segment(spec["name"], pad_files=1)
    x = seg.core
    n_ch = x.shape[1]
    out_dir = FIG_DIR / spec["name"]
    title = f"{spec['name']} {seg.start:%H:%M:%S}-{seg.end:%H:%M:%S}"
    xcm = remove_common_mode(seg.data)

    # FFT per channel ------------------------------------------------------------------
    freqs, amp = spectral.channel_fft(x, FS)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    sel = freqs <= 150
    step = max(1, int(sel.sum() // 3000))
    img = 20 * np.log10(amp[sel][::step].T + EPS)
    im = a1.imshow(img, aspect="auto", origin="lower", cmap="magma",
                   extent=(0, freqs[sel][-1], 0, n_ch * seg.dx),
                   vmin=float(np.percentile(img, 5)), vmax=float(np.percentile(img, 99.9)))
    a1.set_xlabel("frequency [Hz] (0-150 Hz)")
    a1.set_ylabel("distance along fibre [m]")
    a1.set_title("FFT amplitude of each channel (independent rfft), dB")
    fig.colorbar(im, ax=a1, fraction=0.04)
    med = np.median(amp, axis=1)
    a2.loglog(freqs[1:], med[1:], lw=0.3, alpha=0.5, label="median over channels")
    a2.loglog(freqs[1:], np.convolve(med, np.ones(25) / 25, mode="same")[1:], lw=1.5,
              label="smoothed (25 bins)")
    a2.set_xlabel("frequency [Hz]")
    a2.set_ylabel("amplitude")
    a2.legend(fontsize=8)
    a2.set_title("whole 2-min segment, 0-312 Hz")
    fig.suptitle(title)
    save_figure(fig, out_dir / "fft.png", dpi)

    # STFT of selected channels ----------------------------------------------------------
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3.2 * len(channels)), sharex=True,
                             layout="constrained", squeeze=False)
    for ax, c in zip(axes[:, 0], channels):
        f_s, t_s, z = spectral.stft(x[:, c], FS, nperseg=625, overlap=0.75)
        keep = f_s <= 150
        zdb = 20 * np.log10(z[keep] + EPS)
        im = ax.imshow(zdb, aspect="auto", origin="lower", cmap="viridis",
                       extent=(t_s[0], t_s[-1], f_s[0], f_s[keep][-1]),
                       vmin=float(np.percentile(zdb, 5)), vmax=float(np.percentile(zdb, 99.7)))
        ax.set_ylabel("f [Hz]")
        ax.set_title(f"STFT, channel {c} (x = {c * seg.dx:.0f} m), 1 s Hann window, 75% overlap")
        fig.colorbar(im, ax=ax, fraction=0.03, label="dB")
    axes[-1, 0].set_xlabel(f"time from {seg.start:%H:%M:%S} [s]")
    save_figure(fig, out_dir / "stft.png", dpi)

    # Band-limited envelopes --------------------------------------------------------------
    bands = [(0.0, NYQ)] + COMPARE_BANDS + [lf, hf]
    names = ["broadband"] + [band_label(b) for b in COMPARE_BANDS] + \
            [f"LF {band_label(lf)}", f"HF {band_label(hf)}"]
    fig, axes = plt.subplots(1, len(bands), figsize=(2.9 * len(bands), 8), layout="constrained",
                             sharey=True)
    for ax, band, name in zip(axes, bands, names):
        img = band_envelope(seg, xcm, band)
        plot_waterfall(img, seg.start, ROW_DT, seg.dx, ax=ax, vmin=0,
                       vmax=float(np.percentile(img, 99.5)), colorbar=False, title=name)
        ax.set_xlabel("x [m]")
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.suptitle(f"{title}: log10(RMS / channel median), 0.04 s blocks, after common-mode removal")
    save_figure(fig, out_dir / "band_comparison.png", dpi)

    # f-k spectra ------------------------------------------------------------------------
    v_edges = np.array([-1e3, -150 / 3.6, -10 / 3.6, 0, 10 / 3.6, 150 / 3.6, 1e3])
    labels = ["v<-150", "-150..-10", "-10..0", "0..10", "10..150", "v>150"]
    f_r, k_r, s_r = spectral.fk_spectrum(x, seg.dt, seg.dx, pad_x=128)
    fk_frac = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    im = plot_fk(axes[0], f_r, k_r, s_r, 8.0, "raw strain rate (0-8 Hz)")
    fig.colorbar(im, ax=axes[0], fraction=0.04, label="dB")
    for ax, band, name in ((axes[1], lf, "LF"), (axes[2], hf, "HF")):
        f_e, k_e, s_e = spectral.fk_spectrum(band_envelope(seg, xcm, band), ROW_DT, seg.dx,
                                             pad_x=128)
        frac, rest = spectral.fk_velocity_energy(f_e, k_e, s_e, v_edges)
        fk_frac[name] = dict(zip(labels, frac)) | {"|v|>1000 m/s or k=0": rest}
        im = plot_fk(ax, f_e, k_e, s_e, 2.5, f"log envelope, {name} {band_label(band)} (0-2.5 Hz)")
        fig.colorbar(im, ax=ax, fraction=0.04, label="dB")
    fig.suptitle(title)
    save_figure(fig, out_dir / "fk.png", dpi)

    f_w, psd_w = spectral.welch_psd(x, FS, nperseg=4096)
    bp = spectral.band_power(f_w, psd_w, PLAN_BANDS).sum(axis=1)
    return {
        "name": spec["name"],
        "tonal_peaks_median": spectral.find_tonal_peaks(f_w, np.median(psd_w, axis=1)),
        "band_power_fraction": dict(zip([band_label(b) for b in PLAN_BANDS], bp / bp.sum())),
        "fk_energy_fraction_envelope": fk_frac,
    }


# ======================================================================================
def main() -> None:
    args = parse_args()
    specs = config.SEGMENTS if args.segment == "all" else [dio.get_segment_spec(args.segment)]
    files = dio.list_real_files()

    print(f"[1/4] global Welch PSD over {len(files)} files ...")
    freqs, psd = global_psd(files)
    peaks = spectral.find_tonal_peaks(freqs, np.median(psd, axis=1))
    ch_lines = channel_tonal_lines(freqs, psd)
    plot_global_psd(freqs, psd, peaks, ch_lines, args.dpi)
    bp = spectral.band_power(freqs, psd, PLAN_BANDS)
    frac_global = bp.sum(axis=1) / bp.sum()
    print("  tonal lines on channel median:", [round(p["freq_hz"], 2) for p in peaks] or "none")
    print("  tonal lines on single channels:",
          [(ln["freq_hz"], ln["n_channels"]) for ln in ch_lines] or "none")
    print("  power fraction per band:",
          ", ".join(f"{band_label(b)}: {f:.3f}" for b, f in zip(PLAN_BANDS, frac_global)))

    print("[2/4] band selection: f-k fan-energy criterion on all configured segments")
    cache = config.CACHE_DIR / "spectral_fan_grid.npz"
    key = np.array([s["start"] for s in config.SEGMENTS])
    if cache.is_file() and not args.recompute:
        c = np.load(cache)
        ok = (np.array_equal(c["key"], key) and np.array_equal(c["lows"], FAN_LOWS)
              and np.array_equal(c["highs"], FAN_HIGHS))
        fan = c["fan"] if ok else None
    else:
        fan = None
    if fan is None:
        fan = fan_grid(config.SEGMENTS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, fan=fan, key=key, lows=FAN_LOWS, highs=FAN_HIGHS)
    else:
        print(f"  using cached grid {cache.name} (--recompute to refresh)")
    mean = fan.mean(axis=0)                    # NaN (invalid band) stays NaN
    fan_total = mean[..., 0] + mean[..., 1]
    snr, frac4 = mean[..., 2], mean[..., 3]
    lf, hf = choose_bands(snr)
    plot_band_grids(snr, fan_total, lf, hf, args.dpi)

    def at(grid, b):
        return float(grid[list(FAN_LOWS).index(b[0]), list(FAN_HIGHS).index(b[1])])
    bb = (0.2, NYQ)
    for name, b in (("LF", lf), ("HF", hf), ("broadband", bb)):
        print(f"  {name:<9} {band_label(b):<14} line SNR {at(snr, b):.2f}  "
              f">4 sigma {at(frac4, b):.4f}  fan fraction {at(fan_total, b):.3f}")
    order = np.argsort(-np.nan_to_num(snr, nan=-1).ravel())[:10]
    top = [(float(FAN_LOWS[a]), float(FAN_HIGHS[b]), float(snr[a, b]))
           for a, b in zip(*np.unravel_index(order, snr.shape))]
    print("  top-10 bands by line SNR:", ", ".join(f"{a:g}-{b:g} ({v:.2f})" for a, b, v in top))

    print("[3/4] vehicle vs background spectral contrast")
    contrast = contrast_analysis(config.SEGMENTS, lf, hf, args.dpi)
    print("  contrast [dB]:", ", ".join(f"{k}: {v:.1f}" for k, v in contrast["ratio_db_at"].items()))

    n_ch = psd.shape[1]
    channels = args.channels or [int(round(n_ch * 0.15)), 25, int(round(n_ch * 0.8))]
    print(f"[4/4] per-segment analysis {[s['name'] for s in specs]}, STFT channels {channels}")
    segs = []
    for spec in specs:
        r = analyse_segment(spec, lf, hf, channels, args.dpi)
        segs.append(r)
        for name, frac in r["fk_energy_fraction_envelope"].items():
            print(f"  {spec['name']:<12} {name} envelope f-k energy: " +
                  ", ".join(f"{k}: {v:.3f}" for k, v in frac.items()))

    summary = {
        "global": {"tonal_peaks_median": peaks, "tonal_lines_channels": ch_lines,
                   "band_power_fraction": dict(zip([band_label(b) for b in PLAN_BANDS], frac_global))},
        "band_selection": {
            "criterion": "line SNR = p99 of robust z of the f-k fan-filtered log envelope image",
            "v_fan_kmh": V_FAN_KMH, "lf_band_hz": lf, "hf_band_hz": hf,
            "scores": {name: {"line_snr": at(snr, b), "frac_gt_4sigma": at(frac4, b),
                              "fan_fraction": at(fan_total, b)}
                       for name, b in (("LF", lf), ("HF", hf), ("broadband", bb))},
            "top10_line_snr": top,
            "grid": {"f_lows": FAN_LOWS, "f_highs": FAN_HIGHS, "line_snr_mean": snr,
                     "frac_gt_4sigma_mean": frac4, "fan_fraction_mean": fan_total,
                     "per_segment_metrics": fan,
                     "metric_order": ["fan_pos", "fan_neg", "line_snr", "frac_gt_4sigma"]}},
        "contrast": contrast,
        "segments": segs,
    }
    path = config.RESULTS_DIR / "spectral_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(summary), indent=1), encoding="utf-8")
    print(f"\nFigures: {FIG_DIR}\nSummary: {path}")


if __name__ == "__main__":
    main()
