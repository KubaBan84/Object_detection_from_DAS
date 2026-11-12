"""Per-object descriptors used for clustering (plan: Stage 10, extra 2).

Every detected object is described by a fixed set of numbers that capture what the brief
lists as clustering criteria - thick / thin lines, constant / varying speed and the shape
of the line::

    trajectory t(x) (Stage 9 track, or the Stage 7 line) --cut +-2 s around t(x) in every
    channel of the f-k energy map (fancy indexing)--> straightened cross-profiles
    (channels x time offset), re-centred on the ridge maximum of each channel
        -> thickness (FWHM of the cross-profile [s]), peak z, area under the profile,
           number of parallel sub-lines (fringes of the bogies / axles of a tram),
           tail asymmetry
    raw strain rate in a 0.8 s window on the trajectory (every channel) --rFFT-->
        power relative to the background spectrum of the channel
        -> spectral centroid of the excess power, LF/HF energy ratio [dB]
    Stage 8-9 fields -> |v|, relative change of speed, std of v(t), |acceleration|, R^2

All computations are vectorised over channels and windows; loops run only over objects.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FeatureParams, TrackingParams
from .grouping import DetectedObject
from .tracking import tracking_map
from .velocity import reported_velocity_kmh

FEATURE_NAMES = [
    # cross-profile of the line (thick / thin, shape)
    "thickness_s", "peak_z", "log_peak_z", "area_zs", "log_area_zs", "n_sublines",
    "tail_asym", "z_hf", "z_lf",
    # raw-signal spectrum on the trajectory
    "spec_centroid_hz", "lf_hf_db", "snr_lf_db", "snr_hf_db",
    # motion (Stage 8-9)
    "abs_v_kmh", "direction", "dv_rel", "v_std_kmh", "accel_ms2", "abs_accel_ms2", "r2",
    "varying", "log_dv_rel", "log_abs_accel", "log_v_std",
    # geometry and detection
    "duration_s", "extent_m", "n_channels", "strong", "score",
]


@dataclass
class FeatureContext:
    """Maps and signals shared by all objects of one recording."""
    energy: dict[int, np.ndarray]                  # direction -> z map (rows, channels), bands summed
    band_energy: dict[str, dict[int, np.ndarray]]  # band -> direction -> z map
    row_dt: float                                  # [s] per map row
    dx: float                                      # [m] channel spacing
    raw: np.ndarray | None = None                  # (samples, channels) raw strain rate
    fs: float = 0.0
    raw_offset: int = 0                            # raw sample at the start of map row 0
    freqs: np.ndarray | None = None                # rFFT frequencies of the spectral window
    bg_power: np.ndarray | None = None             # (n_freq, channels) median background power


@dataclass
class CrossSection:
    """Straightened cross-profiles of one object (one row per channel)."""
    cols: np.ndarray              # channel indices
    r_peak: np.ndarray            # ridge row in every channel
    profiles: np.ndarray          # (n, 2w + 1) map values, NaN outside the image
    baseline: np.ndarray          # (n,) per-channel baseline

    @property
    def halfwidth(self) -> int:
        return (self.profiles.shape[1] - 1) // 2


# ======================================================================================
# Context
# ======================================================================================
def background_power(raw: np.ndarray, n_win: int) -> tuple[np.ndarray, np.ndarray]:
    """Median power spectrum of every channel over non-overlapping windows of n_win samples.

    The median over ~150 windows of a 2-min recording ignores the few windows with a
    vehicle, so it estimates the background (noise) spectrum. Returns (freqs [cycles per
    sample], power (n_freq, channels)); multiply freqs by fs for Hz.
    """
    n_blocks = raw.shape[0] // n_win
    blocks = raw[:n_blocks * n_win].reshape(n_blocks, n_win, raw.shape[1]).astype(np.float64)
    blocks = blocks - blocks.mean(axis=1, keepdims=True)
    win = np.hanning(n_win)[None, :, None]
    power = np.abs(np.fft.rfft(blocks * win, axis=1)) ** 2
    return np.fft.rfftfreq(n_win), np.median(power, axis=0)


def feature_context(enhanced: dict, row_dt: float, params: FeatureParams,
                    raw: np.ndarray | None = None, fs: float = 0.0, raw_offset: int = 0
                    ) -> FeatureContext:
    """Energy maps (f-k map per direction, robust z, no smoothing) + raw background spectrum.

    ``raw`` is the raw (samples, channels) recording whose sample ``raw_offset`` is the
    start of map row 0 (the padding of a segment); without it the spectral features are NaN.
    """
    tp = TrackingParams(track_map="fk", smooth_rows=0.0)
    energy, dx = {}, 0.0
    for d in (1, -1):
        energy[d], dx = tracking_map(enhanced, d, tp, "sum")
    band_energy = {b: {d: tracking_map(enhanced, d, tp, b)[0] for d in (1, -1)}
                   for b in enhanced}
    ctx = FeatureContext(energy, band_energy, row_dt, dx, raw, fs, raw_offset)
    if raw is not None:
        n_win = int(round(params.spec_window_s * fs))
        f, ctx.bg_power = background_power(raw, n_win)
        ctx.freqs = f * fs
    return ctx


# ======================================================================================
# Cross-profile of one object
# ======================================================================================
def object_trajectory(obj: DetectedObject, n_channels: int, dx: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(channel indices, passage times [s]) of an object: the Stage-9 track if it exists,
    otherwise the averaged Stage-7 line over the detected extent."""
    if obj.track is not None and len(obj.track) >= 2:
        cols = np.rint(obj.track[:, 0] / dx - 0.5).astype(int)
        t = obj.track[:, 1].astype(float)
    else:
        c0 = int(np.ceil(obj.x_min / dx - 0.5))
        c1 = int(np.floor(obj.x_max / dx - 0.5))
        cols = np.arange(max(c0, 0), min(c1, n_channels - 1) + 1)
        t = np.asarray(obj.t_at((cols + 0.5) * dx), float).reshape(-1)
    keep = (cols >= 0) & (cols < n_channels)
    return cols[keep], t[keep]


def _windows(img: np.ndarray, cols: np.ndarray, r_centre: np.ndarray, w: int) -> np.ndarray:
    """(n, 2w + 1) values img[r_centre + k, cols] for k = -w..w, NaN outside the image."""
    rows = r_centre[:, None] + np.arange(-w, w + 1)[None, :]
    inside = (rows >= 0) & (rows < img.shape[0])
    vals = img[np.clip(rows, 0, img.shape[0] - 1), cols[:, None]]
    return np.where(inside, vals, np.nan)


def cross_section(energy: np.ndarray, cols: np.ndarray, t: np.ndarray, row_dt: float,
                  params: FeatureParams) -> CrossSection:
    """Cut the profile across the line in every channel, centred on the ridge maximum.

    The trajectory (track or straight line) can be off the ridge by a few rows, so the
    maximum within +-``peak_search_rows`` of t(x) is taken as the centre first. The
    baseline is a low quantile of each profile (the background next to the line).
    """
    w = int(round(params.profile_halfwidth_s / row_dt))
    s = params.peak_search_rows
    r0 = np.rint(t / row_dt - 0.5).astype(int)
    inside = (r0 >= 0) & (r0 < energy.shape[0])       # extrapolated lines may leave the image
    cols, r0 = cols[inside], r0[inside]
    near = _windows(energy, cols, r0, s)
    shift = np.argmax(np.where(np.isfinite(near), near, -np.inf), axis=1) - s
    r_peak = r0 + shift
    prof = _windows(energy, cols, r_peak, w)
    base = np.nanquantile(prof, params.baseline_quantile, axis=1)
    return CrossSection(cols, r_peak, prof, base)


def fwhm_rows(p: np.ndarray) -> np.ndarray:
    """Full width at half maximum [rows] of profiles peaking at the centre (vectorised).

    ``p`` (n, 2w + 1) baseline-free profiles. The run of samples >= half of the central
    value is followed to both sides; the half-maximum crossings are interpolated linearly
    between the last sample above and the first below (0.5 sample at the window edge).
    NaN where the central value is not positive.
    """
    n, m = p.shape
    w = (m - 1) // 2
    idx = np.arange(n)
    peak = p[:, w]
    half = 0.5 * peak
    above = np.nan_to_num(p, nan=-np.inf) >= half[:, None]
    left = above[:, :w][:, ::-1]                        # centre-1, centre-2, ...
    right = above[:, w + 1:]
    run_l = np.where(left.all(axis=1), w, np.argmin(left, axis=1))
    run_r = np.where(right.all(axis=1), w, np.argmin(right, axis=1))

    def frac(i_in, i_out, edge):
        """Fraction of a sample between the last sample above and the half crossing."""
        y_in = p[idx, i_in]
        y_out = p[idx, np.clip(i_out, 0, m - 1)]
        den = y_in - y_out
        ok = ~edge & np.isfinite(y_out) & (den > 0)
        return np.where(ok, (y_in - half) / np.where(ok, den, 1.0), 0.5)

    i_l, i_r = w - run_l, w + run_r
    f_l = frac(i_l, i_l - 1, i_l - 1 < 0)
    f_r = frac(i_r, i_r + 1, i_r + 1 > m - 1)
    width = (i_r + f_r) - (i_l - f_l)
    return np.where(peak > 0, width, np.nan)


def count_sublines(profile: np.ndarray, rel_height: float, min_dip: float) -> int:
    """Number of parallel lines in a mean cross-profile (1 = a single line).

    A local maximum counts as a sub-line if it reaches ``rel_height`` of the main peak
    and the profile between it and the main peak dips by at least ``min_dip`` x main peak
    below it (two bogies of a tram -> 2, a car -> 1). Loops only over the few maxima.
    """
    p = np.nan_to_num(profile, nan=0.0)
    if p.size < 3 or p.max() <= 0:
        return 0
    main = int(np.argmax(p))
    top = p[main]
    is_max = np.zeros(p.size, bool)
    is_max[1:-1] = (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])
    n = 1
    for k in np.flatnonzero(is_max & (p >= rel_height * top)):
        if k == main:
            continue
        lo, hi = sorted((k, main))
        if p[lo:hi + 1].min() <= p[k] - min_dip * top:
            n += 1
    return n


def _smooth_1d(p: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return p
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(p.reshape(1, -1).astype(np.float64), (k, 1), sigmaX=sigma,
                            borderType=cv2.BORDER_REFLECT).reshape(-1)


def profile_features(cs: CrossSection, row_dt: float, params: FeatureParams) -> dict:
    """Thickness, peak, area, sub-lines and tail asymmetry from the cross-profiles."""
    p = cs.profiles - cs.baseline[:, None]
    w = cs.halfwidth
    peak = p[:, w]
    width = fwhm_rows(p) * row_dt
    pos = np.clip(np.nan_to_num(p, nan=0.0), 0.0, None)
    area = pos.sum(axis=1) * row_dt
    good = peak > 0
    out = {"thickness_s": float(np.nanmedian(width[good])) if good.any() else np.nan,
           "peak_z": float(np.median(cs.profiles[:, w])),
           "area_zs": float(np.median(area))}
    # mean profile normalised by each channel's peak (loud channels do not dominate)
    if good.any():
        q = p[good] / peak[good, None]
        cnt = np.isfinite(q).sum(axis=0)
        mean = np.nansum(q, axis=0) / np.maximum(cnt, 1)             # NaN-safe mean
        mean = _smooth_1d(mean, params.subline_smooth_rows)
        out["n_sublines"] = count_sublines(mean, params.subline_rel_height, params.subline_min_dip)
        mpos = np.clip(mean, 0.0, None)
        before, after = mpos[:w].sum(), mpos[w + 1:].sum()
        out["tail_asym"] = float((after - before) / (after + before)) if after + before > 0 else 0.0
    else:
        out["n_sublines"], out["tail_asym"] = 0, np.nan
    out["log_peak_z"] = float(np.log10(1.0 + max(out["peak_z"], 0.0)))
    out["log_area_zs"] = float(np.log10(1.0 + max(out["area_zs"], 0.0)))
    return out


# ======================================================================================
# Raw spectrum on the trajectory
# ======================================================================================
def spectral_features(ctx: FeatureContext, cols: np.ndarray, t_peak: np.ndarray,
                      params: FeatureParams) -> dict:
    """Spectrum of the raw signal in a window on the trajectory, relative to the background.

    For every channel the raw samples in ``spec_window_s`` centred on the passage time
    are cut out at once ((n, L) fancy indexing), Hann-windowed and transformed (rFFT).
    - ``snr_lf_db`` / ``snr_hf_db``: band energy of the object over the background energy
      of the same channels, in dB; ``lf_hf_db`` = their difference (low-frequency rich
      heavy vehicles > 0);
    - ``spec_centroid_hz``: centroid of the excess power max(P - P_bg, 0) between the
      lower edge of the LF band and the upper edge of the HF band.
    """
    nan = {"spec_centroid_hz": np.nan, "lf_hf_db": np.nan, "snr_lf_db": np.nan,
           "snr_hf_db": np.nan}
    if ctx.raw is None or ctx.bg_power is None or cols.size == 0:
        return nan
    n_win = int(round(params.spec_window_s * ctx.fs))
    s0 = ctx.raw_offset + np.rint(t_peak * ctx.fs).astype(int) - n_win // 2
    ok = (s0 >= 0) & (s0 + n_win <= ctx.raw.shape[0])
    if not ok.any():
        return nan
    rows = s0[ok, None] + np.arange(n_win)[None, :]
    seg = ctx.raw[rows, cols[ok, None]].astype(np.float64)          # (n, L)
    seg = seg - seg.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(seg * np.hanning(n_win)[None, :], axis=1)) ** 2
    bg = ctx.bg_power[:, cols[ok]].T                                 # (n, n_freq)
    f = ctx.freqs
    lf = (f >= params.spec_lf_band[0]) & (f <= params.spec_lf_band[1])
    hf = (f >= params.spec_hf_band[0]) & (f <= params.spec_hf_band[1])
    eps = 1e-30
    snr_lf = 10 * np.log10((power[:, lf].sum() + eps) / (bg[:, lf].sum() + eps))
    snr_hf = 10 * np.log10((power[:, hf].sum() + eps) / (bg[:, hf].sum() + eps))
    band = (f >= params.spec_lf_band[0]) & (f <= params.spec_hf_band[1])
    excess = np.clip(power - bg, 0.0, None)[:, band].sum(axis=0)
    centroid = float((f[band] * excess).sum() / excess.sum()) if excess.sum() > 0 else np.nan
    return {"spec_centroid_hz": centroid, "lf_hf_db": float(snr_lf - snr_hf),
            "snr_lf_db": float(snr_lf), "snr_hf_db": float(snr_hf)}


# ======================================================================================
# All features of one object
# ======================================================================================
def motion_features(obj: DetectedObject) -> dict:
    """|v|, direction, relative change of speed, std of v(t), acceleration, R^2 (Stage 8-9)."""
    v = reported_velocity_kmh(obj)
    dv = abs(obj.v_end_kmh - obj.v_start_kmh) / abs(v) if np.isfinite(obj.v_end_kmh) else np.nan
    a = abs(obj.accel_ms2)
    # the speed-change measures are heavy-tailed (most vehicles ~constant, a few brake
    # hard): their logarithms are closer to symmetric, which suits k-means; the offsets
    # are the measurement noise levels (2 %, 0.05 m/s^2, 1 km/h)
    logs = {"log_dv_rel": float(np.log10(0.02 + dv)) if np.isfinite(dv) else np.nan,
            "log_abs_accel": float(np.log10(0.05 + a)) if np.isfinite(a) else np.nan,
            "log_v_std": float(np.log10(1.0 + obj.v_std_kmh)) if np.isfinite(obj.v_std_kmh)
            else np.nan}
    return {**logs, "abs_v_kmh": abs(v), "direction": int(obj.direction or np.sign(v)),
            "dv_rel": float(dv), "v_std_kmh": float(obj.v_std_kmh),
            "accel_ms2": float(obj.accel_ms2), "abs_accel_ms2": float(abs(obj.accel_ms2)),
            "r2": float(obj.r2),
            "varying": float(obj.speed_class == "varying") if obj.speed_class else np.nan}


def object_features(obj: DetectedObject, ctx: FeatureContext, params: FeatureParams,
                    return_section: bool = False):
    """All descriptors of one object (dict with the keys of ``FEATURE_NAMES``).

    With ``return_section`` the straightened cross-section is returned too (gallery).
    """
    energy = ctx.energy[1 if obj.direction >= 0 else -1]
    cols, t = object_trajectory(obj, energy.shape[1], ctx.dx)
    out = dict.fromkeys(FEATURE_NAMES, np.nan)
    out.update(motion_features(obj))
    out.update({"duration_s": abs(obj.t_end_s - obj.t_start_s), "extent_m": obj.x_max - obj.x_min,
                "n_channels": int(cols.size), "strong": float(obj.kind == "strong"),
                "score": float(obj.score)})
    cs = None
    if cols.size >= params.min_channels:
        cs = cross_section(energy, cols, t, ctx.row_dt, params)
    if cs is not None and cs.cols.size >= params.min_channels:
        out.update(profile_features(cs, ctx.row_dt, params))
        d = 1 if obj.direction >= 0 else -1
        for b in ("hf", "lf"):
            if b in ctx.band_energy:
                rows = np.clip(cs.r_peak, 0, energy.shape[0] - 1)
                out[f"z_{b}"] = float(np.median(ctx.band_energy[b][d][rows, cs.cols]))
        t_peak = (cs.r_peak + 0.5) * ctx.row_dt
        out.update(spectral_features(ctx, cs.cols, t_peak, params))
    else:
        cs = None
    return (out, cs) if return_section else out


def features_matrix(feature_dicts: list[dict[str, float]], keys: list[str] | tuple) -> np.ndarray:
    """(n_objects, n_keys) float matrix (missing values -> NaN)."""
    return np.array([[float(d.get(k, np.nan)) for k in keys] for d in feature_dicts],
                    dtype=np.float64).reshape(len(feature_dicts), len(keys))


def section_thumbnail(cs: CrossSection | None, n_channels: int, halfwidth: int) -> np.ndarray:
    """(2w + 1, n_channels) straightened image of the line for the cluster gallery:
    time offset from the ridge vertical, channel horizontal, NaN outside the object."""
    out = np.full((2 * halfwidth + 1, n_channels), np.nan)
    if cs is not None:
        out[:, cs.cols] = (cs.profiles - cs.baseline[:, None]).T
    return out
