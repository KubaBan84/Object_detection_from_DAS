"""Ridge tracking of each object and velocity-over-time profile (plan: Stage 9, extra 1).

Stage 7 gives every object a *straight* averaged line. A real vehicle brakes and
accelerates, so its trajectory in the (x, t) image is curved. Here the trajectory is
measured directly - one passage time per cable channel - and differentiated::

    averaged line t(x) --pass 1: argmax of the energy in +-1 s around the line,
                         parabolic sub-sample refinement (vectorised over channels)-->
    raw track --RANSAC of a polynomial t(x) (vectorised over trials)-->
    smooth curve --pass 2: argmax in +-0.5 s around the curve--> track (x_k, t_k)
    track --local linear regression in a sliding window of channels--> v(t)
    track --linear vs quadratic x(t)--> constant / varying speed, acceleration

The energy map is the f-k filtered map of the object's direction (Stage 5, one column per
channel, robust z) - it has not been smoothed along a fixed angle by the filter bank, so
the curvature of the track is not biased towards a straight line.
"""
from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import TrackingParams
from .grouping import DetectedObject
from .velocity import fit_constant_velocity

MAD_TO_SIGMA = 1.4826


# ======================================================================================
# Energy map used for tracking
# ======================================================================================
def _robust_z(img: np.ndarray) -> np.ndarray:
    med = np.median(img)
    s = MAD_TO_SIGMA * np.median(np.abs(img - med))
    return (img - med) / (s if s > 0 else 1.0)


def tracking_map(enhanced: dict, direction: int, params: TrackingParams, band: str = "sum"
                 ) -> tuple[np.ndarray, float]:
    """Robust-z energy map of one direction for tracking -> (map, column spacing [m]).

    ``track_map`` "fk": the f-k filtered Stage-4 image of the direction (one column per
    channel); "response": the filter-bank response (upsampled grid). ``band`` "sum" adds
    the z maps of all bands (the LF band is blind for x < 55 m, the HF band has wider
    lines - their sum keeps the ridge everywhere), otherwise one band. Optional Gaussian
    smoothing along time (``smooth_rows``) stabilises the argmax.
    """
    name = "fk_pos" if direction > 0 else "fk_neg"
    bands = list(enhanced) if band == "sum" else [band]
    maps = []
    for b in bands:
        e = enhanced[b]
        if params.track_map == "fk":
            m, dx = e.stages[name], e.col_dx * _upsample_factor(e)
        elif params.track_map == "response":
            m, dx = e.response[direction], e.col_dx
        else:
            raise ValueError(f"Unknown track_map {params.track_map!r}")
        maps.append(_robust_z(np.asarray(m, dtype=np.float64)))
    out = np.sum(maps, axis=0) / np.sqrt(len(maps))     # sum of z maps, again ~unit noise
    if params.smooth_rows > 0:
        k = 2 * int(np.ceil(3 * params.smooth_rows)) + 1
        out = cv2.GaussianBlur(out, (1, k), sigmaX=0.1, sigmaY=params.smooth_rows,
                               borderType=cv2.BORDER_REFLECT)
    return out, dx


def _upsample_factor(e) -> int:
    """Ratio between the response grid and the f-k (channel) grid of an Enhanced object."""
    return int(round(e.response[1].shape[1] / e.stages["fk_pos"].shape[1]))


# ======================================================================================
# Ridge search
# ======================================================================================
def ridge_times(energy: np.ndarray, cols: np.ndarray, t_pred: np.ndarray, halfwidth_s: float,
                row_dt: float, method: str = "parabola", centroid_rows: int = 6
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peak time of the energy in every column inside t_pred +- halfwidth (vectorised).

    For each column a window of rows is cut out at once (fancy indexing) and its argmax
    is refined to sub-row precision:

    - ``parabola``: vertex of the parabola through the peak and its two neighbours;
    - ``centroid``: centroid of the upper half of the peak (values above peak / 2 within
      +-``centroid_rows`` of the argmax) - more stable on the wide, flat-topped ridges of
      heavy vehicles, where noise moves the argmax across the plateau.

    A peak on the window edge is not a maximum of the ridge and is flagged invalid.
    Returns (t [s], peak value, valid mask). Pixel centres: t = (r + 0.5) row_dt.
    """
    n_rows = energy.shape[0]
    w = max(int(round(halfwidth_s / row_dt)), 1)
    r0 = np.rint(t_pred / row_dt - 0.5).astype(int)
    rows = r0[:, None] + np.arange(-w, w + 1)[None, :]
    inside = (rows >= 0) & (rows < n_rows)
    low = -1e30                                     # finite "minus infinity" (no inf arithmetic)
    vals = np.where(inside, energy[np.clip(rows, 0, n_rows - 1), cols[:, None]], low)
    k = np.argmax(vals, axis=1)
    idx = np.arange(len(cols))
    km, kp = np.clip(k - 1, 0, 2 * w), np.clip(k + 1, 0, 2 * w)
    peak, ym, yp = vals[idx, k], vals[idx, km], vals[idx, kp]
    interior = (k > 0) & (k < 2 * w) & inside[idx, k] & inside[idx, km] & inside[idx, kp]
    if method == "parabola":
        denom = np.where(interior, ym - 2 * peak + yp, -1.0)
        delta = np.where(denom < 0, 0.5 * (ym - yp) / np.where(denom < 0, denom, -1.0), 0.0)
        r = rows[idx, k] + np.clip(np.where(interior, delta, 0.0), -0.5, 0.5)
    elif method == "centroid":
        off = np.arange(-centroid_rows, centroid_rows + 1)
        kk = np.clip(k[:, None] + off[None, :], 0, 2 * w)
        near = (k[:, None] + off[None, :] >= 0) & (k[:, None] + off[None, :] <= 2 * w)
        v = np.take_along_axis(vals, kk, axis=1)
        wgt = np.where(near & np.take_along_axis(inside, kk, axis=1),
                       np.clip(v - 0.5 * peak[:, None], 0.0, None), 0.0)
        wsum = wgt.sum(axis=1)
        rr = np.take_along_axis(rows, kk, axis=1)
        r = np.where(wsum > 0, (wgt * rr).sum(axis=1) / np.where(wsum > 0, wsum, 1.0),
                     rows[idx, k])
    else:
        raise ValueError(f"Unknown peak method {method!r}")
    return (r + 0.5) * row_dt, peak, interior


def longest_run(cols: np.ndarray, max_gap: int) -> np.ndarray:
    """Mask of the longest run of sorted column indices whose gaps are <= ``max_gap``."""
    if cols.size == 0:
        return np.zeros(0, dtype=bool)
    breaks = np.r_[0, np.cumsum(np.diff(cols) > max_gap + 1)]
    best = np.argmax(np.bincount(breaks))
    return breaks == best


# ======================================================================================
# Robust trajectory fit
# ======================================================================================
def fit_trajectory_ransac(track: np.ndarray, residual_s: float, order: int = 2,
                          trials: int = 200, seed: int = 0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC of a polynomial t(x) of ``order`` on (N, 2) [x_m, t_s] points.

    All trials are solved at once: ``trials`` random minimal samples (order + 1 points)
    -> batched Vandermonde solves (``np.linalg.solve`` on a (trials, k, k) stack) ->
    residuals of every model on every point -> the model with most inliers
    (|residual| <= ``residual_s``) is refitted by least squares on its inliers.
    x is centred and scaled for conditioning. Returns (np.polyfit-style coefficients in
    the *scaled* variable together with (x_mean, x_scale) appended, inlier mask);
    use :func:`eval_trajectory` to evaluate it.
    """
    x, t = track[:, 0].astype(float), track[:, 1].astype(float)
    k = order + 1
    xm, xs = float(x.mean()), float(max(np.ptp(x) / 2, 1e-9))
    u = (x - xm) / xs
    n = x.size
    if n <= k:
        coef = np.polyfit(u, t, min(order, max(n - 1, 0)))
        return np.r_[np.zeros(k - coef.size), coef, xm, xs], np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    samples = np.argsort(rng.random((trials, n)), axis=1)[:, :k]          # distinct points
    V = u[samples][:, :, None] ** np.arange(order, -1, -1)[None, None, :]  # (trials, k, k)
    ok = np.abs(np.linalg.det(V)) > 1e-12
    coefs = np.zeros((trials, k))
    coefs[ok] = np.linalg.solve(V[ok], t[samples][ok][:, :, None])[:, :, 0]
    pred = (u[None, :, None] ** np.arange(order, -1, -1)[None, None, :] * coefs[:, None, :]).sum(-1)
    inl = (np.abs(pred - t[None, :]) <= residual_s) & ok[:, None]
    best = inl[np.argmax(inl.sum(axis=1))]
    if best.sum() <= k:
        best = np.ones(n, dtype=bool)
    coef = np.polyfit(u[best], t[best], order)
    return np.r_[coef, xm, xs], best


def eval_trajectory(model: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a model of :func:`fit_trajectory_ransac` at positions x [m] -> t [s]."""
    coef, xm, xs = model[:-2], model[-2], model[-1]
    return np.polyval(coef, (np.asarray(x, float) - xm) / xs)


# ======================================================================================
# Tracking of one object
# ======================================================================================
def track_ridge(energy: np.ndarray, obj: DetectedObject, params: TrackingParams,
                row_dt: float, col_dx: float) -> np.ndarray:
    """Two-pass ridge tracking of one object -> (N, 2) [x_m, t_s] sorted by x (N may be 0).

    ``energy`` is a robust-z map with columns spaced ``col_dx`` m (column centres at
    (c + 0.5) col_dx). Channels within the object's extent +- ``extend_m`` are searched:

    1. peak in t_line(x) +- ``search_halfwidth_s`` around the averaged Stage-7 line;
    2. RANSAC polynomial t(x) through the valid peaks (rejects e.g. a crossing vehicle);
    3. peak again in +- ``refine_halfwidth_s`` around the polynomial; points with a peak
       below ``min_peak_z`` or farther than ``ransac_residual_s`` from the curve are
       dropped, and only the longest run of channels (gaps <= ``max_gap_channels``) is kept.

    Peaks within ``edge_margin_s`` of the first / last image row are dropped in both
    passes (a vehicle cut by the window end would otherwise get compressed passage times
    and an exploding v(t) there).
    """
    n_cols = energy.shape[1]
    c_lo = max(int(np.floor((obj.x_min - params.extend_m) / col_dx - 0.5)), 0)
    c_hi = min(int(np.ceil((obj.x_max + params.extend_m) / col_dx - 0.5)), n_cols - 1)
    cols = np.arange(c_lo, c_hi + 1)
    x = (cols + 0.5) * col_dx
    empty = np.zeros((0, 2))
    if cols.size < params.min_points:
        return empty
    # a ridge whose centre lies beyond the first / last image row still has its maximum
    # on the image border, so peaks closer than edge_margin_s to it are biased
    t_lo, t_hi = params.edge_margin_s, energy.shape[0] * row_dt - params.edge_margin_s
    t1, p1, v1 = ridge_times(energy, cols, np.asarray(obj.t_at(x), float),
                             params.search_halfwidth_s, row_dt, params.peak_method,
                             params.centroid_rows)
    good = v1 & (p1 >= params.min_peak_z) & (t1 >= t_lo) & (t1 <= t_hi)
    if good.sum() < params.min_points:
        return empty
    model, _ = fit_trajectory_ransac(np.column_stack([x[good], t1[good]]), params.ransac_residual_s,
                                     params.poly_order, params.ransac_trials, params.seed)
    curve = eval_trajectory(model, x)
    t2, p2, v2 = ridge_times(energy, cols, curve, params.refine_halfwidth_s, row_dt,
                             params.peak_method, params.centroid_rows)
    good = (v2 & (p2 >= params.min_peak_z) & (np.abs(t2 - curve) <= params.ransac_residual_s)
            & (t2 >= t_lo) & (t2 <= t_hi))
    if good.sum() < params.min_points:
        return empty
    run = longest_run(cols[good], params.max_gap_channels)
    if run.sum() < params.min_points:
        return empty
    return np.column_stack([x[good][run], t2[good][run]])


# ======================================================================================
# Velocity profile and variability
# ======================================================================================
def velocity_profile(track: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Local velocity in a sliding window of ``window`` consecutive track points.

    For every window the least-squares slope of t(x) is computed at once
    (``sliding_window_view``); v = 1 / slope. The window is centred in time, so
    v(t) is reported at the mean time of the window. Returns (t [s], v [km/h], signed),
    ordered by time. With fewer points than ``window`` the whole track is one window.
    """
    tr = track[np.argsort(track[:, 0])]
    x, t = tr[:, 0], tr[:, 1]
    w = int(min(max(window, 2), x.size))
    if x.size < 2:
        return np.zeros(0), np.zeros(0)
    X, T = sliding_window_view(x, w), sliding_window_view(t, w)
    xc, tc = X - X.mean(axis=1, keepdims=True), T - T.mean(axis=1, keepdims=True)
    slope = (xc * tc).sum(axis=1) / (xc ** 2).sum(axis=1)       # dt/dx [s/m]
    v = 3.6 / np.where(slope != 0, slope, np.nan)
    tm = T.mean(axis=1)
    o = np.argsort(tm)
    return tm[o], v[o]


def speed_variability(track: np.ndarray, profile_v: np.ndarray | None = None,
                      params: TrackingParams | None = None) -> dict:
    """Constant vs varying speed from linear and quadratic fits of x(t).

    x(t) = b + v t (constant velocity) vs x(t) = c + b t + a t^2 (constant acceleration
    2a). Reports both RMS residuals [m], the acceleration along the direction of motion
    [m/s^2] (2a sign(v): + speeding up, - braking, for both directions), the quadratic-fit
    velocity at the first and last track point [km/h], the std of the local profile and
    a label: "varying" when the speed change |v_end - v_start| is at least
    ``varying_dv_rel`` of the mean speed AND the quadratic model reduces the RMS residual
    by at least ``varying_rms_gain`` (so a noisy but straight track stays "constant").
    """
    p = params or TrackingParams()
    x, t = track[:, 0].astype(float), track[:, 1].astype(float)
    out = {"rms_lin_m": float("nan"), "rms_quad_m": float("nan"), "accel_ms2": float("nan"),
           "v_start_kmh": float("nan"), "v_end_kmh": float("nan"), "v_std_kmh": float("nan"),
           "speed_class": ""}
    if x.size < 4 or np.ptp(t) == 0:
        return out
    tc = t - t.mean()
    lin = np.polyfit(tc, x, 1)
    quad = np.polyfit(tc, x, 2)
    out["rms_lin_m"] = float(np.sqrt(np.mean((x - np.polyval(lin, tc)) ** 2)))
    out["rms_quad_m"] = float(np.sqrt(np.mean((x - np.polyval(quad, tc)) ** 2)))
    out["accel_ms2"] = float(2 * quad[0] * np.sign(lin[0]))    # + speeding up, - braking
    v_at = lambda tt: 3.6 * (2 * quad[0] * tt + quad[1])  # noqa: E731
    out["v_start_kmh"], out["v_end_kmh"] = float(v_at(tc.min())), float(v_at(tc.max()))
    if profile_v is not None and np.isfinite(profile_v).sum() > 1:
        out["v_std_kmh"] = float(np.nanstd(np.abs(profile_v)))
    dv = abs(abs(out["v_end_kmh"]) - abs(out["v_start_kmh"]))
    gain = 1.0 - out["rms_quad_m"] / max(out["rms_lin_m"], 1e-12)
    varying = dv >= p.varying_dv_rel * abs(3.6 * lin[0]) and gain >= p.varying_rms_gain
    out["speed_class"] = "varying" if varying else "constant"
    return out


# ======================================================================================
# All objects of one image
# ======================================================================================
def track_objects(objects: list[DetectedObject], enhanced: dict, params: TrackingParams,
                  row_dt: float) -> list[DetectedObject]:
    """Stages 8-9 for every object (in place): track, constant-velocity fit, v(t), class.

    Fills ``track``, ``v_track_kmh``, ``r2``, ``accel_ms2``, ``v_start_kmh``,
    ``v_end_kmh``, ``v_std_kmh``, ``speed_class`` and ``profile``. Objects whose track has
    fewer than ``min_points`` channels keep only the Stage-7 line velocity.
    """
    maps = {}
    nan = float("nan")
    for o in objects:
        # reset the Stage 8-9 fields (objects may be re-tracked with other parameters)
        o.track, o.profile, o.speed_class = np.zeros((0, 2)), None, ""
        o.v_track_kmh = o.r2 = o.accel_ms2 = o.v_start_kmh = o.v_end_kmh = o.v_std_kmh = nan
        band = params.track_band
        if band == "auto":
            band = "lf" if "lf" in o.bands and "lf" in enhanced else "hf"
        key = (o.direction, band)
        if key not in maps:
            maps[key] = tracking_map(enhanced, o.direction, params, band)
        energy, col_dx = maps[key]
        tr = track_ridge(energy, o, params, row_dt, col_dx)
        o.track = tr
        if len(tr) < params.min_points:
            continue
        v, _, r2 = fit_constant_velocity(tr)
        if not np.isfinite(v) or np.sign(v) != o.direction:
            o.track = np.zeros((0, 2))         # track went to the wrong direction: unusable
            continue
        o.v_track_kmh, o.r2 = 3.6 * v, r2
        tp, vp = velocity_profile(tr, params.velocity_window_channels)
        xp = np.interp(tp, np.sort(tr[:, 1]), tr[np.argsort(tr[:, 1]), 0])
        o.profile = np.column_stack([tp, xp, vp])
        sv = speed_variability(tr, vp, params)
        o.accel_ms2, o.v_start_kmh, o.v_end_kmh = sv["accel_ms2"], sv["v_start_kmh"], sv["v_end_kmh"]
        o.v_std_kmh, o.speed_class = sv["v_std_kmh"], sv["speed_class"]
    return objects
