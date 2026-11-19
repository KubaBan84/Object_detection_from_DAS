"""Grouping of line candidates into objects and averaging of each group (plan: Stage 7).

Stage 6 can return several candidates for one vehicle: a tram line split into two
components where its response drops, the bent end of a line near the cable ends, or the
same tram seen as two touching lines. Here::

    pixel segments --physical units--> pairwise linking (velocity + time gap at the
    midpoint, or continuation along the cable) --> connected components (groups)
    --> weighted robust line fit per group --> duplicate removal + plausibility filter
    --> DetectedObject list

A line is described in physical space by its slowness p = dt/dx [s/m] and its passage
time t(x) = t_c + p (x - x_c); its velocity is v = 1/p (signed, + towards increasing
cable position).

Pixel -> physical convention (same as ``visualization.plot_waterfall`` extents): the
centre of pixel (row r, col c) is at t = (r + 0.5) row_dt, x = (c + 0.5) col_dx.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GroupingParams
from .detection import LineSegment


# ======================================================================================
# Data structures
# ======================================================================================
@dataclass
class DetectedObject:
    """One vehicle: the averaged line of a group of candidates, in physical units.

    ``x_start_m`` / ``t_start_s`` is where the object enters the observed section (first in
    time), ``x_end_m`` / ``t_end_s`` where it leaves; ``t_ref_s`` is the passage time at
    the reference position ``x_ref_m`` (middle of the cable, possibly extrapolated).
    """
    obj_id: int
    velocity_kmh: float               # signed: + towards increasing cable position
    t_ref_s: float                    # time when the object crosses the reference position
    x_start_m: float
    x_end_m: float
    t_start_s: float
    t_end_s: float
    score: float                      # max mean response of the candidates [robust sigma]
    direction: int = 0
    x_ref_m: float = 0.0
    thickness_s: float = float("nan")  # mean temporal width of the line [s]
    support_m: float = 0.0            # length along the cable covered by candidates
    fit_rms_s: float = 0.0            # RMS residual of the group line fit [s]
    segments: list[LineSegment] = field(default_factory=list)

    @property
    def slowness(self) -> float:
        """dt/dx [s/m]."""
        return 3.6 / self.velocity_kmh

    @property
    def x_min(self) -> float:
        return min(self.x_start_m, self.x_end_m)

    @property
    def x_max(self) -> float:
        return max(self.x_start_m, self.x_end_m)

    def t_at(self, x: float | np.ndarray) -> float | np.ndarray:
        """Passage time [s] at position(s) x [m] on the averaged line."""
        return self.t_ref_s + self.slowness * (np.asarray(x) - self.x_ref_m)


# ======================================================================================
# Candidate parametrisation and linking
# ======================================================================================
def to_physical(pts: np.ndarray, row_dt: float, col_dx: float) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2) [col, row] pixel points -> (x [m], t [s])."""
    return (pts[:, 0] + 0.5) * col_dx, (pts[:, 1] + 0.5) * row_dt


def _segment_arrays(segments: list[LineSegment], row_dt: float, col_dx: float) -> dict:
    """Vectorised description of all segments: centre (x_c, t_c), x-range, slowness, velocity."""
    ends = np.array([[s.c0, s.r0, s.c1, s.r1] for s in segments], dtype=np.float64).reshape(-1, 4)
    x0, t0 = to_physical(ends[:, :2], row_dt, col_dx)
    x1, t1 = to_physical(ends[:, 2:], row_dt, col_dx)
    dx = np.where(x1 != x0, x1 - x0, 1e-9)
    p = (t1 - t0) / dx
    thick = np.array([s.thickness_px for s in segments], dtype=np.float64) * row_dt
    return {"x_c": 0.5 * (x0 + x1), "t_c": 0.5 * (t0 + t1), "p": p,
            "x_lo": np.minimum(x0, x1), "x_hi": np.maximum(x0, x1),
            "v": 3.6 / np.where(p != 0, p, 1e-9), "dir": np.array([s.direction for s in segments]),
            "thick": np.nan_to_num(thick, nan=0.0)}


def _time_gap_at(a: dict, xm: np.ndarray) -> np.ndarray:
    """|t_i(x) - t_j(x)| for all pairs, both lines evaluated at the pairwise positions xm."""
    ti = a["t_c"][:, None] + a["p"][:, None] * (xm - a["x_c"][:, None])
    tj = a["t_c"][None, :] + a["p"][None, :] * (xm - a["x_c"][None, :])
    return np.abs(ti - tj)


def link_matrix(segments: list[LineSegment], params: GroupingParams, row_dt: float,
                col_dx: float) -> np.ndarray:
    """Boolean (N, N) adjacency of candidates that belong to the same vehicle (vectorised).

    Candidates of the same direction are linked by either criterion:

    - *same line*: velocities agree (|dv| < ``tol_velocity_kmh`` or |dv| / |v| <
      ``tol_velocity_rel``) and the time gap between the lines at the midpoint
      x_m = (x_ci + x_cj) / 2 is below the tolerance;
    - *continuation*: the segments follow each other along the cable (gap between their
      x-ranges in [-``cont_max_overlap_m``, ``cont_max_gap_m``]), the lines meet at the
      junction (time gap below the tolerance) and the velocity ratio is at most
      ``cont_max_ratio``. This joins the pieces of a bent trajectory (braking,
      accelerating, line ends bent by the filters at the cable ends).

    The time tolerance is widened to half the thickness of the thicker candidate, so
    pieces along the edges of a wide tram line join the tram.
    """
    a = _segment_arrays(segments, row_dt, col_dx)
    thick = np.maximum(a["thick"][:, None], a["thick"][None, :])
    tol_t = np.maximum(params.tol_time_s, 0.5 * thick)
    same_dir = a["dir"][:, None] == a["dir"][None, :]
    vi, vj = np.abs(a["v"][:, None]), np.abs(a["v"][None, :])
    dv = np.abs(vi - vj)
    vmax, vmin = np.maximum(vi, vj), np.maximum(np.minimum(vi, vj), 1e-9)
    same_v = (dv < params.tol_velocity_kmh) | (dv < params.tol_velocity_rel * vmax)
    same_line = same_v & (_time_gap_at(a, 0.5 * (a["x_c"][:, None] + a["x_c"][None, :])) < tol_t)
    # continuation: facing ends = right end of the left segment, left end of the right one
    left_i = a["x_c"][:, None] <= a["x_c"][None, :]
    xa = np.where(left_i, a["x_hi"][:, None], a["x_hi"][None, :])
    xb = np.where(left_i, a["x_lo"][None, :], a["x_lo"][:, None])
    gap_x = xb - xa
    cont = ((gap_x <= params.cont_max_gap_m) & (gap_x >= -params.cont_max_overlap_m)
            & (vmax / vmin <= params.cont_max_ratio)
            & (_time_gap_at(a, 0.5 * (xa + xb)) < tol_t))
    return same_dir & (same_line | cont)


def connected_components(adj: np.ndarray) -> np.ndarray:
    """Component label of every node of a symmetric adjacency matrix.

    Min-label propagation (vectorised over nodes): each node repeatedly takes the
    smallest label among its neighbours until nothing changes. Labels are 0..K-1.
    """
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    a = adj | adj.T | np.eye(n, dtype=bool)
    lab = np.arange(n)
    while True:
        new = np.where(a, lab[None, :], n).min(axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
    return np.unique(lab, return_inverse=True)[1]


def group_segments(segments: list[LineSegment], params: GroupingParams, row_dt: float,
                   col_dx: float) -> list[list[LineSegment]]:
    """Single-linkage grouping of candidates (connected components of :func:`link_matrix`)."""
    if not segments:
        return []
    lab = connected_components(link_matrix(segments, params, row_dt, col_dx))
    return [[segments[i] for i in np.nonzero(lab == k)[0]] for k in range(lab.max() + 1)]


# ======================================================================================
# Averaging a group
# ======================================================================================
def interval_union_length(lo: np.ndarray, hi: np.ndarray) -> float:
    """Total length of the union of intervals [lo_i, hi_i] (vectorised sweep).

    Used as the *support* of an object: the cable length actually covered by its
    candidates (gaps between fragments are not counted).
    """
    if len(lo) == 0:
        return 0.0
    o = np.argsort(lo)
    lo, hi = np.asarray(lo, float)[o], np.asarray(hi, float)[o]
    reach = np.maximum.accumulate(hi)
    start = np.r_[True, lo[1:] > reach[:-1]]          # a new block starts after a gap
    first = np.nonzero(start)[0]
    return float(np.sum(np.maximum.reduceat(hi, first) - lo[first]))


def weighted_line_fit(x: np.ndarray, t: np.ndarray, w: np.ndarray, iterations: int = 2,
                      min_sigma: float = 0.04) -> tuple[float, float, float, np.ndarray]:
    """Robust weighted least squares t = t_c + p (x - x_c).

    After every pass, points with |residual| > 3 sigma (sigma = 1.4826 MAD, at least
    ``min_sigma``) get zero weight. Returns (p, x_c, t_c, inlier mask).
    """
    keep = w > 0
    p = xc = tc = 0.0
    for it in range(iterations + 1):
        ww = w * keep
        xc = float(np.sum(ww * x) / np.sum(ww))
        tc = float(np.sum(ww * t) / np.sum(ww))
        sxx = float(np.sum(ww * (x - xc) ** 2))
        p = float(np.sum(ww * (x - xc) * (t - tc)) / sxx) if sxx > 0 else 0.0
        if it == iterations:
            break
        res = t - (tc + p * (x - xc))
        sigma = max(1.4826 * float(np.median(np.abs(res[keep]))), min_sigma)
        new = (w > 0) & (np.abs(res) <= 3 * sigma)
        if new.sum() < 2 or np.array_equal(new, keep):
            break
        keep = new
    return p, xc, tc, keep


def merge_group(group: list[LineSegment], obj_id: int, row_dt: float, col_dx: float,
                x_ref_m: float = 0.0, iterations: int = 2) -> DetectedObject:
    """Average line of a group: robust weighted fit to points sampled along all segments.

    Every segment contributes one point per pixel of length (so long segments weigh
    more), weighted by its score (stronger candidates weigh more). The object extent is
    the extent of the inlier points; the velocity is 1 / slowness of the fitted line.
    """
    pts, w = [], []
    for s in group:
        p = s.points()
        pts.append(p)
        w.append(np.full(len(p), max(s.score, 1e-3)))
    pts, w = np.concatenate(pts), np.concatenate(w)
    x, t = to_physical(pts, row_dt, col_dx)
    p, xc, tc, inl = weighted_line_fit(x, t, w, iterations, min_sigma=row_dt)
    xi = x[inl]
    xmin, xmax = float(xi.min()), float(xi.max())
    p = p if p != 0 else 1e-9
    t_of = lambda xx: tc + p * (xx - xc)  # noqa: E731
    (xs, ts), (xe, te) = sorted([(xmin, t_of(xmin)), (xmax, t_of(xmax))], key=lambda q: q[1])
    res = t[inl] - t_of(x[inl])
    ga = _segment_arrays(group, row_dt, col_dx)
    return DetectedObject(
        obj_id=obj_id, velocity_kmh=3.6 / p, t_ref_s=float(t_of(x_ref_m)),
        x_start_m=xs, x_end_m=xe, t_start_s=float(ts), t_end_s=float(te),
        score=float(max(s.score for s in group)), direction=int(group[0].direction),
        x_ref_m=x_ref_m, thickness_s=float(np.nanmean([s.thickness_px for s in group]) * row_dt),
        support_m=interval_union_length(ga["x_lo"], ga["x_hi"]),
        fit_rms_s=float(np.sqrt(np.mean(res ** 2))), segments=list(group))


# ======================================================================================
# Clean-up
# ======================================================================================
def _overlap_gap(a: DetectedObject, b: DetectedObject) -> tuple[float, float]:
    """(overlap along x as a fraction of the shorter object, max time gap on the overlap)."""
    lo, hi = max(a.x_min, b.x_min), min(a.x_max, b.x_max)
    shorter = max(min(a.x_max - a.x_min, b.x_max - b.x_min), 1e-9)
    if hi <= lo:
        return 0.0, float("inf")
    xs = np.array([lo, hi])
    return (hi - lo) / shorter, float(np.max(np.abs(a.t_at(xs) - b.t_at(xs))))


def remove_duplicates(objects: list[DetectedObject], params: GroupingParams,
                      channel_dx_m: float | None = None) -> list[DetectedObject]:
    """Remove duplicates of the same vehicle and implausible objects.

    Two objects of the same direction are duplicates when they overlap along the cable by
    at least ``dup_overlap`` of the shorter one and their lines stay within the time
    tolerance at both ends of the overlap (e.g. a fragment whose velocity was too
    different to be linked in :func:`group_segments`); the one with the larger
    support x score is kept. Then objects shorter than ``min_channels`` channels or
    with |v| outside [v_min, v_max] are dropped.
    """
    order = sorted(range(len(objects)), key=lambda i: -objects[i].support_m * objects[i].score)
    kept: list[DetectedObject] = []
    for i in order:
        o = objects[i]
        dup = False
        for k in kept:
            if k.direction != o.direction:
                continue
            frac, gap = _overlap_gap(o, k)
            tol = max(params.tol_time_s, 0.5 * np.nan_to_num(k.thickness_s, nan=0.0))
            if frac >= params.dup_overlap and gap < tol:
                dup = True
                break
        if not dup:
            kept.append(o)
    min_ext = params.min_channels * channel_dx_m if channel_dx_m else 0.0
    return [o for o in kept if (o.x_max - o.x_min) >= min_ext
            and params.v_min_kmh <= abs(o.velocity_kmh) <= params.v_max_kmh]


def build_objects(segments: list[LineSegment], params: GroupingParams, row_dt: float,
                  col_dx: float, x_ref_m: float, channel_dx_m: float
                  ) -> tuple[list[DetectedObject], list[list[LineSegment]]]:
    """Full Stage 7: group -> merge -> remove duplicates -> renumber by passage time.

    Returns (objects, groups); ``groups`` (before clean-up) is kept for the
    "candidates -> groups -> mean line" figure.
    """
    groups = group_segments(segments, params, row_dt, col_dx)
    objs = [merge_group(g, k, row_dt, col_dx, x_ref_m, params.fit_iterations)
            for k, g in enumerate(groups)]
    objs = sorted(remove_duplicates(objs, params, channel_dx_m), key=lambda o: o.t_ref_s)
    for k, o in enumerate(objs):
        o.obj_id = k
    return objs, groups
