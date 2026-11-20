"""Detection of the thick, clearly visible lines (trams, heavy trucks) (plan: Stage 6).

Input: the robust-z response maps of Stage 5 (rows = time, ``row_dt`` s; columns =
upsampled space, ``col_dx`` m), one per direction (+1 / -1). Each map is processed
separately, so a line can only have the slope sign of its direction::

    response --threshold (Otsu, floored at 4 sigma)--> binary mask
             --connected components (area / extent filters)-->
             --ridge = local maxima along time--> sequential RANSAC line fit
             --> LineSegment candidates (pixel coordinates)

Stage 7 (``grouping``) converts the candidates to physical units, groups the candidates
of one vehicle and averages them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import LineModelND, label, ransac, regionprops

from .config import DetectionParams

MAD_TO_SIGMA = 1.4826


# ======================================================================================
# Data structures
# ======================================================================================
@dataclass
class LineSegment:
    """Line segment in image pixel coordinates (col = space, row = time).

    Endpoints are ordered in time (``r0 <= r1``). ``score`` is the mean response on the
    ridge of the line (robust z), ``thickness_px`` the mean temporal width of the
    supporting component in rows.
    """
    c0: float
    r0: float
    c1: float
    r1: float
    score: float
    direction: int                     # +1 / -1
    thickness_px: float = float("nan")

    def __post_init__(self) -> None:
        if self.r1 < self.r0:
            self.c0, self.r0, self.c1, self.r1 = self.c1, self.r1, self.c0, self.r0

    @property
    def length_px(self) -> float:
        return float(np.hypot(self.c1 - self.c0, self.r1 - self.r0))

    @property
    def slope(self) -> float:
        """Columns per row (d col / d row); inf for a horizontal segment."""
        dr = self.r1 - self.r0
        return (self.c1 - self.c0) / dr if dr != 0 else float(np.copysign(np.inf, self.c1 - self.c0))

    def points(self, step: float = 1.0) -> np.ndarray:
        """(N, 2) points [col, row] sampled every ``step`` px along the segment."""
        n = max(int(np.ceil(self.length_px / step)) + 1, 2)
        s = np.linspace(0.0, 1.0, n)
        return np.column_stack([self.c0 + s * (self.c1 - self.c0), self.r0 + s * (self.r1 - self.r0)])


@dataclass
class Detections:
    """All line candidates of one recording (both directions) and the intermediate masks.

    ``stages`` keys: ``{pos|neg}_{binary|ridge|mask}``; ``thresholds[pos|neg]``
    holds the data-driven threshold of every map.
    """
    segments: list[LineSegment] = field(default_factory=list)
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)


# ======================================================================================
# Helpers
# ======================================================================================
def robust_noise_sigma(img: np.ndarray) -> float:
    """1.4826 * MAD of the image.

    Basis of the data-driven threshold: unlike the std, it is not inflated by the lines
    themselves (heavy-tailed data, EDA). Returns 1.0 if the MAD is zero.
    """
    v = np.asarray(img, dtype=np.float64).ravel()
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(v - np.median(v))))
    return sigma if sigma > 0 else 1.0


def velocity_to_slope(v_ms: float | np.ndarray, row_dt: float, col_dx: float):
    """Velocity [m/s] -> image slope [columns per row] (signed)."""
    return np.asarray(v_ms) * row_dt / col_dx


def slope_bounds(v_range_kmh: tuple[float, float], direction: int, row_dt: float,
                 col_dx: float) -> tuple[float, float]:
    """Admissible signed slopes (lo, hi) [col/row] of one direction."""
    s = np.sort(velocity_to_slope(np.asarray(v_range_kmh) / 3.6, row_dt, col_dx) * np.sign(direction))
    return float(s[0]), float(s[1])


def slope_ok(slope: float | np.ndarray, bounds: tuple[float, float], margin: float):
    """True where the slope lies in the admissible range (relative ``margin`` of tolerance)."""
    lo, hi = bounds
    return (np.asarray(slope) >= lo - margin * abs(lo)) & (np.asarray(slope) <= hi + margin * abs(hi))


def ridge_along_time(response: np.ndarray, mask: np.ndarray, nms_rows: int = 1) -> np.ndarray:
    """Pixels of ``mask`` that are maxima of the response within +-``nms_rows`` along time.

    Non-maximum suppression with a vertical max filter (``cv2.dilate``, vectorised). A
    thick slanted line gives one ridge pixel per column (its centre); two parallel lines
    more than ``nms_rows`` apart that touch in the binary mask still give two ridges.
    """
    r = np.asarray(response, dtype=np.float32)
    se = np.ones((2 * max(int(nms_rows), 1) + 1, 1), dtype=np.uint8)
    local_max = cv2.dilate(r, se, borderType=cv2.BORDER_REPLICATE)
    return mask & (r >= local_max)


def fit_line_ransac(cols: np.ndarray, rows: np.ndarray, residual_px: float, max_trials: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC (skimage ``LineModelND``) on [col, row] points -> (origin, unit direction, inliers).

    The final line is a least-squares refit on the inliers (origin = centroid, direction
    = principal axis).
    """
    data = np.column_stack([cols, rows]).astype(np.float64)
    if len(data) < 3:
        d = data[-1] - data[0]
        return data[0], d / max(np.linalg.norm(d), 1e-12), np.ones(len(data), bool)
    model, inliers = ransac(data, LineModelND, min_samples=2, residual_threshold=residual_px,
                            max_trials=max_trials, rng=seed)
    if model is None or inliers is None or inliers.sum() < 2:
        inliers = np.ones(len(data), bool)
    pts = data[inliers]
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
    return origin, vt[0], inliers


def sequential_ransac(cols: np.ndarray, rows: np.ndarray, params: DetectionParams,
                      min_cols: float, bounds: tuple[float, float]
                      ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Several lines in one point set: RANSAC, remove the inliers, repeat.

    Needed when two thick lines (e.g. two trams close behind each other) touch and form
    one connected component. A line is kept when its inliers span at least ``min_cols``
    columns and its slope is admissible; then all points closer than ``ridge_nms_rows``
    (along time) to it are removed, so leftovers of the same thick line cannot start
    another line. A later line that runs alongside no accepted line (column-range overlap
    below ``parallel_overlap``) is not a second vehicle - two vehicles of one direction
    both cross the cable, so they overlap in x - but the bent continuation of a line
    (e.g. near the cable ends), so it only extends that line. The loop stops after
    ``max_lines`` lines or when too few points remain.
    Returns a list of (origin, direction, inlier indices into the input arrays).
    """
    left = np.arange(cols.size)
    out = []
    for k in range(params.max_lines * 2):
        if left.size < max(min_cols / 2, 3) or len(out) >= params.max_lines:
            break
        origin, d, inl = fit_line_ransac(cols[left], rows[left], params.ransac_residual_px,
                                         params.ransac_max_trials, params.seed + k)
        idx = left[inl]
        lo, hi = cols[idx].min(), cols[idx].max()
        # overlap (fraction of the new line's column range) with every accepted line
        overlap = [(min(hi, cols[j].max()) - max(lo, cols[j].min())) / max(hi - lo, 1.0)
                   for _, _, j in out]
        if out and max(overlap) < params.parallel_overlap:
            # does not run alongside any accepted line -> the (bent) continuation of the
            # nearest one (one vehicle), so it only extends that line's extent
            k = int(np.argmax(overlap))
            out[k] = (out[k][0], out[k][1], np.union1d(out[k][2], idx))
        elif (abs(d[0]) > 1e-9 and abs(d[1]) > 1e-9 and hi - lo + 1 >= min_cols
              and slope_ok(d[0] / d[1], bounds, params.slope_margin)):
            out.append((origin, d, idx))
            r_line = origin[1] + (cols[left] - origin[0]) * d[1] / d[0]
            inl = inl | (np.abs(rows[left] - r_line) <= params.ridge_nms_rows)
        left = left[~inl]
    return out


# ======================================================================================
# Detection on one map and on both directions
# ======================================================================================
def detection_threshold(response: np.ndarray, params: DetectionParams) -> float:
    """Otsu threshold of the positive response, floored at ``min_sigma`` robust sigma.

    The floor matters in a direction without heavy traffic: Otsu then separates only
    noise texture from slightly stronger noise texture.
    """
    med = float(np.median(response))
    sigma = robust_noise_sigma(response)
    floor = med + params.min_sigma * sigma
    if params.threshold == "otsu":
        pos = response[response > med]
        thr = float(threshold_otsu(pos)) if pos.size and np.ptp(pos) > 0 else floor
    else:
        thr = med + float(params.threshold) * sigma
    return max(thr, floor)


def detect_thick_lines(response: np.ndarray, params: DetectionParams, row_dt: float,
                       col_dx: float, direction: int, v_range_kmh: tuple[float, float]
                       ) -> tuple[list[LineSegment], dict[str, np.ndarray], float]:
    """Threshold -> components -> ridge -> sequential RANSAC.

    Components smaller than ``min_area_px`` or spanning less than ``min_extent_m`` along
    the cable are rejected. In each remaining component the ridge (local maxima of the
    response along time) is fitted with sequential RANSAC, so touching parallel thick
    lines give separate lines. Each component pixel is assigned to the nearest fitted
    line (in rows); the line thickness is the mean number of assigned pixels per column.
    Returns (segments, stages {binary, ridge, mask}, threshold).
    """
    bounds = slope_bounds(v_range_kmh, direction, row_dt, col_dx)
    thr = detection_threshold(response, params)
    binary = response > thr
    ridge = ridge_along_time(response, binary, params.ridge_nms_rows)
    lab = label(binary, connectivity=2)
    mask = np.zeros(response.shape, dtype=bool)
    segments = []
    min_cols = params.min_extent_m / col_dx
    for reg in regionprops(lab):                          # loop over components
        rr, cc = reg.coords[:, 0], reg.coords[:, 1]
        if reg.area < params.min_area_px or np.ptp(cc) + 1 < min_cols:
            continue
        on_ridge = ridge[rr, cc]
        rcols, rrows = cc[on_ridge], rr[on_ridge]
        lines = sequential_ransac(rcols, rrows, params, min_cols, bounds)
        if not lines:
            continue
        # assign every pixel of the component to the nearest line (vertical distance);
        # row on line k at each pixel's column: r = o_r + (c - o_c) * d_r / d_c
        r_line = np.stack([o[1] + (cc - o[0]) * (d[1] / d[0] if abs(d[0]) > 1e-9 else 0.0)
                           for o, d, _ in lines])
        nearest = np.argmin(np.abs(rr[None, :] - r_line), axis=0)
        for k, (origin, d, idx) in enumerate(lines):
            slope = d[0] / d[1]
            cmin, cmax = float(rcols[idx].min()), float(rcols[idx].max())
            r_at = lambda c: origin[1] + (c - origin[0]) / slope  # noqa: E731
            own = nearest == k
            n_cols = max(np.unique(cc[own]).size, 1)
            segments.append(LineSegment(cmin, r_at(cmin), cmax, r_at(cmax),
                                        score=float(np.mean(response[rrows[idx], rcols[idx]])),
                                        direction=direction,
                                        thickness_px=float(own.sum() / n_cols)))
        mask[rr, cc] = True
    stages = {"binary": binary, "ridge": ridge, "mask": mask}
    return segments, stages, thr


def detect_lines(enhanced, params: DetectionParams, v_range_kmh: tuple[float, float]
                 ) -> Detections:
    """Run :func:`detect_thick_lines` on the response map of both directions.

    ``enhanced`` is the ``enhancement.Enhanced`` result of Stage 5.
    """
    out = Detections()
    for d, dname in ((+1, "pos"), (-1, "neg")):
        segs, stages, thr = detect_thick_lines(enhanced.response[d], params, enhanced.row_dt,
                                               enhanced.col_dx, d, v_range_kmh)
        out.segments += segs
        out.stages.update({f"{dname}_{k}": v for k, v in stages.items()})
        out.thresholds[dname] = thr
    return out
