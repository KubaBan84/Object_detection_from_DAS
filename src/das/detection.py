"""Line detection on the Stage-5 response maps: strong and weak paths (plan: Stage 6).

Input: robust-z response maps (rows = time, ``row_dt`` s; columns = upsampled space,
``col_dx`` m), one per band (HF, LF) and direction (+1 / -1). Every map is processed
separately, so a line can only have the slope sign of its direction.

6A - thick, strong lines (trams, trucks), HF band::

    response --threshold (Otsu, floored at k sigma)--> oriented morphological opening
             --> connected components (area / extent / orientation filters)
             --> per-column response-weighted centroid --> RANSAC line fit

   The kept components form the *strong mask* used by path 6B.

6B - thin, weak lines (cars) - grade-5 condition, HF and LF bands::

    response --mask strong objects (dilated)--> hysteresis threshold (data-driven sigma)
             --> remove small objects --> skeletonize
             --> cv2.HoughLinesP (local segments) + skimage hough_line (global lines)
             --> slope / direction / local-orientation consistency filters

All segments are returned in pixel coordinates; Stage 7 (``grouping``) converts them to
physical units, groups the candidates of one vehicle and averages them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold, threshold_otsu
from skimage.measure import LineModelND, label, ransac, regionprops
from skimage.morphology import remove_small_objects, skeletonize
from skimage.transform import hough_line, hough_line_peaks

from .config import DetectionParams

MAD_TO_SIGMA = 1.4826


# ======================================================================================
# Data structures
# ======================================================================================
@dataclass
class LineSegment:
    """Line segment in image pixel coordinates (col = space, row = time).

    Endpoints are ordered in time (``r0 <= r1``). ``score`` is the mean response along
    the segment (robust z), ``thickness_px`` the mean temporal width of the supporting
    component in rows (strong path only).
    """
    c0: float
    r0: float
    c1: float
    r1: float
    score: float
    source: str            # "strong" | "weak" (HoughLinesP) | "hough_global"
    direction: int         # +1 / -1
    band: str = ""         # "hf" | "lf"
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
    """All line candidates of one image (all bands and directions) + intermediate maps.

    ``stages`` keys: ``{band}_{pos|neg}_{name}`` with names ``strong_binary``,
    ``strong_opened``, ``strong_mask``, ``weak_input``, ``weak_hysteresis``,
    ``weak_skeleton``. ``thresholds`` holds the data-driven thresholds per map.
    """
    segments: list[LineSegment] = field(default_factory=list)
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)


# ======================================================================================
# Helpers
# ======================================================================================
def robust_noise_sigma(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """1.4826 * MAD of the image (or of the pixels where ``mask`` is True).

    Basis for all data-driven thresholds: unlike the std it is not inflated by the lines
    themselves (heavy-tailed data, EDA). Returns 1.0 if the MAD is zero.
    """
    v = np.asarray(img, dtype=np.float64)
    v = v[mask] if mask is not None else v.ravel()
    if v.size == 0:
        return 1.0
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


def sample_image(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Nearest-pixel values of ``img`` at (N, 2) [col, row] points (clipped to the image)."""
    c = np.clip(np.rint(pts[:, 0]).astype(int), 0, img.shape[1] - 1)
    r = np.clip(np.rint(pts[:, 1]).astype(int), 0, img.shape[0] - 1)
    return img[r, c]


def segment_score(img: np.ndarray, seg: LineSegment) -> float:
    """Mean response along the segment."""
    return float(np.mean(sample_image(img, seg.points())))


def line_structuring_elements(length: int, slopes: np.ndarray) -> list[np.ndarray]:
    """Binary line structuring elements (uint8) of ``length`` px along each slope [col/row]."""
    out = []
    for s in np.atleast_1d(slopes):
        a = np.arctan(s)                                   # angle from the vertical (time) axis
        hr = int(np.ceil(0.5 * (length - 1) * abs(np.cos(a))))
        hc = int(np.ceil(0.5 * (length - 1) * abs(np.sin(a))))
        se = np.zeros((2 * hr + 1, 2 * hc + 1), dtype=np.uint8)
        dr, dc = 0.5 * (length - 1) * np.cos(a), 0.5 * (length - 1) * np.sin(a)
        p0 = (int(round(hc - dc)), int(round(hr - dr)))   # cv2 points are (x = col, y = row)
        p1 = (int(round(hc + dc)), int(round(hr + dr)))
        cv2.line(se, p0, p1, 1, 1)
        out.append(se)
    return out


def oriented_opening(mask: np.ndarray, length: int, slopes: np.ndarray) -> np.ndarray:
    """Union of binary openings with line elements of all admissible slopes.

    Keeps pixels that belong to a run of at least ``length`` px along *some* admissible
    direction: removes specks and structures oriented like noise (vertical / horizontal
    streaks), preserves elongated lines.
    """
    src = mask.astype(np.uint8)
    out = np.zeros_like(src)
    for se in line_structuring_elements(length, slopes):
        out |= cv2.morphologyEx(src, cv2.MORPH_OPEN, se)
    return out.astype(bool)


def ridge_along_time(response: np.ndarray, mask: np.ndarray, nms_rows: int = 1) -> np.ndarray:
    """Pixels of ``mask`` that are maxima of the response within +-``nms_rows`` along time.

    Non-maximum suppression with a vertical max filter (``cv2.dilate``, vectorised). A
    slanted line gives one ridge pixel per column (its centre) even when noise makes its
    flat top bumpy; two parallel lines more than ``nms_rows`` apart that touch in the
    binary mask still give two separate ridges.
    """
    r = np.asarray(response, dtype=np.float32)
    se = np.ones((2 * max(int(nms_rows), 1) + 1, 1), dtype=np.uint8)
    local_max = cv2.dilate(r, se, borderType=cv2.BORDER_REPLICATE)
    return mask & (r >= local_max)


def fit_line_ransac(cols: np.ndarray, rows: np.ndarray, residual_px: float, max_trials: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC (skimage ``LineModelND``) on [col, row] points -> (origin, unit direction, inliers)."""
    data = np.column_stack([cols, rows]).astype(np.float64)
    if len(data) < 3:
        d = data[-1] - data[0]
        return data[0], d / max(np.linalg.norm(d), 1e-12), np.ones(len(data), bool)
    model, inliers = ransac(data, LineModelND, min_samples=2, residual_threshold=residual_px,
                            max_trials=max_trials, rng=seed)
    if model is None or inliers is None or inliers.sum() < 2:
        inliers = np.ones(len(data), bool)
    # least-squares refit on the inliers (origin = centroid, direction = principal axis)
    pts = data[inliers]
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
    return origin, vt[0], inliers


def sequential_ransac(cols: np.ndarray, rows: np.ndarray, params: DetectionParams,
                      min_cols: float, bounds: tuple[float, float]
                      ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Several lines in one point set: RANSAC, remove the inliers, repeat.

    Needed when two thick lines (e.g. two trams close behind each other, or an overtaking
    truck) touch and form one connected component. A line is kept when its inliers span
    at least ``min_cols`` columns and its slope is admissible; then all points closer than
    ``strong_ridge_nms_rows`` (along time) to it are removed, so leftovers of the same
    thick line cannot start another line. A later line that runs alongside no accepted line
    (column-range overlap below ``strong_parallel_overlap``) is not a second vehicle - two
    vehicles of one direction both cross the cable, so they overlap in x - but the bent
    continuation of a line (e.g. near the cable ends), so it only extends that line.
    The loop stops after ``strong_max_lines`` lines or when too few points remain.
    Returns a list of (origin, direction, inlier indices into the input arrays).
    """
    left = np.arange(cols.size)
    out = []
    for k in range(params.strong_max_lines * 2):
        if left.size < max(min_cols / 2, 3) or len(out) >= params.strong_max_lines:
            break
        origin, d, inl = fit_line_ransac(cols[left], rows[left], params.ransac_residual_px,
                                         params.ransac_max_trials, params.seed + k)
        idx = left[inl]
        lo, hi = cols[idx].min(), cols[idx].max()
        # overlap (fraction of the new line's column range) with every accepted line
        overlap = [(min(hi, cols[j].max()) - max(lo, cols[j].min())) / max(hi - lo, 1.0)
                   for _, _, j in out]
        if out and max(overlap) < params.strong_parallel_overlap:
            # does not run alongside any accepted line -> it is the (bent) continuation of
            # the nearest one (one vehicle), so it only extends that line's extent
            k = int(np.argmax(overlap))
            out[k] = (out[k][0], out[k][1], np.union1d(out[k][2], idx))
        elif (abs(d[0]) > 1e-9 and abs(d[1]) > 1e-9 and hi - lo + 1 >= min_cols
              and slope_ok(d[0] / d[1], bounds, params.slope_margin)):
            out.append((origin, d, idx))
            r_line = origin[1] + (cols[left] - origin[0]) * d[1] / d[0]
            inl = inl | (np.abs(rows[left] - r_line) <= params.strong_ridge_nms_rows)
        left = left[~inl]
    return out


# ======================================================================================
# 6A - strong lines
# ======================================================================================
def strong_threshold(response: np.ndarray, params: DetectionParams) -> float:
    """Otsu threshold of the positive response, floored at ``strong_min_sigma`` robust sigma.

    The floor matters in a direction without heavy traffic: Otsu then separates only
    noise texture from slightly stronger noise texture.
    """
    med = float(np.median(response))
    sigma = robust_noise_sigma(response)
    floor = med + params.strong_min_sigma * sigma
    if params.strong_threshold == "otsu":
        pos = response[response > med]
        thr = float(threshold_otsu(pos)) if pos.size and np.ptp(pos) > 0 else floor
    else:
        thr = med + float(params.strong_threshold) * sigma
    return max(thr, floor)


def detect_strong_lines(response: np.ndarray, params: DetectionParams, row_dt: float,
                        col_dx: float, direction: int, v_range_kmh: tuple[float, float],
                        band: str = "hf") -> tuple[list[LineSegment], np.ndarray, dict]:
    """Path 6A: threshold -> oriented opening -> components -> ridge -> sequential RANSAC.

    Components smaller than ``strong_min_area_px`` or spanning less than ``min_extent_m``
    along the cable are rejected. In each remaining component the ridge (local maxima of
    the response along time) is fitted with sequential RANSAC, so touching parallel thick
    lines give separate lines. Each component pixel is assigned to the nearest fitted
    line (in rows); the line thickness is the mean number of assigned pixels per column.
    Returns (segments, strong mask, info with stages and the threshold).
    """
    bounds = slope_bounds(v_range_kmh, direction, row_dt, col_dx)
    thr = strong_threshold(response, params)
    binary = response > thr
    slopes = np.linspace(bounds[0], bounds[1], params.strong_open_angles)
    opened = oriented_opening(binary, params.strong_open_length_px, slopes)
    ridge = ridge_along_time(response, opened, params.strong_ridge_nms_rows)
    lab = label(opened, connectivity=2)
    mask = np.zeros(response.shape, dtype=bool)
    segments = []
    min_cols = params.min_extent_m / col_dx
    for reg in regionprops(lab, intensity_image=response):
        rr, cc = reg.coords[:, 0], reg.coords[:, 1]
        if reg.area < params.strong_min_area_px or np.ptp(cc) + 1 < min_cols:
            continue
        on_ridge = ridge[rr, cc]
        lines = sequential_ransac(cc[on_ridge], rr[on_ridge], params, min_cols, bounds)
        if not lines:
            continue
        # assign every pixel of the component to the nearest line (vertical distance);
        # row on line k at each pixel's column: r = o_r + (c - o_c) / slope
        r_line = np.stack([o[1] + (cc - o[0]) * (d[1] / d[0] if abs(d[0]) > 1e-9 else 0.0)
                           for o, d, _ in lines])
        nearest = np.argmin(np.abs(rr[None, :] - r_line), axis=0)
        rcols, rrows = cc[on_ridge], rr[on_ridge]
        for k, (origin, d, idx) in enumerate(lines):
            slope = d[0] / d[1]
            cmin, cmax = float(rcols[idx].min()), float(rcols[idx].max())
            r_at = lambda c: origin[1] + (c - origin[0]) / slope  # noqa: E731
            own = nearest == k
            n_cols = max(np.unique(cc[own]).size, 1)
            seg = LineSegment(cmin, r_at(cmin), cmax, r_at(cmax),
                              score=float(np.mean(response[rrows[idx], rcols[idx]])),
                              source="strong", direction=direction, band=band,
                              thickness_px=float(own.sum() / n_cols))
            segments.append(seg)
        mask[rr, cc] = True
    stages = {"strong_binary": binary, "strong_opened": opened, "strong_mask": mask,
              "strong_ridge": ridge}
    return segments, mask, {"stages": stages, "threshold": thr}


# ======================================================================================
# 6B - weak lines
# ======================================================================================
def mask_strong(response: np.ndarray, strong_mask: np.ndarray, dilation_px: tuple[int, int]
                ) -> tuple[np.ndarray, np.ndarray]:
    """Set the response under the (dilated) strong mask to the background median.

    ``dilation_px`` = (rows, cols) half-sizes of the elliptical dilation element, so the
    halo of a tram along time and space is also excluded. Returns (masked, excluded).
    """
    hr, hc = dilation_px
    if strong_mask.any() and (hr > 0 or hc > 0):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hc + 1, 2 * hr + 1))
        excl = cv2.dilate(strong_mask.astype(np.uint8), se).astype(bool)
    else:
        excl = strong_mask.astype(bool)
    out = np.array(response, dtype=np.float64)
    out[excl] = np.median(out[~excl]) if (~excl).any() else 0.0
    return out, excl


def local_orientation_ok(seg: LineSegment, angle_slopes: np.ndarray | None,
                         angle_index: np.ndarray | None, tol_deg: float) -> bool:
    """Does the segment agree with the local orientation found by the filter bank?

    ``angle_slopes[angle_index]`` is the slope of the strongest kernel at every pixel
    (Stage 5). The median local angle along the segment must be within ``tol_deg`` of
    the segment angle - this rejects Hough segments that cut across several lines.
    """
    if angle_slopes is None or angle_index is None or tol_deg <= 0:
        return True
    local = angle_slopes[sample_image(angle_index, seg.points())]
    a_local = np.degrees(np.arctan(np.median(local)))
    return abs(a_local - np.degrees(np.arctan(seg.slope))) <= tol_deg


def hough_global(skeleton: np.ndarray, response: np.ndarray, bounds: tuple[float, float],
                 direction: int, params: DetectionParams, band: str = "") -> list[LineSegment]:
    """Complementary global path: skimage ``hough_line`` + ``hough_line_peaks``.

    Votes are accumulated along whole lines (only angles of the admissible slopes), which
    helps for long, straight, very weak lines that are fragmented in the skeleton. Each
    peak is turned into segments using the skeleton pixels within ``global_support_px``
    of the line, split at gaps longer than ``global_max_gap_px`` rows. The gap can be
    larger than in HoughLinesP because the pieces are collinear by construction, so a
    fragmented thin line becomes one long segment.
    """
    # skimage: col*cos(theta) + row*sin(theta) = rho; a line with slope s = dcol/drow
    # has its normal at theta = -atan(s)
    step = np.radians(params.hough_theta_deg)
    thetas = np.arange(-np.arctan(bounds[1]), -np.arctan(bounds[0]) + step / 2, step)
    if thetas.size == 0 or not skeleton.any():
        return []
    hspace, angles, dists = hough_line(skeleton, theta=thetas)
    _, th, rho = hough_line_peaks(hspace, angles, dists, min_distance=params.global_min_distance,
                                  min_angle=params.global_min_angle,
                                  threshold=params.global_min_votes, num_peaks=params.global_max_peaks)
    rr, cc = np.nonzero(skeleton)
    out = []
    for t, p in zip(th, rho):
        dist = np.abs(cc * np.cos(t) + rr * np.sin(t) - p)
        near = dist <= params.global_support_px
        if near.sum() < 2:
            continue
        r_sup = np.sort(rr[near])
        # split the support into runs separated by gaps (along time)
        breaks = np.nonzero(np.diff(r_sup) > params.global_max_gap_px)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, r_sup.size - 1]
        for s, e in zip(starts, ends):
            r0, r1 = float(r_sup[s]), float(r_sup[e])
            # column on the line at a given row: c = (rho - r sin) / cos
            c0, c1 = (p - r0 * np.sin(t)) / np.cos(t), (p - r1 * np.sin(t)) / np.cos(t)
            seg = LineSegment(c0, r0, c1, r1, 0.0, "hough_global", direction, band)
            if seg.length_px < params.hough_min_line_length:
                continue
            seg.score = segment_score(response, seg)
            out.append(seg)
    return out


def detect_weak_lines(response: np.ndarray, strong_mask: np.ndarray, params: DetectionParams,
                      row_dt: float, col_dx: float, direction: int,
                      v_range_kmh: tuple[float, float], band: str = "",
                      angle_slopes: np.ndarray | None = None,
                      angle_index: np.ndarray | None = None) -> tuple[list[LineSegment], dict]:
    """Path 6B: mask strong objects -> hysteresis -> skeleton -> HoughLinesP + global Hough.

    Thresholds are relative to the robust noise level of the background (pixels outside
    the dilated strong mask): low = median + ``weak_low_sigma`` sigma, high = median +
    ``weak_high_sigma`` sigma. Hysteresis keeps weak pixels only when they are connected
    to a strong seed, so a faint but continuous ridge survives while isolated noise does
    not. Segments are kept when the slope is admissible, the local orientation from the
    filter bank agrees and the mean response is above the low threshold.
    Returns (segments, info with stages and thresholds).
    """
    bounds = slope_bounds(v_range_kmh, direction, row_dt, col_dx)
    masked, excl = mask_strong(response, strong_mask, (params.strong_mask_dilation_rows,
                                                       params.strong_mask_dilation_cols))
    med = float(np.median(masked[~excl])) if (~excl).any() else 0.0
    sigma = robust_noise_sigma(masked, ~excl)
    low, high = med + params.weak_low_sigma * sigma, med + params.weak_high_sigma * sigma
    hyst = apply_hysteresis_threshold(masked, low, high)
    hyst = remove_small_objects(hyst, max_size=params.min_area_px)
    skel = skeletonize(hyst)

    segments = []
    if params.use_hough_p and skel.any():
        lines = cv2.HoughLinesP(skel.astype(np.uint8) * 255, rho=1,
                                theta=np.radians(params.hough_theta_deg),
                                threshold=params.hough_threshold,
                                minLineLength=params.hough_min_line_length,
                                maxLineGap=params.hough_max_line_gap)
        for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):
            segments.append(LineSegment(float(x1), float(y1), float(x2), float(y2), 0.0,
                                        "weak", direction, band))
    if params.use_hough_global:
        segments += hough_global(skel, response, bounds, direction, params, band)

    kept = []
    for seg in segments:
        if seg.r1 - seg.r0 < 1 or not slope_ok(seg.slope, bounds, params.slope_margin):
            continue
        if not local_orientation_ok(seg, angle_slopes, angle_index, params.orientation_tol_deg):
            continue
        seg.score = segment_score(response, seg)
        if seg.score < low:
            continue
        kept.append(seg)
    info = {"stages": {"weak_input": masked, "weak_hysteresis": hyst, "weak_skeleton": skel},
            "threshold_low": low, "threshold_high": high, "sigma": sigma}
    return kept, info


# ======================================================================================
# Both paths on all maps of one image
# ======================================================================================
def detect_candidates(enhanced: dict, params: DetectionParams, v_range_kmh: tuple[float, float]
                      ) -> Detections:
    """Run 6A and 6B on every band / direction map of a segment.

    ``enhanced`` maps a band name to an ``enhancement.Enhanced`` object (same image grid
    for all bands). 6A runs on ``params.strong_bands``; the union of their strong masks
    (per direction) is excluded in 6B, which runs on ``params.weak_bands``.
    """
    out = Detections()
    any_enh = next(iter(enhanced.values()))
    row_dt, col_dx = any_enh.row_dt, any_enh.col_dx
    for d, dname in ((+1, "pos"), (-1, "neg")):
        strong_mask = np.zeros(any_enh.response[d].shape, dtype=bool)
        for band in params.strong_bands:
            segs, mask, info = detect_strong_lines(enhanced[band].response[d], params, row_dt,
                                                   col_dx, d, v_range_kmh, band)
            out.segments += segs
            strong_mask |= mask
            out.stages.update({f"{band}_{dname}_{k}": v for k, v in info["stages"].items()})
            out.thresholds[f"{band}_{dname}"] = {"strong": info["threshold"]}
        for band in params.weak_bands:
            e = enhanced[band]
            slopes = velocity_to_slope(e.velocities[d], row_dt, col_dx) if d in e.velocities else None
            segs, info = detect_weak_lines(
                e.response[d], strong_mask if params.mask_strong else np.zeros_like(strong_mask),
                params, row_dt, col_dx, d, v_range_kmh, band, slopes, e.angle_index.get(d))
            out.segments += segs
            out.stages.update({f"{band}_{dname}_{k}": v for k, v in info["stages"].items()})
            out.thresholds.setdefault(f"{band}_{dname}", {}).update(
                weak_low=info["threshold_low"], weak_high=info["threshold_high"],
                weak_sigma=info["sigma"])
    return out
