"""Quantitative evaluation: synthetic ground truth and manual real-data labels (plan: Stage 11).

Pixel-level helpers (``gt_envelope_mask``, ``pixel_auc``, ``enhancement_auc``) are used
already in Stages 4-5 to compare preprocessing / enhancement variants on synthetic data.
"""
from __future__ import annotations

import cv2
import numpy as np

from .features import object_trajectory
from .grouping import DetectedObject
from .tracking import speed_variability, velocity_profile
from .velocity import reported_velocity_kmh

HEAVY_TYPES = ("tram", "truck", "thick")   # thick lines (synthetic types + manual label)
LIGHT_TYPES = ("car", "van", "thin")       # thin lines


def vehicle_class(vtype: str) -> str:
    """"heavy" (thick line), "light" (thin line) or "unknown"."""
    return "heavy" if vtype in HEAVY_TYPES else "light" if vtype in LIGHT_TYPES else "unknown"


def predicted_class(obj: DetectedObject) -> str:
    """Class predicted by the detector: the strong path (6A) finds thick lines."""
    return "heavy" if obj.kind == "strong" else "light"


# ======================================================================================
# Pixel-level evaluation of enhanced images (Stages 4-5)
# ======================================================================================
def gt_envelope_mask(gt: np.ndarray, factor: int, markers: list[int] | None = None,
                     upsample: int = 1) -> np.ndarray:
    """Ground-truth marker mask at the image resolution.

    (T, C) marker array -> boolean (T // factor, C * upsample): a cell is True if any of
    its ``factor`` samples carries one of ``markers`` (all vehicles if None).
    """
    n = (gt.shape[0] // factor) * factor
    sel = gt[:n] > 0 if markers is None else np.isin(gt[:n], markers)
    m = sel.reshape(n // factor, factor, gt.shape[1]).any(axis=1)
    return np.repeat(m, upsample, axis=1) if upsample > 1 else m


def pixel_auc(score: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    """Area under the ROC curve of ``score`` for positive vs negative pixels.

    Mann-Whitney U statistic computed from ranks (vectorised, ties broken by order -
    scores are continuous). 0.5 = no separation, 1.0 = perfect.
    """
    pos, neg = score[positive], score[negative]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.empty(pos.size + neg.size)
    ranks[np.argsort(np.concatenate([pos, neg]), kind="stable")] = np.arange(1, ranks.size + 1)
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def auc_score(score: np.ndarray, positive: np.ndarray) -> float:
    """ROC AUC of ``score`` for a boolean ``positive`` labelling, with tied scores given
    their average rank (needed for discrete features such as counts)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    end = np.cumsum(cnt)
    mid = (end - cnt + 1 + end) / 2.0                 # average rank of every tied group
    ranks = mid[inv]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def enhancement_auc(score: np.ndarray, gt: np.ndarray, meta: dict, factor: int,
                    upsample: int = 1, guard_rows: int = 25, guard_cols: int = 2) -> dict:
    """AUC of an enhanced image against the synthetic ground truth, per vehicle class.

    Positives: pixels on the trajectories of the class (heavy = tram/truck, light =
    car/van, all). Negatives: pixels farther than ``guard_rows`` rows / ``guard_cols``
    channels from any vehicle (so the halo of a line is not counted as background).
    """
    types = {v["marker_value"]: v["type"] for v in meta["ground_truth"]["vehicles"]}
    any_v = gt_envelope_mask(gt, factor, upsample=1)
    se = np.ones((2 * guard_rows + 1, 2 * guard_cols + 1), np.uint8)
    near = cv2.dilate(any_v.astype(np.uint8), se).astype(bool)
    negative = np.repeat(~near, upsample, axis=1) if upsample > 1 else ~near
    out = {}
    for name, sel in (("heavy", HEAVY_TYPES), ("light", LIGHT_TYPES), ("all", None)):
        markers = None if sel is None else [m for m, t in types.items() if t in sel]
        if markers == []:
            out[name] = float("nan")
            continue
        out[name] = pixel_auc(score, gt_envelope_mask(gt, factor, markers, upsample), negative)
    return out


# ======================================================================================
# Object-level evaluation (used from Stage 6 on; full study in Stage 11)
# ======================================================================================
def gt_objects_from_mask(gt: np.ndarray, meta: dict, dt: float, dx: float) -> list[dict]:
    """Reference objects from the ground-truth marker mask (not from the metadata list).

    For every marker present in the mask: the passage time in every channel (mean time of
    the marked samples) gives a track (x, t); a least-squares line t = t_c + p (x - x_c)
    gives the mean velocity 1/p. Type and direction come from the metadata (sample 43 has
    a vehicle listed in the metadata but missing from the mask - it is skipped).
    Physical convention as in ``grouping``: x = (channel + 0.5) dx, t = (sample + 0.5) dt.
    Returns dicts with marker, type, direction, velocity_kmh, x_min/x_max, t_min/t_max,
    x_c, t_c, slowness and track (N, 2) [x_m, t_s].
    """
    info = {v["marker_value"]: v for v in meta["ground_truth"]["vehicles"]}
    rows, cols = np.nonzero(gt)
    vals = gt[rows, cols]
    out = []
    for m in np.unique(vals):
        sel = vals == m
        r, c = rows[sel], cols[sel]
        ch = np.unique(c)
        idx = np.searchsorted(ch, c)
        t_ch = (np.bincount(idx, weights=r) / np.bincount(idx) + 0.5) * dt
        x_ch = (ch + 0.5) * dx
        if ch.size >= 2:
            xc, tc = x_ch.mean(), t_ch.mean()
            p = float(np.sum((x_ch - xc) * (t_ch - tc)) / np.sum((x_ch - xc) ** 2))
        else:
            xc, tc, p = float(x_ch[0]), float(t_ch[0]), float("inf")
        v = info.get(int(m), {})
        out.append({"marker": int(m), "type": v.get("type", "unknown"),
                    "direction": int(v.get("direction", np.sign(p) if np.isfinite(p) else 0)),
                    "velocity_kmh": 3.6 / p if p not in (0.0, float("inf")) else float("nan"),
                    "slowness": p, "x_c": float(xc), "t_c": float(tc),
                    "x_min": float(x_ch.min()), "x_max": float(x_ch.max()),
                    "t_min": float(t_ch.min()), "t_max": float(t_ch.max()),
                    "track": np.column_stack([x_ch, t_ch])})
    return out


def _reference_time_at(ref: dict, x: float) -> float:
    """Passage time of a reference object at x (interpolated track, extrapolated line)."""
    tr = ref["track"]
    if tr[0, 0] <= x <= tr[-1, 0]:
        return float(np.interp(x, tr[:, 0], tr[:, 1]))
    return float(ref["t_c"] + ref["slowness"] * (x - ref["x_c"]))


def match_detections(detections: list[DetectedObject], references: list[dict],
                     tol_t_s: float, tol_v_kmh: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by normalised cost |dt| / tol_t + |dv| / tol_v.

    ``dt`` is the time difference between the detection and the reference track at the
    centre of the detection (where the detection is reliable); ``dv`` the difference of
    signed velocities. Pairs need the same direction, |dt| < tol_t and |dv| < tol_v
    (gating). Returns (detection index, reference index) pairs.
    """
    cand = []
    for i, d in enumerate(detections):
        xm = 0.5 * (d.x_min + d.x_max)
        for j, r in enumerate(references):
            if np.sign(d.velocity_kmh) != r["direction"]:
                continue
            dt = abs(float(d.t_at(xm)) - _reference_time_at(r, xm))
            dv = abs(d.velocity_kmh - r["velocity_kmh"])
            if dt < tol_t_s and dv < tol_v_kmh:
                cand.append((dt / tol_t_s + dv / tol_v_kmh, i, j))
    used_d, used_r, out = set(), set(), []
    for _, i, j in sorted(cand):
        if i not in used_d and j not in used_r:
            used_d.add(i)
            used_r.add(j)
            out.append((i, j))
    return out


def detection_metrics(n_det: int, n_ref: int, matches: list[tuple[int, int]]) -> dict:
    """TP, FP, FN, precision, recall, F1."""
    tp = len(matches)
    fp, fn = n_det - tp, n_ref - tp
    prec = tp / n_det if n_det else float("nan")
    rec = tp / n_ref if n_ref else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if n_det and n_ref and (prec + rec) > 0 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def velocity_errors(detections: list[DetectedObject], references: list[dict],
                    matches: list[tuple[int, int]], reported: bool = False) -> dict:
    """MAE, RMSE [km/h] and MAPE [%] of |velocity| over matched pairs.

    ``reported`` False: the Stage-7 line velocity; True: the reported velocity (Stage-8
    track fit, line velocity as fallback).
    """
    if not matches:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0}
    vel = reported_velocity_kmh if reported else (lambda o: o.velocity_kmh)
    est = np.array([abs(vel(detections[i])) for i, _ in matches])
    ref = np.array([abs(references[j]["velocity_kmh"]) for _, j in matches])
    err = est - ref
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": float(100 * np.mean(np.abs(err) / ref)), "n": len(matches)}


def profile_errors(obj: DetectedObject, ref: dict, params, ref_window: int = 3) -> dict:
    """Velocity-over-time error of a tracked object against its ground-truth track.

    The reference v(t) is computed from the GT track (exact passage time per channel)
    with :func:`tracking.velocity_profile` and a short window (``ref_window`` channels,
    close to the instantaneous velocity), so the error includes both the measurement noise
    and the smoothing bias of the detection window. Errors on the common time span:

    - ``vt_mae_kmh``: mean |v_det(t) - v_gt(t)| of |v| (interpolated at the object times);
    - ``accel_err_ms2``: |a_det - a_gt| of the quadratic-fit accelerations;
    - ``class_gt`` / ``class_det``: constant / varying speed labels;
    - ``coverage``: fraction of the GT track channels covered by the object track.
    Returns NaNs / "" when the object has no track.
    """
    out = {"vt_mae_kmh": float("nan"), "accel_err_ms2": float("nan"), "accel_gt_ms2": float("nan"),
           "class_gt": "", "class_det": obj.speed_class, "coverage": 0.0}
    tr = ref["track"]
    tg, vg = velocity_profile(tr, ref_window)
    sv = speed_variability(tr, vg, params)
    out["class_gt"], out["accel_gt_ms2"] = sv["speed_class"], sv["accel_ms2"]
    no_track = obj.track is None or len(obj.track) == 0
    if no_track or obj.profile is None or len(obj.profile) == 0 or tg.size < 2:
        return out
    td, vd = obj.profile[:, 0], obj.profile[:, 2]
    common = (td >= tg.min()) & (td <= tg.max()) & np.isfinite(vd)
    if common.any():
        out["vt_mae_kmh"] = float(np.mean(np.abs(np.abs(vd[common]) - np.abs(np.interp(td[common], tg, vg)))))
    out["accel_err_ms2"] = float(abs(obj.accel_ms2 - sv["accel_ms2"]))
    x_det = obj.track[:, 0]
    out["coverage"] = float(np.mean((tr[:, 0] >= x_det.min() - 1e-6) & (tr[:, 0] <= x_det.max() + 1e-6)))
    return out


# ======================================================================================
# Stage 11: per-class metrics, trajectory overlap, manual references
# ======================================================================================
def class_metrics(detections: list[DetectedObject], references: list[dict],
                  matches: list[tuple[int, int]]) -> dict:
    """Precision / recall / F1 separately for thick (heavy) and thin (light) lines.

    Matching is class-agnostic (one vehicle = one object). Recall of a class counts its
    reference objects that were matched; precision of a class counts the detections
    *predicted* as that class (strong path = thick, weak path = thin) that were matched to
    any vehicle. The confusion of the matched pairs shows how well the two paths separate
    thick and thin lines.
    """
    ref_cls = [vehicle_class(r["type"]) for r in references]
    det_cls = [predicted_class(d) for d in detections]
    m_ref = {j for _, j in matches}
    m_det = {i for i, _ in matches}
    out = {}
    for c in ("heavy", "light"):
        n_ref = sum(rc == c for rc in ref_cls)
        n_det = sum(dc == c for dc in det_cls)
        tp_r = sum(rc == c and j in m_ref for j, rc in enumerate(ref_cls))
        tp_d = sum(dc == c and i in m_det for i, dc in enumerate(det_cls))
        rec = tp_r / n_ref if n_ref else float("nan")
        prec = tp_d / n_det if n_det else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if n_ref and n_det and prec + rec > 0 else float("nan")
        out.update({f"n_ref_{c}": n_ref, f"tp_{c}": tp_r, f"n_det_{c}": n_det,
                    f"tp_det_{c}": tp_d, f"recall_{c}": rec, f"precision_{c}": prec,
                    f"f1_{c}": f1})
    for rc in ("heavy", "light"):
        for dc in ("heavy", "light"):
            out[f"conf_{rc}_as_{dc}"] = sum(ref_cls[j] == rc and det_cls[i] == dc
                                            for i, j in matches)
    return out


def rasterize_tracks(tracks: list[np.ndarray], shape: tuple[int, int], row_dt: float,
                     dx: float) -> np.ndarray:
    """Boolean (rows, channels) image with one pixel per channel on every trajectory.

    Each track (N, 2) [x_m, t_s] is interpolated at all channel centres between its
    first and last point (x = (c + 0.5) dx, t = (r + 0.5) row_dt).
    """
    mask = np.zeros(shape, bool)
    for tr in tracks:                                   # loop over objects
        if tr is None or len(tr) == 0:
            continue
        tr = tr[np.argsort(tr[:, 0])]
        c0 = int(np.ceil(tr[0, 0] / dx - 0.5))
        c1 = int(np.floor(tr[-1, 0] / dx - 0.5))
        cols = np.arange(max(c0, 0), min(c1, shape[1] - 1) + 1)
        if cols.size == 0:
            cols = np.clip(np.rint(tr[:, 0] / dx - 0.5).astype(int), 0, shape[1] - 1)
        t = np.interp((cols + 0.5) * dx, tr[:, 0], tr[:, 1])
        rows = np.rint(t / row_dt - 0.5).astype(int)
        ok = (rows >= 0) & (rows < shape[0])
        mask[rows[ok], cols[ok]] = True
    return mask


def detection_tracks(detections: list[DetectedObject], n_channels: int, dx: float
                     ) -> list[np.ndarray]:
    """(x, t) trajectory of every detection: the Stage-9 track, else the averaged line."""
    out = []
    for d in detections:
        cols, t = object_trajectory(d, n_channels, dx)
        out.append(np.column_stack([(cols + 0.5) * dx, t]))
    return out


def trajectory_iou(det_mask: np.ndarray, gt_mask: np.ndarray, tol_px: int = 3) -> float:
    """Intersection over union of two trajectory images with a tolerance along time.

    Both masks are dilated by +-``tol_px`` rows (vertical element - a trajectory has one
    pixel per channel) and IoU = |D' & G'| / |D' | G'|. 1 = all trajectories found exactly
    (within the tolerance); missed, false or shifted trajectory parts lower it.
    """
    inter, union = trajectory_overlap(det_mask, gt_mask, tol_px)
    return inter / union if union else float("nan")


def trajectory_overlap(det_mask: np.ndarray, gt_mask: np.ndarray, tol_px: int = 3
                       ) -> tuple[int, int]:
    """(|D' & G'|, |D' | G'|) of the dilated masks (see :func:`trajectory_iou`); the counts
    can be summed over recordings for a pooled IoU."""
    se = np.ones((2 * tol_px + 1, 1), np.uint8)
    d = cv2.dilate(det_mask.astype(np.uint8), se).astype(bool)
    g = cv2.dilate(gt_mask.astype(np.uint8), se).astype(bool)
    return int(np.count_nonzero(d & g)), int(np.count_nonzero(d | g))


def references_from_annotations(lines: list[dict]) -> list[dict]:
    """Reference objects from manual labels (``data/annotations/manual_labels.json``).

    Each line has ``points`` [[x_m, t_s], ...] (>= 2 clicks along the visible trajectory,
    t in seconds from the segment start) and ``kind`` "thick" / "thin". The track is the
    polyline sorted by x; the velocity is the least-squares slope through the points.
    Returns dicts with the keys of :func:`gt_objects_from_mask` (type = "thick" / "thin").
    """
    out = []
    for k, ln in enumerate(lines):
        pts = np.asarray(ln["points"], dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            continue
        pts = pts[np.argsort(pts[:, 0])]
        xc, tc = pts[:, 0].mean(), pts[:, 1].mean()
        sxx = np.sum((pts[:, 0] - xc) ** 2)
        p = float(np.sum((pts[:, 0] - xc) * (pts[:, 1] - tc)) / sxx) if sxx > 0 else float("inf")
        vel = 3.6 / p if np.isfinite(p) and p != 0 else float("nan")
        out.append({"marker": int(ln.get("id", k + 1)), "type": ln.get("kind", "unknown"),
                    "direction": int(np.sign(p)) if np.isfinite(p) else 0,
                    "velocity_kmh": vel, "slowness": p, "x_c": float(xc), "t_c": float(tc),
                    "x_min": float(pts[0, 0]), "x_max": float(pts[-1, 0]),
                    "t_min": float(pts[:, 1].min()), "t_max": float(pts[:, 1].max()),
                    "track": pts})
    return out


def split_references(references: list[dict], v_range_kmh: tuple[float, float],
                     min_extent_m: float) -> tuple[list[dict], list[dict]]:
    """Split manual references into evaluable ones and excluded ones.

    A label is excluded when its velocity (least-squares slope of the clicks) is outside
    ``v_range_kmh`` - the fan searched by the detector; a near-horizontal stroke is a
    horizontal noise stripe or an imprecise click, not a vehicle - or when it spans less
    than ``min_extent_m`` along the cable. Excluded labels get a ``reason``.
    """
    keep, excluded = [], []
    for r in references:
        v = abs(r["velocity_kmh"])
        if not np.isfinite(v) or not v_range_kmh[0] <= v <= v_range_kmh[1]:
            excluded.append({**r, "reason": "velocity outside the detector fan"})
        elif r["x_max"] - r["x_min"] < min_extent_m:
            excluded.append({**r, "reason": "too short"})
        else:
            keep.append(r)
    return keep, excluded


def false_positive_kinds(detections: list[DetectedObject], references: list[dict],
                         matches: list[tuple[int, int]], n_channels: int, dx: float,
                         near_s: float, min_channels: int) -> dict[int, str]:
    """Kind of every unmatched detection: "near_label" if its trajectory runs within
    ``near_s`` of some labelled line on >= ``min_channels`` channels (a fragment or a
    duplicate of a labelled vehicle, or a pair rejected by the gates), else "no_label"
    (noise, or a vehicle the annotator did not mark)."""
    matched = {i for i, _ in matches}
    out = {}
    for i, d in enumerate(detections):
        if i in matched:
            continue
        cols, t = object_trajectory(d, n_channels, dx)
        x = (cols + 0.5) * dx
        near = False
        for r in references:                            # loop over labels (few)
            tr = r["track"]
            inside = (x >= tr[0, 0]) & (x <= tr[-1, 0])
            if inside.sum() >= min_channels:
                gap = np.abs(np.interp(x[inside], tr[:, 0], tr[:, 1]) - t[inside])
                if np.sum(gap < near_s) >= min_channels:
                    near = True
                    break
        out[i] = "near_label" if near else "no_label"
    return out


def evaluate_objects(detections: list[DetectedObject], references: list[dict], params,
                     image_shape: tuple[int, int], row_dt: float, dx: float) -> dict:
    """All Stage-11 object metrics of one recording.

    ``params`` (``config.EvaluationParams``): matching gates and the IoU tolerance;
    ``image_shape`` (rows, channels) of the trajectory raster. Returns overall TP / FP /
    FN / P / R / F1, the per-class metrics of :func:`class_metrics`, errors of the reported
    velocity (all / heavy / light), the trajectory IoU and the matches.
    """
    m = match_detections(detections, references, params.tol_t_s, params.tol_v_kmh)
    out = detection_metrics(len(detections), len(references), m)
    out.update(class_metrics(detections, references, m))
    out.update({f"vel_{k}": v for k, v in velocity_errors(detections, references, m, True).items()})
    for c in ("heavy", "light"):
        mm = [(i, j) for i, j in m if vehicle_class(references[j]["type"]) == c]
        out[f"vel_mae_{c}"] = velocity_errors(detections, references, mm, True)["mae"]
    tol = max(int(round(params.iou_tol_s / row_dt)), 0)
    det_mask = rasterize_tracks(detection_tracks(detections, image_shape[1], dx), image_shape,
                                row_dt, dx)
    ref_mask = rasterize_tracks([r["track"] for r in references], image_shape, row_dt, dx)
    out["traj_inter"], out["traj_union"] = trajectory_overlap(det_mask, ref_mask, tol)
    out["traj_iou"] = out["traj_inter"] / out["traj_union"] if out["traj_union"] else float("nan")
    out["pairs"] = [{"det": i, "ref": j, "class": vehicle_class(references[j]["type"]),
                     "pred": predicted_class(detections[i]),
                     "v_ref": abs(references[j]["velocity_kmh"]),
                     "v_det": abs(reported_velocity_kmh(detections[i]))} for i, j in m]
    fpk = false_positive_kinds(detections, references, m, image_shape[1], dx,
                               params.fp_near_s, params.fp_min_channels)
    out["fp_near_label"] = sum(v == "near_label" for v in fpk.values())
    out["fp_no_label"] = sum(v == "no_label" for v in fpk.values())
    out["fp_kinds"] = fpk
    out["matches"] = m
    return out


def summarize_evaluations(evals: list[dict]) -> dict:
    """Pooled (micro-averaged) metrics over several recordings of :func:`evaluate_objects`.

    Counts are summed (TP, FP, FN per class), velocity errors are recomputed from all
    matched pairs and the trajectory IoU from the summed overlap counts.
    """
    def prf(tp_r, n_ref, tp_d, n_det):
        rec = tp_r / n_ref if n_ref else float("nan")
        prec = tp_d / n_det if n_det else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if n_ref and n_det and prec + rec > 0 else float("nan")
        return prec, rec, f1

    tot = {k: sum(e[k] for e in evals) for k in ("tp", "fp", "fn", "traj_inter", "traj_union")}
    for k in ("fp_near_label", "fp_no_label"):
        tot[k] = sum(e.get(k, 0) for e in evals)
    n_det, n_ref = tot["tp"] + tot["fp"], tot["tp"] + tot["fn"]
    tot["precision"], tot["recall"], tot["f1"] = prf(tot["tp"], n_ref, tot["tp"], n_det)
    for c in ("heavy", "light"):
        for k in (f"n_ref_{c}", f"tp_{c}", f"n_det_{c}", f"tp_det_{c}"):
            tot[k] = sum(e[k] for e in evals)
        tot[f"precision_{c}"], tot[f"recall_{c}"], tot[f"f1_{c}"] = prf(
            tot[f"tp_{c}"], tot[f"n_ref_{c}"], tot[f"tp_det_{c}"], tot[f"n_det_{c}"])
        for d in ("heavy", "light"):
            tot[f"conf_{c}_as_{d}"] = sum(e[f"conf_{c}_as_{d}"] for e in evals)
    pairs = [p for e in evals for p in e["pairs"]]
    for name, sel in (("", None), ("_heavy", "heavy"), ("_light", "light")):
        pp = [p for p in pairs if sel is None or p["class"] == sel]
        err = np.array([p["v_det"] - p["v_ref"] for p in pp])
        ref = np.array([p["v_ref"] for p in pp])
        tot[f"vel_mae{name}"] = float(np.mean(np.abs(err))) if pp else float("nan")
        tot[f"vel_rmse{name}"] = float(np.sqrt(np.mean(err ** 2))) if pp else float("nan")
        tot[f"vel_mape{name}"] = float(100 * np.mean(np.abs(err) / ref)) if pp else float("nan")
        tot[f"vel_n{name}"] = len(pp)
    tot["traj_iou"] = tot["traj_inter"] / tot["traj_union"] if tot["traj_union"] else float("nan")
    return tot
