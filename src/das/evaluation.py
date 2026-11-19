"""Quantitative evaluation of the detections (plan: Stage 9).

Two kinds of reference objects are compared with the detections:

- **synthetic data**: the ground-truth marker mask (``gt_objects_from_mask``), with the
  vehicle type from the metadata (tram / truck = thick line, car / van = thin line);
- **real data**: the lines marked by hand on the raw waterfall
  (``references_from_annotations``, label "thick" / "thin").

Grade-4 scoring (the goal is to detect the thick lines). Detections are matched
one-to-one with *all* reference objects (``match_detections``), then:

- ``recall_thick`` = matched thick references / thick references - the main metric;
- ``precision`` = matched detections / detections. A detection matched to a thin
  reference is a real vehicle (a loud car), so it is *not* a false positive;
- ``f1`` = harmonic mean of ``recall_thick`` and ``precision``;
- thin references that were found are reported as a bonus (``tp_thin``);
- velocity errors (|v| MAE, RMSE, MAPE) of the matched pairs, thick and all, and the
  mean time offset |dt| of the matched lines.
"""
from __future__ import annotations

import numpy as np

from .grouping import DetectedObject

HEAVY_TYPES = ("tram", "truck", "thick")   # thick lines (synthetic types + manual label)
LIGHT_TYPES = ("car", "van", "thin")       # thin lines


def vehicle_class(vtype: str) -> str:
    """"thick", "thin" or "unknown"."""
    return "thick" if vtype in HEAVY_TYPES else "thin" if vtype in LIGHT_TYPES else "unknown"


# ======================================================================================
# Reference objects
# ======================================================================================
def gt_objects_from_mask(gt: np.ndarray, meta: dict, dt: float, dx: float) -> list[dict]:
    """Reference objects from the synthetic ground-truth marker mask.

    For every marker present in the mask (not from the metadata list - sample 43 lists a
    vehicle that is missing from the mask): the passage time in every channel (mean time
    of the marked samples) gives a track (x, t), and a least-squares line
    t = t_c + p (x - x_c) gives the mean velocity 1/p. Type and direction come from the
    metadata. Physical convention as in ``grouping``: x = (channel + 0.5) dx,
    t = (sample + 0.5) dt. Returns dicts with marker, type, direction, velocity_kmh,
    slowness, x_c, t_c, x_min/x_max, t_min/t_max and track (N, 2) [x_m, t_s].
    """
    info = {v["marker_value"]: v for v in meta["ground_truth"]["vehicles"]}
    rows, cols = np.nonzero(gt)
    vals = gt[rows, cols]
    out = []
    for m in np.unique(vals):                          # loop over vehicles
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


def references_from_annotations(lines: list[dict]) -> list[dict]:
    """Reference objects from the manual labels (``data/annotations/manual_labels.json``).

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
    """Split manual references into scored and not scored ones.

    A label is not scored when its velocity (least-squares slope of the clicks) is
    outside ``v_range_kmh`` - the fan searched by the detector; a near-horizontal stroke
    is a horizontal noise stripe or an imprecise click, not a vehicle - or when it spans
    less than ``min_extent_m`` along the cable. Excluded labels get a ``reason``.
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


# ======================================================================================
# Matching
# ======================================================================================
def _reference_time_at(ref: dict, x: float) -> float:
    """Passage time of a reference object at x (interpolated track, extrapolated line)."""
    tr = ref["track"]
    if tr[0, 0] <= x <= tr[-1, 0]:
        return float(np.interp(x, tr[:, 0], tr[:, 1]))
    return float(ref["t_c"] + ref["slowness"] * (x - ref["x_c"]))


def _time_offset(det: DetectedObject, ref: dict) -> float:
    """|t_det - t_ref| at the centre of the detection (where the detection is reliable)."""
    xm = 0.5 * (det.x_min + det.x_max)
    return abs(float(det.t_at(xm)) - _reference_time_at(ref, xm))


def match_detections(detections: list[DetectedObject], references: list[dict],
                     tol_t_s: float, tol_v_kmh: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by the normalised cost |dt| / tol_t + |dv| / tol_v.

    ``dt`` is the time difference between the detection and the reference at the centre
    of the detection, ``dv`` the difference of the signed velocities. A pair needs the
    same direction, |dt| < tol_t and |dv| < tol_v. Returns (detection, reference) index pairs.
    """
    cand = []
    for i, d in enumerate(detections):
        for j, r in enumerate(references):
            if np.sign(d.velocity_kmh) != r["direction"]:
                continue
            dt = _time_offset(d, r)
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
    """TP, FP, FN, precision, recall, F1 of a one-to-one matching."""
    tp = len(matches)
    prec = tp / n_det if n_det else float("nan")
    rec = tp / n_ref if n_ref else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if n_det and n_ref and (prec + rec) > 0 else float("nan")
    return {"tp": tp, "fp": n_det - tp, "fn": n_ref - tp, "precision": prec, "recall": rec,
            "f1": f1}


def velocity_errors(v_det: np.ndarray, v_ref: np.ndarray) -> dict:
    """MAE, RMSE [km/h] and MAPE [%] of |velocity| over matched pairs."""
    est, ref = np.abs(np.asarray(v_det, float)), np.abs(np.asarray(v_ref, float))
    if est.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0}
    err = est - ref
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": float(100 * np.mean(np.abs(err) / ref)), "n": int(est.size)}


def line_trajectory(obj: DetectedObject, n_channels: int, dx: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(channel indices, passage times [s]) of the averaged line inside the object extent."""
    cols = np.arange(n_channels)
    x = (cols + 0.5) * dx
    inside = (x >= obj.x_min - 0.5 * dx) & (x <= obj.x_max + 0.5 * dx)
    return cols[inside], np.asarray(obj.t_at(x[inside]), float)


def false_positive_kinds(detections: list[DetectedObject], references: list[dict],
                         matches: list[tuple[int, int]], n_channels: int, dx: float,
                         near_s: float, min_channels: int) -> dict[int, str]:
    """Kind of every unmatched detection: "near_label" if its line runs within ``near_s``
    of some reference line on >= ``min_channels`` channels (a fragment or duplicate of a
    vehicle, or a pair rejected by the gates), else "no_label" (noise, or a vehicle the
    annotator did not mark)."""
    matched = {i for i, _ in matches}
    out = {}
    for i, d in enumerate(detections):
        if i in matched:
            continue
        cols, t = line_trajectory(d, n_channels, dx)
        x = (cols + 0.5) * dx
        near = False
        for r in references:                            # loop over references (few)
            tr = r["track"]
            inside = (x >= tr[0, 0]) & (x <= tr[-1, 0])
            if inside.sum() >= min_channels:
                gap = np.abs(np.interp(x[inside], tr[:, 0], tr[:, 1]) - t[inside])
                if np.sum(gap < near_s) >= min_channels:
                    near = True
                    break
        out[i] = "near_label" if near else "no_label"
    return out


# ======================================================================================
# All metrics of one recording, pooled metrics of several recordings
# ======================================================================================
def _prf(tp_thick: int, n_thick: int, tp_det: int, n_det: int) -> tuple[float, float, float]:
    """(precision, thick recall, F1 of the two)."""
    rec = tp_thick / n_thick if n_thick else float("nan")
    prec = tp_det / n_det if n_det else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if n_thick and n_det and prec + rec > 0 else float("nan")
    return prec, rec, f1


def evaluate_objects(detections: list[DetectedObject], references: list[dict], params,
                     n_channels: int, dx: float) -> dict:
    """Grade-4 metrics of one recording (see the module docstring).

    ``params`` is a ``config.EvaluationParams`` (matching gates, false-positive rule);
    ``n_channels`` / ``dx`` describe the cable (for the false-positive check).
    Returns the counts, precision / thick recall / F1, the bonus thin recall, velocity
    errors, the mean time offset, the false-positive split, the pairs and the matches.
    """
    m = match_detections(detections, references, params.tol_t_s, params.tol_v_kmh)
    cls = [vehicle_class(r["type"]) for r in references]
    out = {"n_det": len(detections), "n_ref": len(references),
           "n_ref_thick": cls.count("thick"), "n_ref_thin": cls.count("thin"),
           "tp": len(m), "fp": len(detections) - len(m),
           "tp_thick": sum(cls[j] == "thick" for _, j in m),
           "tp_thin": sum(cls[j] == "thin" for _, j in m)}
    out["fn_thick"] = out["n_ref_thick"] - out["tp_thick"]
    out["precision"], out["recall_thick"], out["f1"] = _prf(
        out["tp_thick"], out["n_ref_thick"], out["tp"], out["n_det"])
    out["recall_thin"] = out["tp_thin"] / out["n_ref_thin"] if out["n_ref_thin"] else float("nan")
    out["pairs"] = [{"det": i, "ref": j, "class": cls[j],
                     "v_ref": float(references[j]["velocity_kmh"]),
                     "v_det": float(detections[i].velocity_kmh),
                     "dt_s": _time_offset(detections[i], references[j])} for i, j in m]
    out.update(_pair_errors(out["pairs"]))
    fpk = false_positive_kinds(detections, references, m, n_channels, dx, params.fp_near_s,
                               params.fp_min_channels)
    out["fp_near_label"] = sum(v == "near_label" for v in fpk.values())
    out["fp_no_label"] = sum(v == "no_label" for v in fpk.values())
    out["fp_kinds"] = fpk
    out["matches"] = m
    return out


def _pair_errors(pairs: list[dict]) -> dict:
    """Velocity errors (all and thick pairs) and the mean |dt| of the matched pairs."""
    out = {}
    for name, sel in (("", None), ("_thick", "thick")):
        pp = [p for p in pairs if sel is None or p["class"] == sel]
        e = velocity_errors([p["v_det"] for p in pp], [p["v_ref"] for p in pp])
        out.update({f"vel_{k}{name}": v for k, v in e.items()})
    out["mean_dt_s"] = float(np.mean([p["dt_s"] for p in pairs])) if pairs else float("nan")
    return out


SUMMED = ("n_det", "n_ref", "n_ref_thick", "n_ref_thin", "tp", "fp", "tp_thick", "tp_thin",
          "fn_thick", "fp_near_label", "fp_no_label")


def summarize_evaluations(evals: list[dict]) -> dict:
    """Pooled (micro-averaged) metrics over several recordings of :func:`evaluate_objects`:
    counts are summed and the ratios / errors are recomputed from the pooled counts and pairs."""
    tot = {k: sum(e[k] for e in evals) for k in SUMMED}
    tot["precision"], tot["recall_thick"], tot["f1"] = _prf(
        tot["tp_thick"], tot["n_ref_thick"], tot["tp"], tot["n_det"])
    tot["recall_thin"] = tot["tp_thin"] / tot["n_ref_thin"] if tot["n_ref_thin"] else float("nan")
    tot.update(_pair_errors([p for e in evals for p in e["pairs"]]))
    return tot
