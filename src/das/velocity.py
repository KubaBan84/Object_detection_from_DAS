"""Velocity computation and unit conversion (plan: Stage 8).

Two velocity estimates exist for every object:

- the *line velocity* of the averaged Stage-7 line (fit to the Hough / RANSAC
  candidates in the image) - available for every object;
- the *track velocity*: least-squares fit x(t) = v t + b to the ridge points tracked in
  every channel (Stage 9). It uses the measured passage time at each position instead of
  candidate segments, so it is the value reported in the tables when a track exists;
  its R^2 tells how well a constant velocity describes the object.
"""
from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

import numpy as np


def fit_constant_velocity(track: np.ndarray) -> tuple[float, float, float]:
    """Least-squares x(t) = v t + b on tracked points -> (v [m/s], b [m], R^2).

    ``track`` is (N, 2) [x_m, t_s]. Time is the independent variable (the passage time is
    what is measured, but v = dx/dt is what is reported; for N >= 3 points with a
    non-degenerate time span). R^2 = 1 - SS_res / SS_tot of x.
    """
    x, t = np.asarray(track[:, 0], float), np.asarray(track[:, 1], float)
    if x.size < 2 or np.ptp(t) == 0:
        return float("nan"), float("nan"), float("nan")
    tc = t - t.mean()
    v = float(np.sum(tc * (x - x.mean())) / np.sum(tc ** 2))
    b = float(x.mean() - v * t.mean())
    ss_res = float(np.sum((x - (v * t + b)) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    return v, b, 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def reported_velocity_kmh(obj) -> float:
    """Velocity for the tables: the track fit if available, else the Stage-7 line."""
    v = getattr(obj, "v_track_kmh", float("nan"))
    return float(v) if np.isfinite(v) else float(obj.velocity_kmh)


# ======================================================================================
# Object table (Stage 8)
# ======================================================================================
TABLE_FIELDS = ["id", "kind", "direction", "velocity_kmh", "v_line_kmh", "r2", "speed_class",
                "accel_ms2", "v_start_kmh", "v_end_kmh", "v_std_kmh", "t_ref_s", "t_start_s",
                "t_end_s", "x_start_m", "x_end_m", "n_track", "support_m", "score",
                "thickness_s", "bands", "sources", "n_candidates", "t_ref_clock"]


def objects_table(objects: list, t0=None) -> list[dict]:
    """One row per object; ``velocity_kmh`` is the reported (track-fit) velocity.

    Times are seconds from the segment start (``t_ref_clock`` = wall-clock time of t_ref
    if the segment start ``t0`` is given). Numbers are rounded for readability.
    """
    def r(v, nd=2):
        return "" if v is None or not np.isfinite(v) else round(float(v), nd)

    rows = []
    for o in objects:
        rows.append({
            "id": o.obj_id, "kind": o.kind, "direction": o.direction,
            "velocity_kmh": r(reported_velocity_kmh(o)), "v_line_kmh": r(o.velocity_kmh),
            "r2": r(o.r2, 4), "speed_class": o.speed_class, "accel_ms2": r(o.accel_ms2, 3),
            "v_start_kmh": r(o.v_start_kmh), "v_end_kmh": r(o.v_end_kmh),
            "v_std_kmh": r(o.v_std_kmh), "t_ref_s": r(o.t_ref_s), "t_start_s": r(o.t_start_s),
            "t_end_s": r(o.t_end_s), "x_start_m": r(o.x_start_m, 1), "x_end_m": r(o.x_end_m, 1),
            "n_track": 0 if o.track is None else len(o.track), "support_m": r(o.support_m, 1),
            "score": r(o.score), "thickness_s": r(o.thickness_s), "bands": "+".join(o.bands),
            "sources": "+".join(o.sources), "n_candidates": len(o.segments),
            "t_ref_clock": "" if t0 is None else
            f"{t0 + timedelta(seconds=o.t_ref_s):%H:%M:%S.%f}"[:-4]})
    return rows


def write_objects_csv(objects: list, path, t0=None) -> None:
    """Per-segment object table (csv module, no pandas)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_FIELDS)
        w.writeheader()
        w.writerows(objects_table(objects, t0))
