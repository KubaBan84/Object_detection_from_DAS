"""Velocity of the detected objects and the object table (plan: Stage 8).

In the (x, t) image a vehicle moving with a constant velocity v draws a straight line
t(x) = t_c + p (x - x_c). Its slope p = dt/dx [s/m] (the *slowness*) is estimated in
Stage 7 by a robust weighted least-squares fit to all candidate segments of the
vehicle, so

    v = 1 / p   [m/s],  x 3.6 -> km/h,

signed: v > 0 = towards increasing cable position, v < 0 = towards decreasing position.
In pixel units of the response map a slope of s columns per row corresponds to
v = s * col_dx / row_dt (``slope_to_velocity_kmh``).
"""
from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

import numpy as np


def slope_to_velocity_kmh(slope_col_per_row: float | np.ndarray, row_dt: float,
                          col_dx: float):
    """Image slope [columns per row] -> velocity [km/h] (signed)."""
    return np.asarray(slope_col_per_row) * col_dx / row_dt * 3.6


TABLE_FIELDS = ["id", "direction", "velocity_kmh", "t_ref_s", "t_ref_clock", "t_start_s",
                "t_end_s", "x_start_m", "x_end_m", "support_m", "score", "thickness_s",
                "fit_rms_s", "n_candidates"]


def objects_table(objects: list, t0=None) -> list[dict]:
    """One row per object (``velocity_kmh`` signed, from the averaged line).

    Times are seconds from the segment start; ``t_ref_clock`` is the wall-clock time of
    t_ref if the segment start ``t0`` is given. Numbers are rounded for readability.
    """
    def r(v, nd=2):
        return "" if v is None or not np.isfinite(v) else round(float(v), nd)

    rows = []
    for o in objects:
        rows.append({
            "id": o.obj_id, "direction": o.direction, "velocity_kmh": r(o.velocity_kmh),
            "t_ref_s": r(o.t_ref_s),
            "t_ref_clock": "" if t0 is None else
            f"{t0 + timedelta(seconds=o.t_ref_s):%H:%M:%S.%f}"[:-4],
            "t_start_s": r(o.t_start_s), "t_end_s": r(o.t_end_s),
            "x_start_m": r(o.x_start_m, 1), "x_end_m": r(o.x_end_m, 1),
            "support_m": r(o.support_m, 1), "score": r(o.score), "thickness_s": r(o.thickness_s),
            "fit_rms_s": r(o.fit_rms_s, 3), "n_candidates": len(o.segments)})
    return rows


def write_objects_csv(objects: list, path, t0=None) -> Path:
    """Per-recording object table (csv module, no pandas). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_FIELDS)
        w.writeheader()
        w.writerows(objects_table(objects, t0))
    return path
