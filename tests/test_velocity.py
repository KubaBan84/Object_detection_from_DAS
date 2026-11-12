"""Known slope -> known velocity in km/h; sign convention; constant-velocity fit (Stage 8)."""
import csv

import numpy as np
import pytest

from das.detection import velocity_to_slope
from das.grouping import DetectedObject
from das.velocity import (TABLE_FIELDS, fit_constant_velocity, objects_table,
                          reported_velocity_kmh, write_objects_csv)

ROW_DT, COL_DX = 0.04, 5.1065 / 8


def test_velocity_to_slope_known_values_and_sign():
    # 50 km/h = 13.9 m/s towards increasing x -> 13.9 * 0.04 / 0.638 = 0.871 columns per row
    s = velocity_to_slope(50 / 3.6, ROW_DT, COL_DX)
    assert float(s) == pytest.approx(0.8705, rel=1e-3)
    assert float(velocity_to_slope(-50 / 3.6, ROW_DT, COL_DX)) == pytest.approx(-s)
    assert velocity_to_slope(np.array([10.0, 20.0]), ROW_DT, COL_DX) == pytest.approx([s * 0.72, s * 1.44])


def test_fit_constant_velocity_known_line_and_r2():
    t = np.linspace(10, 30, 40)
    x = 250.0 - (60 / 3.6) * (t - 10)                           # 60 km/h towards decreasing x
    v, b, r2 = fit_constant_velocity(np.column_stack([x, t]))
    assert 3.6 * v == pytest.approx(-60.0) and r2 == pytest.approx(1.0)
    assert b == pytest.approx(250.0 + 60 / 3.6 * 10)
    rng = np.random.default_rng(0)
    noisy = np.column_stack([x + rng.normal(0, 5, x.size), t])
    assert 0.9 < fit_constant_velocity(noisy)[2] < 1.0
    assert np.isnan(fit_constant_velocity(np.array([[1.0, 2.0]]))[0])


def _obj(v_track=float("nan")):
    o = DetectedObject(3, -50.0, 40.0, 250.0, 10.0, 30.0, 47.0, "weak", 6.0, direction=-1,
                       x_ref_m=130.0, bands=("hf", "lf"), sources=("weak",))
    o.v_track_kmh, o.r2, o.speed_class = v_track, 0.995, "constant"
    o.track = np.zeros((12, 2))
    return o


def test_reported_velocity_prefers_track_fit():
    assert reported_velocity_kmh(_obj()) == -50.0
    assert reported_velocity_kmh(_obj(-52.5)) == -52.5


def test_objects_table_and_csv(tmp_path):
    from datetime import datetime

    rows = objects_table([_obj(-52.5)], datetime(2024, 5, 7, 9, 4, 52))
    assert rows[0]["velocity_kmh"] == -52.5 and rows[0]["v_line_kmh"] == -50.0
    assert rows[0]["t_ref_clock"] == "09:05:32.00" and rows[0]["n_track"] == 12
    assert rows[0]["accel_ms2"] == "" and rows[0]["bands"] == "hf+lf"
    path = tmp_path / "t.csv"
    write_objects_csv([_obj(-52.5), _obj()], path)
    with open(path, newline="", encoding="utf-8") as fh:
        read = list(csv.DictReader(fh))
    assert list(read[0].keys()) == TABLE_FIELDS and len(read) == 2
    assert read[1]["velocity_kmh"] == "-50.0"
