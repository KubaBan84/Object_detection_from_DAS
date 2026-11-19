"""Known slope -> known velocity in km/h; sign convention; object table (Stage 8)."""
import csv
from datetime import datetime

import numpy as np
import pytest

from das.detection import velocity_to_slope
from das.grouping import DetectedObject
from das.velocity import TABLE_FIELDS, objects_table, slope_to_velocity_kmh, write_objects_csv

ROW_DT, COL_DX = 0.04, 5.1065 / 8


def test_slope_to_velocity_round_trip_and_sign():
    for v_kmh in (-120.0, -50.0, 10.0, 75.0):
        s = velocity_to_slope(v_kmh / 3.6, ROW_DT, COL_DX)
        assert float(slope_to_velocity_kmh(s, ROW_DT, COL_DX)) == pytest.approx(v_kmh)
    # one column (0.64 m) per row (0.04 s) = 16 m/s = 57.4 km/h towards increasing x
    assert float(slope_to_velocity_kmh(1.0, ROW_DT, COL_DX)) == pytest.approx(57.447, rel=1e-4)
    assert float(slope_to_velocity_kmh(-1.0, ROW_DT, COL_DX)) < 0


def test_object_velocity_from_slowness():
    o = DetectedObject(0, -60.0, 40.0, 250.0, 10.0, 30.0, 47.0, 6.0, direction=-1, x_ref_m=130.0)
    assert o.slowness == pytest.approx(-0.06)                 # 3.6 / -60 s/m
    # 60 km/h = 16.67 m/s: 100 m further along -x takes 6 s
    assert float(o.t_at(30.0) - o.t_at(130.0)) == pytest.approx(6.0)


def test_objects_table_and_csv(tmp_path):
    o = DetectedObject(3, -52.456, 40.0, 250.0, 10.0, 30.0, 47.0, 6.0, direction=-1,
                       x_ref_m=130.0, thickness_s=1.2, support_m=200.0)
    rows = objects_table([o], datetime(2024, 5, 7, 9, 4, 52))
    assert rows[0]["velocity_kmh"] == -52.46 and rows[0]["direction"] == -1
    assert rows[0]["t_ref_clock"] == "09:05:32.00" and rows[0]["n_candidates"] == 0
    path = write_objects_csv([o, o], tmp_path / "t.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        read = list(csv.DictReader(fh))
    assert list(read[0].keys()) == TABLE_FIELDS and len(read) == 2
    assert read[1]["velocity_kmh"] == "-52.46" and read[0]["t_ref_clock"] == ""
    assert np.isfinite(float(read[0]["thickness_s"]))
