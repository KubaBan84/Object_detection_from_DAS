"""Tests of das.evaluation (Stage 9): references, matching, grade-4 scoring."""
from __future__ import annotations

import numpy as np
import pytest

from das.config import EvaluationParams
from das.evaluation import (detection_metrics, evaluate_objects, false_positive_kinds,
                            gt_objects_from_mask, line_trajectory, match_detections,
                            references_from_annotations, split_references,
                            summarize_evaluations, vehicle_class, velocity_errors)
from das.grouping import DetectedObject

DT, DX = 0.0016, 5.1065


def _mask_line(gt: np.ndarray, marker: int, v_kmh: float, t0: float, width_s: float = 0.05):
    """Mark the samples of a vehicle passing channel c at t0 + x_c / v."""
    x = (np.arange(gt.shape[1]) + 0.5) * DX
    t = np.arange(gt.shape[0]) * DT
    tc = t0 + (x - (x[0] if v_kmh > 0 else x[-1])) / (v_kmh / 3.6)
    gt[np.abs(t[:, None] - tc[None, :]) < width_s] = marker


def test_gt_objects_from_mask():
    gt = np.zeros((40000, 20), np.int32)       # 64 s x 20 channels
    _mask_line(gt, 1, 50.0, 5.0)
    _mask_line(gt, 2, -80.0, 20.0)
    meta = {"ground_truth": {"vehicles": [
        {"vehicle_id": 0, "type": "car", "direction": 1, "marker_value": 1},
        {"vehicle_id": 1, "type": "tram", "direction": -1, "marker_value": 2},
        {"vehicle_id": 2, "type": "car", "direction": 1, "marker_value": 3}]}}   # not in mask
    refs = gt_objects_from_mask(gt, meta, DT, DX)
    assert [r["marker"] for r in refs] == [1, 2]          # marker 3 is missing from the mask
    assert refs[0]["velocity_kmh"] == pytest.approx(50, rel=0.01)
    assert refs[1]["velocity_kmh"] == pytest.approx(-80, rel=0.01)
    assert np.sign(refs[1]["velocity_kmh"]) == refs[1]["direction"]
    assert refs[1]["type"] == "tram" and refs[0]["track"].shape == (20, 2)


def _obj(v_kmh, t_at_0, x0=10.0, x1=200.0):
    """Detection whose averaged line passes x = 0 at t_at_0 (x_ref = 0)."""
    return DetectedObject(0, v_kmh, t_at_0, x0, x1, t_at_0 + x0 * 3.6 / v_kmh,
                          t_at_0 + x1 * 3.6 / v_kmh, 5.0, direction=int(np.sign(v_kmh)),
                          x_ref_m=0.0)


def _ref(v_kmh, t_at_0, vtype, x0=10.0, x1=200.0):
    pts = np.array([[x0, t_at_0 + x0 * 3.6 / v_kmh], [x1, t_at_0 + x1 * 3.6 / v_kmh]])
    return references_from_annotations([{"id": 1, "kind": vtype, "points": pts.tolist()}])[0]


def test_vehicle_class():
    assert vehicle_class("tram") == vehicle_class("truck") == vehicle_class("thick") == "thick"
    assert vehicle_class("van") == vehicle_class("thin") == "thin"
    assert vehicle_class("bike") == "unknown"


def test_references_from_annotations():
    lines = [{"id": 3, "kind": "thin", "points": [[200.0, 20.0], [20.0, 7.04]]},   # any order
             {"id": 4, "kind": "thick", "points": [[50.0, 30.0], [100.0, 32.0], [150.0, 36.0]]},
             {"id": 5, "kind": "thin", "points": [[50.0, 3.0]]}]                     # skipped
    refs = references_from_annotations(lines)
    assert len(refs) == 2 and refs[0]["marker"] == 3 and refs[0]["type"] == "thin"
    assert refs[0]["velocity_kmh"] == pytest.approx(180 / 12.96 * 3.6)
    assert refs[0]["track"][0, 0] == 20.0 and refs[0]["direction"] == 1    # sorted by x
    assert refs[1]["velocity_kmh"] == pytest.approx(3.6 / 0.06)            # LSQ slope 0.06 s/m
    neg = references_from_annotations([{"kind": "thin", "points": [[0, 10], [100, 5]]}])[0]
    assert neg["direction"] == -1 and neg["velocity_kmh"] == pytest.approx(-72.0)


def test_match_detections_one_to_one_with_gating():
    refs = [_ref(50, 10, "thick"), _ref(-80, 40, "thick")]
    dets = [_obj(52, 10.2),          # matches ref 0
            _obj(49, 10.6),          # duplicate of ref 0 -> FP (one-to-one)
            _obj(80, 40.0),          # wrong direction for ref 1 -> FP
            _obj(-78, 39.9),         # matches ref 1
            _obj(-80, 43.0)]         # outside the time gate
    m = match_detections(dets, refs, tol_t_s=1.5, tol_v_kmh=20)
    assert sorted(m) == [(0, 0), (3, 1)]
    met = detection_metrics(len(dets), len(refs), m)
    assert (met["tp"], met["fp"], met["fn"]) == (2, 3, 0)
    assert met["precision"] == 0.4 and met["recall"] == 1.0


def test_metrics_edge_cases():
    assert np.isnan(detection_metrics(0, 0, [])["f1"])
    assert detection_metrics(0, 5, [])["recall"] == 0.0
    assert np.isnan(velocity_errors([], [])["mae"])
    e = velocity_errors([52.0, -78.0], [50.0, -80.0])
    assert e["mae"] == pytest.approx(2.0) and e["mape"] == pytest.approx(100 * (2 / 50 + 2 / 80) / 2)


def test_split_references_drops_implausible_and_short_labels():
    ok = _ref(50, 10, "thin")
    fast = _ref(400, 20, "thin")                                # near-horizontal stroke
    short = _ref(50, 30, "thin", x0=10.0, x1=25.0)               # 15 m only
    flat = references_from_annotations([{"id": 4, "kind": "thin",
                                         "points": [[1.0, 3.7], [35.0, 3.7]]}])[0]
    keep, excluded = split_references([ok, fast, short, flat], (5.0, 150.0), 30.0)
    assert keep == [ok]
    assert [e["reason"] for e in excluded] == ["velocity outside the detector fan", "too short",
                                               "velocity outside the detector fan"]
    assert np.isnan(flat["velocity_kmh"])                       # dt = 0: no velocity


def test_line_trajectory_inside_extent():
    cols, t = line_trajectory(_obj(36.0, 0.0, x0=10.0, x1=50.0), 52, DX)
    assert cols[0] == 1 and cols[-1] == 9                       # channel centres 7.7 ... 48.5 m
    np.testing.assert_allclose(t, (cols + 0.5) * DX / 10.0)    # 36 km/h = 10 m/s


def test_grade4_scoring_thin_match_is_not_a_false_positive():
    refs = [_ref(50, 10, "thick"), _ref(-60, 40, "thick"), _ref(60, 80, "thin"),
            _ref(70, 100, "thin")]
    dets = [_obj(52, 10.2),           # thick found
            _obj(61, 80.1),           # loud car (thin label) found: a correct detection
            _obj(50, 60.0)]           # nothing there: FP
    ev = evaluate_objects(dets, refs, EvaluationParams(), 52, DX)
    assert (ev["n_ref_thick"], ev["tp_thick"], ev["fn_thick"]) == (2, 1, 1)
    assert (ev["n_ref_thin"], ev["tp_thin"]) == (2, 1)
    assert ev["recall_thick"] == 0.5 and ev["recall_thin"] == 0.5
    assert (ev["tp"], ev["fp"]) == (2, 1) and ev["precision"] == pytest.approx(2 / 3)
    assert ev["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert ev["vel_mae_thick"] == pytest.approx(2.0) and ev["vel_mae"] == pytest.approx(1.5)
    assert ev["mean_dt_s"] < 0.5 and len(ev["pairs"]) == 2
    tot = summarize_evaluations([ev, ev])
    assert (tot["tp_thick"], tot["n_ref_thick"], tot["fp"]) == (2, 4, 2)
    assert tot["recall_thick"] == ev["recall_thick"] and tot["precision"] == ev["precision"]
    assert tot["vel_n_thick"] == 2 and tot["vel_mae"] == pytest.approx(ev["vel_mae"])


def test_false_positive_kinds_near_label_or_unlabelled():
    refs = [_ref(50, 10, "thick")]
    dets = [_obj(50, 10.1),                                     # matched
            _obj(50, 10.4, x0=60.0, x1=120.0),                  # fragment in the halo
            _obj(50, 60.0)]                                     # nothing labelled there
    m = match_detections(dets, refs, 1.5, 20.0)
    assert m == [(0, 0)]
    assert false_positive_kinds(dets, refs, m, 52, DX, 1.0, 3) == {1: "near_label", 2: "no_label"}
    ev = evaluate_objects(dets, refs, EvaluationParams(), 52, DX)
    assert (ev["fp"], ev["fp_near_label"], ev["fp_no_label"]) == (2, 1, 1)
