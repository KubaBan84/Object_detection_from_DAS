"""Tests of the object-level evaluation helpers used from Stage 6 on (das.evaluation)."""
from __future__ import annotations

import numpy as np
import pytest

from das.config import EvaluationParams
from das.evaluation import (auc_score, class_metrics, detection_metrics, evaluate_objects,
                            false_positive_kinds, gt_objects_from_mask, match_detections,
                            predicted_class, split_references,
                            rasterize_tracks, references_from_annotations,
                            summarize_evaluations, trajectory_iou, trajectory_overlap,
                            vehicle_class, velocity_errors)
from das.grouping import DetectedObject

DT, DX = 0.0016, 5.1065


def _mask_line(gt: np.ndarray, marker: int, v_kmh: float, t0: float, width_s: float = 0.05):
    """Mark the samples of a vehicle passing channel c at t0 + x_c / v."""
    x = (np.arange(gt.shape[1]) + 0.5) * DX
    t = np.arange(gt.shape[0]) * DT
    tc = t0 + (x - (x[0] if v_kmh > 0 else x[-1])) / (v_kmh / 3.6)
    sel = np.abs(t[:, None] - tc[None, :]) < width_s
    gt[sel] = marker


@pytest.fixture
def gt_scene():
    gt = np.zeros((40000, 20), np.int32)       # 64 s x 20 channels
    _mask_line(gt, 1, 50.0, 5.0)
    _mask_line(gt, 2, -80.0, 20.0)
    meta = {"ground_truth": {"vehicles": [
        {"vehicle_id": 0, "type": "car", "direction": 1, "marker_value": 1},
        {"vehicle_id": 1, "type": "tram", "direction": -1, "marker_value": 2},
        {"vehicle_id": 2, "type": "car", "direction": 1, "marker_value": 3}]}}   # not in mask
    return gt, meta


def test_gt_objects_from_mask(gt_scene):
    gt, meta = gt_scene
    refs = gt_objects_from_mask(gt, meta, DT, DX)
    assert [r["marker"] for r in refs] == [1, 2]          # marker 3 is missing from the mask
    assert refs[0]["velocity_kmh"] == pytest.approx(50, rel=0.01)
    assert refs[1]["velocity_kmh"] == pytest.approx(-80, rel=0.01)
    assert np.sign(refs[0]["velocity_kmh"]) == refs[0]["direction"]
    assert np.sign(refs[1]["velocity_kmh"]) == refs[1]["direction"]
    assert refs[1]["type"] == "tram" and refs[0]["track"].shape == (20, 2)


def _det(v, t_ref, x_ref, x0=0.0, x1=100.0):
    return DetectedObject(0, v, t_ref, x0, x1, 0.0, 0.0, "weak", 1.0, direction=int(np.sign(v)),
                          x_ref_m=x_ref)


def test_match_detections_one_to_one_with_gating(gt_scene):
    gt, meta = gt_scene
    refs = gt_objects_from_mask(gt, meta, DT, DX)
    xm = 50.0
    t1 = float(np.interp(xm, refs[0]["track"][:, 0], refs[0]["track"][:, 1]))
    t2 = float(np.interp(xm, refs[1]["track"][::-1, 0], refs[1]["track"][::-1, 1])) \
        if refs[1]["track"][0, 0] > refs[1]["track"][-1, 0] else \
        float(np.interp(xm, refs[1]["track"][:, 0], refs[1]["track"][:, 1]))
    dets = [_det(52, t1 + 0.2, xm),          # matches ref 0
            _det(49, t1 + 0.6, xm),          # duplicate of ref 0 -> FP (one-to-one)
            _det(80, t2, xm),                # wrong direction for ref 1 -> FP
            _det(-78, t2 - 0.1, xm)]         # matches ref 1
    m = match_detections(dets, refs, tol_t_s=1.5, tol_v_kmh=20)
    assert sorted(m) == [(0, 0), (3, 1)]
    met = detection_metrics(len(dets), len(refs), m)
    assert (met["tp"], met["fp"], met["fn"]) == (2, 2, 0)
    assert met["precision"] == 0.5 and met["recall"] == 1.0
    assert met["f1"] == pytest.approx(2 / 3)
    ve = velocity_errors(dets, refs, m)
    assert ve["n"] == 2 and ve["mae"] == pytest.approx(2.0, abs=0.5)


def test_metrics_edge_cases():
    assert np.isnan(detection_metrics(0, 0, [])["f1"])
    assert detection_metrics(0, 5, [])["recall"] == 0.0
    assert np.isnan(velocity_errors([], [], [])["mae"])


# ======================================================================================
# Stage 11: per-class metrics, trajectory IoU, manual references, pooled summary
# ======================================================================================
def _line_obj(v_kmh, t_at_0, kind, x0=10.0, x1=200.0):
    """Detection whose averaged line passes x = 0 at t_at_0 (x_ref = 0)."""
    return DetectedObject(0, v_kmh, t_at_0, x0, x1, t_at_0 + x0 * 3.6 / v_kmh,
                          t_at_0 + x1 * 3.6 / v_kmh, kind, 5.0, direction=int(np.sign(v_kmh)),
                          x_ref_m=0.0)


def _ref(v_kmh, t_at_0, vtype, x0=10.0, x1=200.0):
    pts = np.array([[x0, t_at_0 + x0 * 3.6 / v_kmh], [x1, t_at_0 + x1 * 3.6 / v_kmh]])
    return references_from_annotations([{"id": 1, "kind": vtype, "points": pts.tolist()}])[0]


def test_vehicle_and_predicted_class():
    assert vehicle_class("tram") == vehicle_class("thick") == "heavy"
    assert vehicle_class("van") == vehicle_class("thin") == "light"
    assert vehicle_class("bike") == "unknown"
    assert predicted_class(_line_obj(50, 0, "strong")) == "heavy"
    assert predicted_class(_line_obj(50, 0, "weak")) == "light"


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


def test_class_metrics_precision_by_detection_path():
    refs = [_ref(50, 10, "tram"), _ref(60, 40, "car"), _ref(70, 80, "car")]
    dets = [_line_obj(50, 10.2, "strong"),          # tram found by the strong path
            _line_obj(60, 40.1, "strong"),          # car found by the strong path
            _line_obj(40, 100.0, "weak")]           # false weak detection
    m = match_detections(dets, refs, 1.5, 20)
    cm = class_metrics(dets, refs, m)
    assert (cm["n_ref_heavy"], cm["tp_heavy"], cm["recall_heavy"]) == (1, 1, 1.0)
    assert (cm["n_ref_light"], cm["tp_light"], cm["recall_light"]) == (2, 1, 0.5)
    assert (cm["n_det_heavy"], cm["tp_det_heavy"], cm["precision_heavy"]) == (2, 2, 1.0)
    assert (cm["n_det_light"], cm["precision_light"]) == (1, 0.0)
    assert cm["conf_light_as_heavy"] == 1 and cm["conf_heavy_as_heavy"] == 1


def test_rasterize_and_trajectory_iou():
    shape, row_dt, dx = (500, 20), 0.04, 5.0
    tr = np.array([[2.5, 4.0], [97.5, 8.0]])                    # channels 0..19, 4 -> 8 s
    a = rasterize_tracks([tr], shape, row_dt, dx)
    assert a.sum() == 20 and np.all(a.sum(axis=0) == 1)         # one pixel per channel
    b = rasterize_tracks([tr + [0.0, 0.08]], shape, row_dt, dx)  # 2 rows later
    far = rasterize_tracks([tr + [0.0, 5.0]], shape, row_dt, dx)
    assert trajectory_iou(a, a, 0) == 1.0
    assert trajectory_iou(a, b, 0) == 0.0 and trajectory_iou(a, b, 3) == pytest.approx(5 / 9)
    assert trajectory_iou(a, far, 3) == 0.0
    assert np.isnan(trajectory_iou(np.zeros(shape, bool), np.zeros(shape, bool), 3))
    assert trajectory_overlap(a, a, 1) == (60, 60)
    half = rasterize_tracks([tr[:, :] * [0.5, 1.0] + [0.0, 0.0]], shape, row_dt, dx)
    assert half.sum() < a.sum()                                 # shorter track, fewer channels


def test_evaluate_objects_and_pooled_summary():
    refs = [_ref(50, 10, "tram"), _ref(60, 40, "car"), _ref(70, 80, "car")]
    dets = [_line_obj(52, 10.2, "strong"), _line_obj(60, 40.1, "weak")]
    ev = evaluate_objects(dets, refs, EvaluationParams(), (3000, 52), 0.04, DX)
    assert (ev["tp"], ev["fp"], ev["fn"]) == (2, 0, 1)
    assert ev["vel_mae"] == pytest.approx(1.0) and ev["vel_mae_heavy"] == pytest.approx(2.0)
    assert 0.3 < ev["traj_iou"] < 1.0 and len(ev["pairs"]) == 2
    tot = summarize_evaluations([ev, ev])
    assert (tot["tp"], tot["fn"], tot["n_ref_light"]) == (4, 2, 4)
    assert tot["recall"] == pytest.approx(ev["recall"]) and tot["f1"] == pytest.approx(ev["f1"])
    assert tot["vel_mae"] == pytest.approx(ev["vel_mae"]) and tot["vel_n"] == 4
    assert tot["traj_iou"] == pytest.approx(ev["traj_iou"])


def test_auc_score_with_ties():
    assert auc_score([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert auc_score([0, 1, 2, 3], [False, False, True, True]) == 1.0
    assert auc_score([3, 2, 1, 0], [False, False, True, True]) == 0.0
    assert np.isnan(auc_score([1, 2], [True, True]))


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


def test_false_positive_kinds_near_label_or_unlabelled():
    refs = [_ref(50, 10, "tram")]
    dets = [_line_obj(50, 10.1, "strong"),                      # matched
            _line_obj(50, 10.4, "weak", x0=60.0, x1=120.0),     # fragment in the halo
            _line_obj(50, 60.0, "weak")]                        # nothing labelled there
    m = match_detections(dets, refs, 1.5, 20.0)
    assert m == [(0, 0)]
    kinds = false_positive_kinds(dets, refs, m, 52, DX, 1.0, 3)
    assert kinds == {1: "near_label", 2: "no_label"}
    ev = evaluate_objects(dets, refs, EvaluationParams(), (3000, 52), 0.04, DX)
    assert (ev["fp"], ev["fp_near_label"], ev["fp_no_label"]) == (2, 1, 1)
    tot = summarize_evaluations([ev, ev])
    assert (tot["fp_near_label"], tot["fp_no_label"]) == (2, 2)
