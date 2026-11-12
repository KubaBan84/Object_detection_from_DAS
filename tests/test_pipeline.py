"""Tests of das.pipeline (Stage 12): the whole chain on a raw toy recording (two vehicles
in opposite directions), timings, counts, features, saving and the stage cache."""
from __future__ import annotations

import csv
import json
from datetime import datetime

import numpy as np
import pytest

from das.config import PipelineParams
from das.pipeline import (STAGE_TIMERS, TABLE_FEATURES, load_cache, result_table,
                          run_pipeline, save_result, stage_images)
from das.velocity import TABLE_FIELDS, reported_velocity_kmh

FS, DX, N_CH, DUR = 625.0, 5.1065, 52, 60.0
VEHICLES = [(50.0, 10.0), (-60.0, 30.0)]        # (velocity km/h, time at the entry end [s])


def raw_recording(seed: int = 0) -> np.ndarray:
    """Noise + two vehicles: 100 Hz vibration under a 0.4 s Gaussian envelope moving
    along the cable (v > 0 enters at x = 0, v < 0 at the far end)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(DUR * FS)) / FS
    x = (np.arange(N_CH) + 0.5) * DX
    sig = 0.1 * rng.standard_normal((t.size, N_CH))
    for v_kmh, t_in in VEHICLES:
        v = v_kmh / 3.6
        t_pass = t_in + (x if v > 0 else x - x[-1]) / v        # passage time per channel
        env = np.exp(-0.5 * ((t[:, None] - t_pass[None, :]) / 0.4) ** 2)
        sig += env * np.sin(2 * np.pi * 100.0 * t)[:, None]
    return sig.astype(np.float32)


@pytest.fixture(scope="module")
def result():
    return run_pipeline(raw_recording(), PipelineParams(), FS, DX)


def test_pipeline_finds_both_vehicles_with_velocities(result):
    for v_true, _ in VEHICLES:
        cand = [o for o in result.objects if o.direction == np.sign(v_true)]
        assert cand, f"no object in direction {np.sign(v_true)}"
        best = max(cand, key=lambda o: o.x_max - o.x_min)
        assert reported_velocity_kmh(best) == pytest.approx(v_true, rel=0.08)
        assert best.x_max - best.x_min > 150


def test_timings_counts_and_features(result):
    assert set(STAGE_TIMERS) <= set(result.timings)
    assert result.timings["total_s"] == pytest.approx(sum(result.timings[k] for k in STAGE_TIMERS))
    c = result.counts
    assert c["n_objects"] == len(result.objects)
    assert c["strong_pos"] + c["strong_neg"] + c["weak_pos"] + c["weak_neg"] == c["n_objects"]
    assert c["strong_pos"] + c["weak_pos"] >= 1 and c["strong_neg"] + c["weak_neg"] >= 1
    assert len(result.features) == len(result.objects)
    f = max(result.features, key=lambda d: d["extent_m"])
    assert np.isfinite(f["thickness_s"]) and np.isfinite(f["snr_hf_db"]) and f["snr_hf_db"] > 3


def test_options_features_off_and_stages_dropped():
    res = run_pipeline(raw_recording(1)[: int(30 * FS)], PipelineParams(), FS, DX,
                       keep_stages=False, features=False)
    assert res.features == [] and "features_s" not in res.timings
    assert res.pre.common == {} and all(e.stages == {} for e in res.enhanced.values())
    assert res.objects                                            # still detected and tracked


def test_reused_stages_give_the_same_objects(result):
    res = run_pipeline(raw_recording(), PipelineParams(), FS, DX, features=False,
                       pre=result.pre, enhanced=result.enhanced, detections=result.detections)
    assert res.timings["enhance_s"] < 0.01                       # not recomputed
    assert [(o.direction, round(o.v_track_kmh, 6)) for o in res.objects] ==         [(o.direction, round(o.v_track_kmh, 6)) for o in result.objects]


def test_save_result_csv_json_and_cache(result, tmp_path):
    result.clusters = [f"c{k % 2}" for k in range(len(result.objects))]
    t0 = datetime(2024, 5, 7, 9, 0, 0)
    paths = save_result(result, "toy", t0, PipelineParams(), out_dir=tmp_path,
                        cache_dir=tmp_path / "cache", extra={"kind": "test"})
    with open(paths["csv"], encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(result.objects)
    assert list(rows[0]) == TABLE_FIELDS + TABLE_FEATURES + ["cluster"]
    assert [r["cluster"] for r in rows] == result.clusters
    assert rows[0]["t_ref_clock"].startswith("09:00:")
    s = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert s["name"] == "toy" and s["kind"] == "test" and s["start"] == t0.isoformat()
    assert s["counts"] == result.counts and s["timings_s"]["total_s"] > 0
    assert s["params"]["enh"]["n_angles"] == PipelineParams().enh.n_angles
    assert sum(s["clusters"].values()) == len(result.objects)
    cached = load_cache("toy", tmp_path / "cache")
    imgs = stage_images(result)
    assert set(cached) == set(imgs) == set(s["cache"]["images"])
    assert {"raw_log_rms", "image_hf", "response_hf_pos", "fk_lf_neg"} <= set(cached)
    assert np.allclose(cached["response_hf_pos"], imgs["response_hf_pos"])
    assert load_cache("missing", tmp_path / "cache") == {}
    result.clusters = []
    assert all(r["cluster"] == "" for r in result_table(result))
