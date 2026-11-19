"""Tests of das.pipeline (Stage 10): the whole chain on a raw toy recording (two vehicles
in opposite directions), timings, counts, saving and the stage cache."""
from __future__ import annotations

import csv
import json
from datetime import datetime

import numpy as np
import pytest

from das.config import PipelineParams
from das.pipeline import STAGE_TIMERS, load_cache, run_pipeline, save_result, stage_images
from das.velocity import TABLE_FIELDS

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
    assert len(result.objects) == 2
    for v_true, _ in VEHICLES:
        (obj,) = [o for o in result.objects if o.direction == np.sign(v_true)]
        assert obj.velocity_kmh == pytest.approx(v_true, rel=0.05)
        assert obj.x_max - obj.x_min > 150


def test_timings_and_counts(result):
    assert set(STAGE_TIMERS) <= set(result.timings)
    assert result.timings["total_s"] == pytest.approx(sum(result.timings[k] for k in STAGE_TIMERS))
    assert result.counts == {"n_objects": 2, "n_pos": 1, "n_neg": 1}
    assert result.n_channels == N_CH and result.col_dx == pytest.approx(DX / 8)


def test_reused_stages_give_the_same_objects(result):
    res = run_pipeline(raw_recording(), PipelineParams(), FS, DX, pre=result.pre,
                       enhanced=result.enhanced)
    assert res.timings["enhance_s"] < 0.01                       # not recomputed
    assert [round(o.velocity_kmh, 6) for o in res.objects] == \
        [round(o.velocity_kmh, 6) for o in result.objects]


def test_save_result_csv_json_and_cache(result, tmp_path):
    t0 = datetime(2024, 5, 7, 9, 0, 0)
    paths = save_result(result, "toy", t0, PipelineParams(), out_dir=tmp_path,
                        cache_dir=tmp_path / "cache", extra={"kind": "test"})
    with open(paths["csv"], encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(result.objects) and list(rows[0]) == TABLE_FIELDS
    assert rows[0]["t_ref_clock"].startswith("09:00:")
    s = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert s["name"] == "toy" and s["kind"] == "test" and s["start"] == t0.isoformat()
    assert s["counts"] == result.counts and s["timings_s"]["total_s"] > 0
    assert s["params"]["enh"]["n_angles"] == PipelineParams().enh.n_angles
    assert set(s["thresholds"]) == {"pos", "neg"}
    cached = load_cache("toy", tmp_path / "cache")
    imgs = stage_images(result)
    assert set(cached) == set(imgs) == set(s["cache"]["images"])
    assert {"raw_log_rms", "image", "response_pos", "fk_neg"} <= set(cached)
    assert np.allclose(cached["response_pos"], imgs["response_pos"])
    assert load_cache("missing", tmp_path / "cache") == {}
