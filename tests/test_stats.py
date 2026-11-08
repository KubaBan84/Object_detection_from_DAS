"""Tests of das.stats (Stage 2)."""
from __future__ import annotations

import numpy as np
import pytest

from das import stats


@pytest.fixture
def gauss() -> np.ndarray:
    return np.random.default_rng(1).standard_normal((20000, 6))


def test_describe_gaussian(gauss):
    d = stats.describe(gauss)
    assert d["n"] == gauss.size
    assert d["std"] == pytest.approx(1.0, abs=0.02)
    assert d["robust_sigma"] == pytest.approx(1.0, abs=0.03)
    assert abs(d["skewness"]) < 0.05 and abs(d["kurtosis"]) < 0.1
    assert d["p50"] == pytest.approx(d["median"])
    assert d["min"] <= d["p0.1"] <= d["p99.9"] <= d["max"]


def test_describe_constant_has_no_nan():
    d = stats.describe(np.ones((10, 3)))
    assert d["std"] == 0 and d["skewness"] == 0 and d["kurtosis"] == 0


def test_per_channel_stats(gauss):
    x = gauss * np.arange(1, 7)  # channel c has std c+1
    pc = stats.per_channel_stats(x)
    np.testing.assert_allclose(pc["std"], np.arange(1, 7), rtol=0.03)
    np.testing.assert_allclose(1.4826 * pc["mad"], np.arange(1, 7), rtol=0.05)
    assert all(v.shape == (6,) for v in pc.values())


def test_block_rms_and_per_time_stats():
    x = np.ones((1003, 2))
    x[:, 1] = 2.0
    r = stats.block_rms(x, 10)
    assert r.shape == (100, 2)  # trailing 3 samples dropped
    np.testing.assert_allclose(r, [[1.0, 2.0]] * 100)
    t = stats.per_time_stats(x, 100)
    assert t["rms_map"].shape == (10, 2)
    np.testing.assert_allclose(t["rms_mean"], 1.5)
    with pytest.raises(ValueError):
        stats.block_rms(x, 0)


def test_running_stats_matches_numpy(gauss):
    x = gauss * 3 + 0.5 + 0.3 * gauss ** 2  # skewed, non-zero mean
    rs = stats.RunningStats(6, bins=np.linspace(-20, 20, 81), abs_bins=np.logspace(-4, 2, 61))
    for chunk in np.array_split(x, 7):  # unequal batches
        rs.update(chunk)
    res = rs.result()
    pc, g = res["per_channel"], res["global"]
    np.testing.assert_allclose(pc["mean"], x.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(pc["std"], x.std(axis=0), rtol=1e-10)
    np.testing.assert_allclose(pc["rms"], np.sqrt((x ** 2).mean(axis=0)), rtol=1e-10)
    ref = stats.describe(x)
    assert g["std"] == pytest.approx(ref["std"], rel=1e-10)
    assert g["skewness"] == pytest.approx(ref["skewness"], rel=1e-8)
    assert g["kurtosis"] == pytest.approx(ref["kurtosis"], rel=1e-8)
    assert res["hist"]["counts"].sum() + res["hist"]["under"] + res["hist"]["over"] == x.size
    assert g["abs_p50"] == pytest.approx(np.median(np.abs(x)), rel=0.1)


def test_running_stats_rejects_wrong_shape():
    rs = stats.RunningStats(3, np.linspace(-1, 1, 5), np.logspace(-2, 0, 5))
    with pytest.raises(ValueError):
        rs.update(np.zeros((10, 4)))
    with pytest.raises(ValueError):
        stats.RunningStats(3, np.linspace(-1, 1, 5), np.logspace(-2, 0, 5)).result()


def test_neighbour_correlation():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(5000)
    x = np.column_stack([a, a + 0.01 * rng.standard_normal(5000), rng.standard_normal(5000)])
    r = stats.neighbour_correlation(x)
    assert r[0] > 0.99 and abs(r[1]) < 0.05
    np.testing.assert_allclose(np.diag(stats.channel_correlation_matrix(x)), 1.0)


def test_detect_bad_channels():
    rms = np.full(30, 1e-7) * np.random.default_rng(3).uniform(0.8, 1.25, 30)
    rms[5] = 1e-9   # dead channel
    rms[20] = 1e-5  # very noisy channel
    mask, z = stats.detect_bad_channels(rms)
    assert np.flatnonzero(mask).tolist() == [5, 20]
    assert z[5] < 0 < z[20]


def test_activity_metrics():
    ref = np.ones(4)
    quiet = np.full((10, 4), 0.5)
    m = stats.activity_metrics(quiet, ref)
    assert m["active_fraction"] == 0 and m["n_events"] == 0
    busy = quiet.copy()
    busy[2:4] = 5.0   # event 1 (all channels active)
    busy[7, :2] = 20  # event 2 (half of the channels, strong)
    m = stats.activity_metrics(busy, ref)
    assert m["n_events"] == 2
    assert m["active_fraction"] == pytest.approx(10 / 40)
    assert m["strong_fraction"] == pytest.approx(2 / 40)


def test_hist_percentiles():
    edges = np.array([0.0, 1.0, 2.0])
    p = stats.hist_percentiles(edges, np.array([50, 50]), [50, 100])
    np.testing.assert_allclose(p, [1.0, 2.0])
