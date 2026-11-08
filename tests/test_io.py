"""Tests of das.io: timestamps, continuity, segment loading, synthetic samples (Stage 1)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from das import config
from das import io as dio

N_CH = 4  # small channel count keeps the fake files tiny


def _write_real(directory: Path, stems: list[str]) -> None:
    """Fake real files; every sample of file i equals i (to check concatenation order)."""
    for i, stem in enumerate(stems):
        np.save(directory / f"{stem}.npy",
                np.full((config.SAMPLES_PER_FILE, N_CH), i, dtype=np.float32))


# Mirrors the real layout: a short run, a 25 s gap, a longer run.
RUN_A = ["090017", "090027", "090037"]
RUN_B = ["090052", "090102", "090112", "090122", "090132"]


@pytest.fixture
def real_dir(tmp_path: Path) -> Path:
    _write_real(tmp_path, RUN_A + RUN_B)
    (tmp_path / "notes.txt").write_text("not a data file")
    (tmp_path / "12345.npy").write_bytes(b"")  # wrong name pattern, must be ignored
    return tmp_path


# ---------------------------------------------------------------- timestamps / listing
def test_parse_timestamp():
    assert dio.parse_timestamp("090452.npy") == datetime(2024, 5, 7, 9, 4, 52)
    assert dio.parse_timestamp(Path("x") / "093412.npy") == datetime(2024, 5, 7, 9, 34, 12)
    assert dio.parse_timestamp("090452") == datetime(2024, 5, 7, 9, 4, 52)


@pytest.mark.parametrize("bad", ["0904.npy", "abcdef.npy", "256000.npy", "090460.npy"])
def test_parse_timestamp_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        dio.parse_timestamp(bad)


def test_list_real_files_filters_and_sorts(real_dir):
    names = [f.stem for f in dio.list_real_files(real_dir)]
    assert names == sorted(RUN_A + RUN_B)


def test_list_real_files_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        dio.list_real_files(tmp_path / "nope")


# ---------------------------------------------------------------- continuity
def test_find_continuous_runs_splits_on_gap():
    files = [Path(f"{s}.npy") for s in reversed(RUN_A + RUN_B)]  # unsorted input
    runs = dio.find_continuous_runs(files)
    assert [[f.stem for f in r] for r in runs] == [RUN_A, RUN_B]


def test_find_continuous_runs_crosses_minute_boundary():
    runs = dio.find_continuous_runs([Path(s + ".npy") for s in ["085950", "090000", "090010"]])
    assert len(runs) == 1


def test_find_continuous_runs_empty():
    assert dio.find_continuous_runs([]) == []


# ---------------------------------------------------------------- loading files / segments
def test_load_real_file_validates_shape(tmp_path):
    path = tmp_path / "090000.npy"
    np.save(path, np.zeros((100, N_CH), dtype=np.float32))
    with pytest.raises(ValueError):
        dio.load_real_file(path)


def test_load_segment_shape_order_and_time(real_dir):
    seg = dio.load_segment("090102", n_files=3, directory=real_dir)
    assert seg.data.shape == (3 * config.SAMPLES_PER_FILE, N_CH)
    assert seg.data.dtype == np.float32
    assert seg.start == datetime(2024, 5, 7, 9, 1, 2)
    assert seg.end == datetime(2024, 5, 7, 9, 1, 32)
    assert seg.duration_s == pytest.approx(30.0)
    # file indices in the sorted listing: 090102 -> 4, 090112 -> 5, 090122 -> 6
    firsts = seg.data[::config.SAMPLES_PER_FILE, 0]
    np.testing.assert_array_equal(firsts, [4, 5, 6])
    assert seg.pad_before == seg.pad_after == 0
    assert seg.core.shape == seg.data.shape


def test_load_segment_accepts_npy_suffix(real_dir):
    seg = dio.load_segment("090052.npy", n_files=2, directory=real_dir)
    assert [f.stem for f in seg.files] == ["090052", "090102"]


def test_load_segment_across_gap_raises(real_dir):
    with pytest.raises(ValueError, match="does not fit"):
        dio.load_segment("090027", n_files=3, directory=real_dir)


def test_load_segment_unknown_start_raises(real_dir):
    with pytest.raises(FileNotFoundError):
        dio.load_segment("091500", n_files=1, directory=real_dir)


@pytest.mark.parametrize("n_files, pad", [(0, 0), (2, -1)])
def test_load_segment_invalid_arguments(real_dir, n_files, pad):
    with pytest.raises(ValueError):
        dio.load_segment("090052", n_files=n_files, pad_files=pad, directory=real_dir)


def test_load_segment_padding_inside_run(real_dir):
    seg = dio.load_segment("090102", n_files=2, pad_files=1, directory=real_dir)
    spf = config.SAMPLES_PER_FILE
    assert seg.data.shape[0] == 4 * spf
    assert (seg.pad_before, seg.pad_after) == (spf, spf)
    assert [f.stem for f in seg.pad_files] == ["090052", "090122"]
    np.testing.assert_array_equal(seg.core, dio.load_segment("090102", 2, directory=real_dir).data)
    # crop() of a decimated array removes the correspondingly decimated margin
    decimated = seg.data[::25]
    assert seg.crop(decimated, factor=25).shape[0] == seg.n_samples // 25


def test_load_segment_padding_truncated_at_run_edges(real_dir):
    seg = dio.load_segment("090052", n_files=5, pad_files=2, directory=real_dir)
    assert seg.pad_before == 0 and seg.pad_after == 0  # whole run B, no neighbours
    assert seg.core.shape[0] == 5 * config.SAMPLES_PER_FILE


def test_crop_rejects_indivisible_factor(real_dir):
    seg = dio.load_segment("090102", n_files=1, pad_files=1, directory=real_dir)
    with pytest.raises(ValueError):
        seg.crop(seg.data, factor=7)


def test_get_segment_spec():
    assert dio.get_segment_spec("seg_kuba")["start"] == "090452"
    with pytest.raises(KeyError):
        dio.get_segment_spec("does_not_exist")


# ---------------------------------------------------------------- synthetic
def _write_synthetic(directory: Path, sid: str, shape=(50, 3)) -> None:
    signal = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    gt = np.zeros(shape, dtype=np.int32)
    gt[10:20, 1] = 1
    np.save(directory / f"{sid}_signal.npy", signal)
    np.save(directory / f"{sid}_ground_truth.npy", gt)
    meta = {
        "axes": {"time_unit": "s", "space_unit": "m", "dt": 0.0016, "dx": 5.1},
        "signal": {"file": f"{sid}_signal.npy", "shape": list(shape), "dtype": "float32"},
        "ground_truth": {"file": f"{sid}_ground_truth.npy", "shape": list(shape),
                         "dtype": "int32", "background_value": 0,
                         "vehicles": [{"vehicle_id": 0, "type": "car", "direction": 1,
                                       "lane_number": 1, "marker_value": 1}]},
    }
    (directory / f"{sid}_metadata.json").write_text(json.dumps(meta))


def test_list_and_load_synthetic(tmp_path):
    for sid in ["10", "9", "40"]:
        _write_synthetic(tmp_path, sid)
    assert dio.list_synthetic_ids(tmp_path) == ["9", "10", "40"]
    signal, gt, meta = dio.load_synthetic("9", tmp_path)
    assert signal.shape == gt.shape == (50, 3)
    assert signal.dtype == np.float32
    assert dio.synthetic_axes(meta) == (0.0016, 5.1)


def test_load_synthetic_shape_mismatch(tmp_path):
    _write_synthetic(tmp_path, "1")
    meta_path = tmp_path / "1_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["signal"]["shape"] = [99, 3]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        dio.load_synthetic("1", tmp_path)


# ---------------------------------------------------------------- real dataset (integration)
_HAS_REAL = config.REAL_DIR.is_dir() and any(config.REAL_DIR.glob("*.npy"))
_HAS_SYNTH = config.SYNTH_DIR.is_dir() and any(config.SYNTH_DIR.glob("*_metadata.json"))


@pytest.mark.skipif(not _HAS_REAL, reason="real data not extracted")
def test_real_dataset_runs():
    files = dio.list_real_files()
    runs = dio.find_continuous_runs(files)
    assert len(files) == config.EXPECTED_REAL_FILES
    assert [(r[0].stem, r[-1].stem, len(r)) for r in runs] == [
        ("090017", "090227", 14), ("090252", "094332", 245)]


@pytest.mark.skipif(not _HAS_REAL, reason="real data not extracted")
@pytest.mark.parametrize("name", [s["name"] for s in config.SEGMENTS])
def test_configured_segments_load(name):
    seg = dio.load_named_segment(name, pad_files=1)
    assert seg.core.shape == (75000, 52)
    assert seg.duration_s == pytest.approx(120.0)
    assert seg.pad_before > 0 and seg.pad_after > 0  # both assigned segments have neighbours


@pytest.mark.skipif(not _HAS_SYNTH, reason="synthetic data not extracted")
def test_synthetic_dataset():
    ids = dio.list_synthetic_ids()
    assert len(ids) == config.EXPECTED_SYNTH_SAMPLES
    signal, gt, meta = dio.load_synthetic(ids[0])
    assert signal.shape == gt.shape == (75000, 59)
    assert dio.synthetic_axes(meta) == (config.DT, config.DX)
