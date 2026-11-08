"""Data loading: real files, continuous segments, synthetic samples (plan: Stage 1).

Real dataset
    259 files ``HHMMSS.npy`` recorded on 2024-05-07. Each file holds 10 s of data as a
    float32 matrix of shape (6250, n_channels) - axis 0 = time (dt = 1.6 ms), axis 1 =
    position along the fibre (dx = 5.1 m). The files are **not** continuous, so every
    multi-file segment is validated with :func:`find_continuous_runs`.

Synthetic dataset
    ``{id}_signal.npy`` (time, space) float32, ``{id}_ground_truth.npy`` int32 marker mask
    of the same shape and ``{id}_metadata.json`` with sampling intervals and vehicle info.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import config

_REAL_NAME = re.compile(r"^(\d{6})\.npy$")
_SYNTH_META = re.compile(r"^(.+)_metadata\.json$")


# ======================================================================================
# Real data - files and timestamps
# ======================================================================================
def list_real_files(directory: Path | str | None = None) -> list[Path]:
    """Sorted list of real ``HHMMSS.npy`` files (other files are ignored).

    Lexicographic order of ``HHMMSS`` equals chronological order (single recording day).
    """
    directory = Path(directory) if directory is not None else config.REAL_DIR
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Real data directory not found: {directory} (run scripts/extract_data.py)")
    return sorted(p for p in directory.iterdir() if p.is_file() and _REAL_NAME.match(p.name))


def _stem(name: Path | str) -> str:
    """'090452.npy' / Path(.../090452.npy) / '090452' -> '090452'."""
    name = Path(name).name
    return name[:-4] if name.endswith(".npy") else name


def parse_timestamp(path: Path | str) -> datetime:
    """``HHMMSS.npy`` (or bare ``HHMMSS``) + RECORDING_DATE -> naive local ``datetime``."""
    stem = _stem(path)
    if not re.fullmatch(r"\d{6}", stem):
        raise ValueError(f"Not a HHMMSS file name: {Path(path).name!r}")
    # strptime also rejects impossible times such as 256000
    return datetime.strptime(f"{config.RECORDING_DATE} {stem}", "%Y-%m-%d %H%M%S")


def find_continuous_runs(files: list[Path], step_s: float = config.FILE_DURATION_S
                         ) -> list[list[Path]]:
    """Split files into runs in which consecutive timestamps differ by exactly ``step_s``.

    The input does not need to be sorted. Returns runs in chronological order.
    """
    if not files:
        return []
    files = sorted((Path(f) for f in files), key=parse_timestamp)
    t0 = parse_timestamp(files[0])
    seconds = np.array([(parse_timestamp(f) - t0).total_seconds() for f in files])
    breaks = np.flatnonzero(np.diff(seconds) != step_s) + 1
    bounds = [0, *breaks.tolist(), len(files)]
    return [files[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def load_real_file(path: Path | str, expected_samples: int = config.SAMPLES_PER_FILE
                   ) -> np.ndarray:
    """Load one real file -> float32 array (time, space).

    Raises ``ValueError`` if the array is not 2-D, has an unexpected number of time
    samples or is not floating point.
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[0] != expected_samples:
        raise ValueError(f"{Path(path).name}: expected shape ({expected_samples}, n_channels), "
                         f"got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{Path(path).name}: expected a floating dtype, got {arr.dtype}")
    return arr.astype(np.float32, copy=False)


# ======================================================================================
# Real data - continuous 2-minute segments
# ======================================================================================
@dataclass
class Segment:
    """A continuous block of real data, optionally padded with neighbouring files.

    ``data`` contains the padding; ``core`` is the requested segment only. Padding exists
    so that FFT-based filtering does not distort the segment edges - process ``data`` and
    then remove the margin with :meth:`crop`.
    """
    data: np.ndarray                     # (time, space) including padding
    start: datetime                      # start time of the core (unpadded) segment
    files: list[Path]                    # core files
    pad_before: int = 0                  # padding samples before the core
    pad_after: int = 0                   # padding samples after the core
    dt: float = config.DT
    dx: float = config.DX
    name: str = ""
    pad_files: list[Path] = field(default_factory=list)

    @property
    def core(self) -> np.ndarray:
        """View of the requested segment without padding."""
        return self.crop(self.data)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0] - self.pad_before - self.pad_after

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples * self.dt

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)

    def crop(self, arr: np.ndarray, factor: int = 1) -> np.ndarray:
        """Remove the padding from an array derived from ``data`` along the time axis.

        ``factor`` is the time-decimation factor applied to ``arr`` (e.g. 25 after the
        RMS envelope); the padding must be divisible by it.
        """
        if self.pad_before % factor or self.pad_after % factor:
            raise ValueError(f"Padding ({self.pad_before}, {self.pad_after}) "
                             f"is not divisible by factor {factor}")
        b, a = self.pad_before // factor, self.pad_after // factor
        return arr[b:arr.shape[0] - a]

    def time_axis(self) -> np.ndarray:
        """Seconds from ``start`` for every sample of ``core``."""
        return np.arange(self.n_samples) * self.dt

    def distance_axis(self) -> np.ndarray:
        """Metres along the fibre for every channel (relative to the first channel)."""
        return np.arange(self.n_channels) * self.dx


def resolve_segment_files(start: str, n_files: int = config.FILES_PER_SEGMENT,
                          pad_files: int = 0, directory: Path | str | None = None
                          ) -> tuple[list[Path], list[Path], list[Path]]:
    """Find the files of a segment without loading them -> (before, core, after).

    Raises ``FileNotFoundError`` if the start file does not exist and ``ValueError`` if
    the segment does not fit into one continuous run. Padding is taken only from the same
    continuous run, so it may be shorter than ``pad_files`` at the edge of a run.
    """
    if n_files < 1:
        raise ValueError(f"n_files must be >= 1, got {n_files}")
    if pad_files < 0:
        raise ValueError(f"pad_files must be >= 0, got {pad_files}")
    stem = _stem(start)
    files = list_real_files(directory)
    for run in find_continuous_runs(files):
        stems = [f.stem for f in run]
        if stem not in stems:
            continue
        i0 = stems.index(stem)
        if i0 + n_files > len(run):
            raise ValueError(
                f"Segment {stem} + {n_files} files does not fit into the continuous run "
                f"{run[0].stem}-{run[-1].stem} (only {len(run) - i0} files left; "
                f"the next file after {run[-1].stem} is missing or not consecutive)")
        n_before = min(pad_files, i0)
        n_after = min(pad_files, len(run) - (i0 + n_files))
        return run[i0 - n_before:i0], run[i0:i0 + n_files], run[i0 + n_files:i0 + n_files + n_after]
    raise FileNotFoundError(f"No file {stem}.npy in {Path(directory or config.REAL_DIR)}")


def load_segment(start: str, n_files: int = config.FILES_PER_SEGMENT, pad_files: int = 0,
                 directory: Path | str | None = None, name: str = "") -> Segment:
    """Load ``n_files`` consecutive files starting at ``HHMMSS`` as one :class:`Segment`.

    With the defaults this is one 2-minute segment of shape (75000, n_channels).
    ``pad_files`` extra files are loaded on each side when available (see ``Segment.crop``).
    """
    before, core, after = resolve_segment_files(start, n_files, pad_files, directory)
    arrays = [load_real_file(f) for f in (*before, *core, *after)]
    n_channels = {a.shape[1] for a in arrays}
    if len(n_channels) != 1:
        raise ValueError(f"Files of segment {_stem(start)} differ in channel count: {n_channels}")
    return Segment(
        data=np.concatenate(arrays, axis=0),
        start=parse_timestamp(core[0]),
        files=core,
        pad_before=len(before) * config.SAMPLES_PER_FILE,
        pad_after=len(after) * config.SAMPLES_PER_FILE,
        name=name or _stem(start),
        pad_files=[*before, *after],
    )


def get_segment_spec(name: str) -> dict:
    """Entry of ``config.SEGMENTS`` with the given name."""
    for spec in config.SEGMENTS:
        if spec["name"] == name:
            return spec
    known = ", ".join(s["name"] for s in config.SEGMENTS) or "(none)"
    raise KeyError(f"Unknown segment {name!r}; configured: {known}")


def load_named_segment(name: str, pad_files: int = 0,
                       directory: Path | str | None = None) -> Segment:
    """Load a segment defined in ``config.SEGMENTS`` by its name."""
    spec = get_segment_spec(name)
    return load_segment(spec["start"], spec.get("n_files", config.FILES_PER_SEGMENT),
                        pad_files=pad_files, directory=directory, name=name)


# ======================================================================================
# Synthetic data
# ======================================================================================
def list_synthetic_ids(directory: Path | str | None = None) -> list[str]:
    """Ids of all synthetic samples (taken from ``{id}_metadata.json``), numeric order."""
    directory = Path(directory) if directory is not None else config.SYNTH_DIR
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Synthetic data directory not found: {directory} (run scripts/extract_data.py)")
    ids = [m.group(1) for p in directory.iterdir() if (m := _SYNTH_META.match(p.name))]
    return sorted(ids, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))


def load_synthetic_metadata(sample_id: str, directory: Path | str | None = None) -> dict:
    directory = Path(directory) if directory is not None else config.SYNTH_DIR
    with open(directory / f"{sample_id}_metadata.json", encoding="utf-8") as fh:
        return json.load(fh)


def synthetic_axes(meta: dict) -> tuple[float, float]:
    """(dt [s], dx [m]) of a synthetic sample."""
    return float(meta["axes"]["dt"]), float(meta["axes"]["dx"])


def load_synthetic(sample_id: str, directory: Path | str | None = None
                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> (signal float32 (time, space), ground_truth int (time, space), metadata).

    Shapes are validated against the metadata and against each other.
    """
    directory = Path(directory) if directory is not None else config.SYNTH_DIR
    meta = load_synthetic_metadata(sample_id, directory)
    # Path(...).name guards against paths in the metadata pointing outside the directory
    signal = np.load(directory / Path(meta["signal"]["file"]).name, allow_pickle=False)
    gt = np.load(directory / Path(meta["ground_truth"]["file"]).name, allow_pickle=False)
    for label, arr, key in (("signal", signal, "signal"), ("ground_truth", gt, "ground_truth")):
        expected = tuple(meta[key]["shape"])
        if arr.shape != expected:
            raise ValueError(f"Sample {sample_id}: {label} shape {arr.shape} != metadata {expected}")
    return signal.astype(np.float32, copy=False), gt, meta
