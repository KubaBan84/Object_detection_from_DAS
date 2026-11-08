"""Data loading: real files, continuous segments, synthetic samples (plan: Etap 1)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np


def list_real_files(directory: Path | None = None) -> list[Path]:
    """Sorted list of real `HHMMSS.npy` files."""
    raise NotImplementedError


def parse_timestamp(path: Path) -> datetime:
    """`HHMMSS.npy` + RECORDING_DATE -> datetime."""
    raise NotImplementedError


def find_continuous_runs(files: list[Path]) -> list[list[Path]]:
    """Split files into runs where consecutive timestamps differ by exactly 10 s."""
    raise NotImplementedError


def load_real_file(path: Path) -> np.ndarray:
    """Load one file -> float32 array (time, space); validates shape (6250, n_ch)."""
    raise NotImplementedError


def load_segment(start: str, n_files: int = 12) -> tuple[np.ndarray, datetime]:
    """Concatenate `n_files` consecutive files starting at HHMMSS; raises if not continuous."""
    raise NotImplementedError


def list_synthetic_ids(directory: Path | None = None) -> list[str]:
    raise NotImplementedError


def load_synthetic(sample_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> (signal, ground_truth, metadata)."""
    raise NotImplementedError
