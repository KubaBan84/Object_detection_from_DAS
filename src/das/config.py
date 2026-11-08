"""Global constants, paths, analysed segments and algorithm parameters (plan: Stage 1).

All tunable parameters live in dataclasses so that every experiment can be described
by a single PipelineParams object (saved together with the results).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Acquisition metadata (real dataset) -------------------------------------------
DT: float = 0.0016                    # [s] time sampling interval
FS: float = 1.0 / DT                  # [Hz] = 625 Hz
DX: float = 5.106500953873407         # [m] channel spacing
SAMPLES_PER_FILE: int = 6250
FILE_DURATION_S: int = 10
RECORDING_DATE: str = "2024-05-07"
FILES_PER_SEGMENT: int = 12           # 12 x 10 s = 2 min

# --- Paths -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REAL_DIR = DATA_DIR / "real"
SYNTH_DIR = DATA_DIR / "synthetic"
ANNOT_FILE = DATA_DIR / "annotations" / "manual_labels.json"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
CACHE_DIR = OUTPUT_DIR / "cache"

# --- Analysed 2-minute segments (fill in from the course spreadsheet!) ---------------
# kind: "assigned" (from the spreadsheet) or "own" (chosen, non-overlapping)
SEGMENTS: list[dict] = [
    # {"name": "seg_A", "start": "HHMMSS", "n_files": FILES_PER_SEGMENT, "kind": "assigned"},
    # {"name": "seg_B", "start": "HHMMSS", "n_files": FILES_PER_SEGMENT, "kind": "own"},
    # {"name": "seg_C", "start": "HHMMSS", "n_files": FILES_PER_SEGMENT, "kind": "own"},
]


@dataclass
class PreprocessingParams:
    f_low: float = 1.0                # [Hz] band-pass lower cut-off (set from Stage 3)
    f_high: float = 50.0              # [Hz] band-pass upper cut-off (set from Stage 3)
    taper_hz: float = 0.5             # width of the raised-cosine transition band
    common_mode: str = "median"       # "median" | "mean" | "none"
    time_decimation: int = 25         # 0.0016 s * 25 = 0.04 s per image row
    envelope: str = "rms"             # "rms" | "abs_mean"
    space_upsample: int = 8           # interpolation factor along the cable
    log_compress: bool = True
    clip_percentile: float = 99.5


@dataclass
class EnhancementParams:
    v_min_kmh: float = 10.0           # f-k fan filter: min |apparent velocity|
    v_max_kmh: float = 150.0          # f-k fan filter: max |apparent velocity|
    fk_taper: float = 0.2
    kernel_length_px: int = 41        # oriented line filter length
    n_angles: int = 24                # angles sampled inside [v_min, v_max]
    bg_kernel_px: int = 51            # background (top-hat / median) kernel


@dataclass
class DetectionParams:
    strong_threshold: str = "otsu"    # "otsu" | sigma multiplier as string, e.g. "6"
    weak_low_sigma: float = 2.0       # hysteresis low threshold [robust sigma]
    weak_high_sigma: float = 4.0      # hysteresis high threshold [robust sigma]
    min_area_px: int = 150
    hough_threshold: int = 30
    hough_min_line_length: int = 60
    hough_max_line_gap: int = 20
    strong_mask_dilation_px: int = 7


@dataclass
class GroupingParams:
    tol_velocity_kmh: float = 8.0
    tol_time_s: float = 1.5
    min_channels: int = 8             # minimal spatial extent of an object [channels]


@dataclass
class TrackingParams:
    search_halfwidth_s: float = 1.0
    ransac_residual_s: float = 0.3
    velocity_window_channels: int = 5


@dataclass
class PipelineParams:
    pre: PreprocessingParams = field(default_factory=PreprocessingParams)
    enh: EnhancementParams = field(default_factory=EnhancementParams)
    det: DetectionParams = field(default_factory=DetectionParams)
    grp: GroupingParams = field(default_factory=GroupingParams)
    trk: TrackingParams = field(default_factory=TrackingParams)
