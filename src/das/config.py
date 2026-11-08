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

# --- Raw archives (Stage 0) ----------------------------------------------------------
REAL_ARCHIVE = DATA_DIR / "20240507.rar"
SYNTH_ARCHIVE = DATA_DIR / "Synthetic Dataset.zip"
EXPECTED_REAL_FILES: int = 259
EXPECTED_SYNTH_SAMPLES: int = 6

# --- Analysed 2-minute segments (plan section 1.1, pair work) ------------------------
# kind: "assigned" (from the course spreadsheet) or "own" (chosen, non-overlapping)
SEGMENTS: list[dict] = [
    {"name": "seg_kuba", "start": "090452", "n_files": FILES_PER_SEGMENT,
     "kind": "assigned", "owner": "Kuba"},
    {"name": "seg_mateusz", "start": "093222", "n_files": FILES_PER_SEGMENT,
     "kind": "assigned", "owner": "Mateusz"},
    # Chosen jointly after the EDA (Stage 2): highest traffic activity among windows that
    # do not overlap the assigned segments; many thin lines + crossing lines (both directions).
    {"name": "seg_own", "start": "091152", "n_files": FILES_PER_SEGMENT,
     "kind": "own", "owner": "Kuba + Mateusz"},
]


@dataclass
class PreprocessingParams:
    # Two processing paths chosen in Stage 3 (max line SNR after the f-k fan filter,
    # outputs/results/spectral_summary.json):
    #  HF - vibration energy: clean background, thick (tram/truck) and thin lines
    #  LF - quasi-static load: sharpest thin (car) lines, but blind to some lanes
    hf_low: float = 60.0              # [Hz] HF path band-pass (line SNR 5.84)
    hf_high: float = 312.5            # [Hz] (= Nyquist -> effectively a high-pass)
    lf_low: float = 1.0               # [Hz] LF path band-pass (line SNR 5.39)
    lf_high: float = 8.0              # [Hz]
    taper_hz: float = 1.0             # raised-cosine transition (<= 0.5 * low edge)
    detrend: bool = True              # per-channel mean + linear trend removal
    common_mode: str = "median"       # "median" | "mean" | "none"
    interpolate_bad_channels: bool = False  # EDA: flagged channel 25 is a physical hotspot
    time_decimation: int = 25         # 0.0016 s * 25 = 0.04 s per image row
    envelope: str = "rms"             # "rms" | "abs_mean" | "decimate" (negative reference)
    normalization: str = "log_ratio"  # "log_ratio" (log10 env/median) | "robust_z" (log1p z+)
    clip_low: bool = True             # clip log_ratio at 0 (= channel median)
    clip_percentile: float = 99.5     # upper clip
    smooth_sigma_t: float = 0.0       # [rows] anisotropic Gaussian along time (0 = off)
    smooth_sigma_x: float = 0.0       # [channels] along space (0 = off)
    clahe: bool = False               # optional CLAHE (skimage) - compared in Stage 4


@dataclass
class EnhancementParams:
    # Defaults = best combination of the Stage 4-5 variant study (scripts/run_enhancement.py,
    # outputs/results/enhancement_variants.csv): AUC light fused 0.962 -> 0.974.
    v_min_kmh: float = 5.0            # f-k fan filter: min |apparent velocity| (keeps slow vehicles)
    v_max_kmh: float = 150.0          # f-k fan filter: max |apparent velocity|
    fk_filter: bool = True            # False only for the ablation study
    fk_taper: float = 0.2             # relative width of the fan edges
    fk_pad_mode: str = "zero"         # "zero" | "reflect" | "tukey" (edge leakage, Stage 5)
    space_upsample: int = 8           # interpolation factor along the cable (after f-k)
    background: bool = False          # white top-hat with a horizontal line element
    bg_kernel_px: int = 101           # top-hat element length [upsampled px] (~65 m)
    enhancer: str = "filterbank"      # "filterbank" | "sato" | "meijering" | "gabor" | "none"
    kernel_length_px: int = 81        # oriented line filter length (along the line)
    kernel_width_sigma: float = 2.0   # Gaussian cross-profile of the line kernels [px]
    n_angles: int = 24                # angles sampled inside [v_min, v_max] per direction
    ridge_sigmas: tuple = (1.0, 2.0, 3.0)   # Sato / Meijering scales [px]
    gabor_wavelength_px: float = 12.0
    gabor_sigma_px: float = 6.0


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
