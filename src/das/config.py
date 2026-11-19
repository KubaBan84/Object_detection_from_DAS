"""Global constants, paths, analysed segments and algorithm parameters (plan: Stage 1).

All tunable parameters live in dataclasses, so that every run is described by a single
``PipelineParams`` object, which is saved together with the results.
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
    # Chosen jointly after the EDA (Stage 2): the highest traffic activity among the windows
    # that do not overlap the assigned segments, with traffic in both directions.
    {"name": "seg_own", "start": "091152", "n_files": FILES_PER_SEGMENT,
     "kind": "own", "owner": "Kuba + Mateusz"},
]


@dataclass
class PreprocessingParams:
    """Stage 4: band-pass, energy envelope, decimation, normalisation (preprocessing.preprocess)."""
    # Pass band chosen in Stage 3 (highest line SNR after the f-k fan filter,
    # outputs/results/spectral_summary.json): vibration energy of the vehicles, clean
    # background, thick tram / truck lines clearly visible.
    band_low: float = 60.0            # [Hz]
    band_high: float = 312.5          # [Hz] (= Nyquist -> effectively a high-pass)
    taper_hz: float = 1.0             # width of the raised-cosine band edges
    detrend: bool = True              # remove the mean and linear trend of every channel
    common_mode: str = "median"       # "median" | "none" (ablation)
    time_decimation: int = 25         # 0.0016 s * 25 = 0.04 s per image row
    envelope: str = "rms"             # "rms" | "decimate" (|x| of every 25th sample, ablation)
    normalization: str = "log_ratio"  # "log_ratio" = log10(env / channel median)
                                      # | "ratio" (no log compression, ablation)
    clip_low: bool = True             # clip at 0 (= the channel median)
    clip_percentile: float = 99.5     # upper clip


@dataclass
class EnhancementParams:
    """Stage 5: f-k fan filter per direction and directional filter bank (enhancement.enhance)."""
    v_min_kmh: float = 5.0            # f-k fan: min |apparent velocity| (keeps slow vehicles)
    v_max_kmh: float = 150.0          # f-k fan: max |apparent velocity|
    fk_filter: bool = True            # False only for the ablation
    fk_taper: float = 0.2             # relative width of the fan edges
    space_upsample: int = 8           # interpolation factor along the cable (after f-k)
    enhancer: str = "filterbank"      # "filterbank" | "sato" | "none" (method comparison)
    kernel_length_px: int = 81        # length of the oriented line kernels (~3.2 s)
    kernel_width_sigma: float = 2.0   # Gaussian cross-profile of the kernels [px]
    n_angles: int = 24                # kernel angles inside [v_min, v_max] per direction
    ridge_sigmas: tuple = (1.0, 2.0, 3.0)   # Sato scales [px]


@dataclass
class DetectionParams:
    """Stage 6: detection of the thick lines (detection.detect_lines)."""
    # Pixel units refer to the Stage-5 response grid: 1 row = 0.04 s, 1 col = 0.64 m.
    threshold: str = "otsu"           # "otsu" | sigma multiplier as string, e.g. "6"
    # floor of the threshold [robust sigma]: without it, Otsu in a direction without heavy
    # traffic separates only noise texture; 4 (not 6) so that wide, diffuse lines are kept
    min_sigma: float = 4.0
    open_length_px: int = 41          # line structuring element of the oriented opening
    open_angles: int = 9              # number of element slopes inside [v_min, v_max]
    min_area_px: int = 1500           # minimal component area after the opening
    min_extent_m: float = 40.0        # minimal extent of a component along the cable
    slope_margin: float = 0.1         # relative tolerance of the velocity fan for line slopes
    ridge_nms_rows: int = 5           # ridge non-maximum suppression along time (0.2 s)
    ransac_residual_px: float = 3.0   # RANSAC inlier distance [px]
    ransac_max_trials: int = 200
    max_lines: int = 4                # sequential RANSAC: max lines per connected component
    parallel_overlap: float = 0.5     # min. x-overlap of a 2nd line in one component
    seed: int = 0


@dataclass
class GroupingParams:
    """Stage 7: linking of line candidates into objects, clean-up (grouping.build_objects)."""
    # Candidates of one vehicle are linked when, at the midpoint between them, their lines
    # are closer than tol_time_s in time and their velocities agree (absolute OR relative).
    tol_velocity_kmh: float = 8.0
    tol_velocity_rel: float = 0.15
    tol_time_s: float = 1.5
    cont_max_gap_m: float = 20.0      # continuation link: max gap between the x-ranges
    cont_max_overlap_m: float = 20.0  # ... max overlap of the x-ranges
    cont_max_ratio: float = 2.0       # ... max ratio of the velocities
    min_channels: int = 8             # minimal spatial extent of an object [channels]
    v_min_kmh: float = 5.0            # plausibility filter (same fan as Stage 5)
    v_max_kmh: float = 150.0
    dup_overlap: float = 0.5          # min. overlap (fraction of the shorter) for a duplicate
    fit_iterations: int = 2           # robust re-weighting passes of the group line fit


@dataclass
class EvaluationParams:
    """Stage 9: matching of detections with reference objects."""
    tol_t_s: float = 1.5              # matching gate: time gap at the detection centre [s]
    tol_v_kmh: float = 20.0           # matching gate: velocity difference [km/h]
    # manual labels of real data (rough clicks): a label is scored if its velocity is
    # inside the fan searched by the detector (Stage 5) and it is long enough
    ref_v_range_kmh: tuple[float, float] = (5.0, 150.0)
    ref_min_extent_m: float = 30.0
    # an unmatched detection is "near a label" if it runs within fp_near_s of a labelled
    # line on at least fp_min_channels channels (fragment / duplicate of a vehicle)
    fp_near_s: float = 1.0
    fp_min_channels: int = 3


@dataclass
class PipelineParams:
    """All parameters of the method; the defaults are the final values (Stage 11)."""
    pre: PreprocessingParams = field(default_factory=PreprocessingParams)
    enh: EnhancementParams = field(default_factory=EnhancementParams)
    det: DetectionParams = field(default_factory=DetectionParams)
    grp: GroupingParams = field(default_factory=GroupingParams)
    ev: EvaluationParams = field(default_factory=EvaluationParams)
