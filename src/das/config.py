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
    """Stage 4: band-pass paths, envelope, decimation, normalisation (preprocessing.preprocess)."""
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
                                      # | "ratio" (no log compression, ablation)
    clip_low: bool = True             # clip log_ratio at 0 (= channel median)
    clip_percentile: float = 99.5     # upper clip
    smooth_sigma_t: float = 0.0       # [rows] anisotropic Gaussian along time (0 = off)
    smooth_sigma_x: float = 0.0       # [channels] along space (0 = off)
    clahe: bool = False               # optional CLAHE (skimage) - compared in Stage 4


@dataclass
class EnhancementParams:
    """Stage 5: f-k fan filter per direction and directional filter bank (enhancement.enhance)."""
    # Defaults = best combination of the Stage 4-5 variant study (scripts/run_enhancement.py,
    # outputs/results/enhancement_variants.csv): AUC light fused 0.962 -> 0.974.
    v_min_kmh: float = 5.0            # f-k fan filter: min |apparent velocity| (keeps slow vehicles)
    v_max_kmh: float = 150.0          # f-k fan filter: max |apparent velocity|
    fk_filter: bool = True            # False only for the ablation study
    fk_taper: float = 0.2             # relative width of the fan edges
    fk_pad_mode: str = "zero"         # "zero" | "reflect" | "tukey" | "none" (edge leakage)
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
    """Stage 6: strong-line path 6A and weak-line path 6B (detection.detect_candidates)."""
    # Pixel units refer to the Stage-5 response grid: 1 row = 0.04 s, 1 col = 0.64 m.
    # --- 6A: strong (thick) lines -------------------------------------------------------
    strong_bands: tuple = ("hf",)     # maps on which the strong path runs
    strong_threshold: str = "otsu"    # "otsu" | sigma multiplier as string, e.g. "6"
    # floor of the strong threshold [robust sigma]; 4 (not 6) so that wide, diffuse lines on
    # real data reach path 6A - their skeleton in 6B is branched (Stage 6 study)
    strong_min_sigma: float = 4.0
    strong_open_length_px: int = 41   # line structuring element for the oriented opening
    strong_open_angles: int = 9       # number of element slopes inside [v_min, v_max]
    strong_min_area_px: int = 1500    # minimal component area after the opening
    min_extent_m: float = 40.0        # minimal extent along the cable (both paths)
    slope_margin: float = 0.1         # relative tolerance of the velocity fan for line slopes
    ransac_residual_px: float = 3.0   # RANSAC inlier distance of column centroids [px]
    ransac_max_trials: int = 200
    strong_max_lines: int = 4         # sequential RANSAC: max lines per connected component
    strong_ridge_nms_rows: int = 5    # ridge non-maximum suppression along time (0.2 s)
    strong_parallel_overlap: float = 0.5  # min. x-overlap of a 2nd line in one component
    seed: int = 0
    # --- 6B: weak (thin) lines ----------------------------------------------------------
    weak_bands: tuple = ("hf", "lf")  # maps on which the weak path runs
    mask_strong: bool = True          # exclude strong objects before the weak path
    strong_mask_dilation_rows: int = 12   # half-size of the exclusion halo along time (0.5 s)
    strong_mask_dilation_cols: int = 6    # ... and along space (~4 m)
    weak_low_sigma: float = 2.0       # hysteresis low threshold [robust sigma of background]
    weak_high_sigma: float = 4.0      # hysteresis high threshold [robust sigma]
    min_area_px: int = 150            # remove_small_objects before skeletonisation
    use_hough_p: bool = True          # cv2.HoughLinesP on the skeleton (local segments)
    hough_threshold: int = 30         # accumulator votes
    hough_min_line_length: int = 60   # [px]
    hough_max_line_gap: int = 20      # [px]
    hough_theta_deg: float = 0.5      # angular resolution of both Hough transforms
    orientation_tol_deg: float = 10.0 # max |segment angle - local filter-bank angle| (0 = off)
    use_hough_global: bool = True     # skimage hough_line + hough_line_peaks
    global_min_votes: int = 80        # minimal number of skeleton pixels on a global line
    global_max_peaks: int = 60
    global_min_distance: int = 15     # [px] between peaks (rho)
    global_min_angle: int = 3         # [theta bins] between peaks
    global_support_px: float = 2.0    # skeleton pixels closer than this support a global line
    global_max_gap_px: int = 60       # gaps along a global line bridged [rows] (collinear)


@dataclass
class GroupingParams:
    """Stage 7: linking of line candidates into objects, clean-up (grouping.build_objects)."""
    # Candidates of one vehicle are linked when, at the midpoint between them, their lines
    # are closer than tol_time_s in time and their velocities agree (absolute OR relative).
    tol_velocity_kmh: float = 8.0
    tol_velocity_rel: float = 0.15
    # 1.5 s (Stage 13): synthetic F1 unchanged, real fragments next to strong lines merged
    # (real F1 vs the manual labels 0.721 -> 0.788); 1.0 s left 19 of 24 real FP on labelled lines
    tol_time_s: float = 1.5
    cont_max_gap_m: float = 20.0      # continuation link: max gap between the x-ranges
    cont_max_overlap_m: float = 20.0  # ... max overlap of the x-ranges
    cont_max_ratio: float = 2.0       # ... max ratio of the velocities
    band_weights: tuple = (("hf", 1.0), ("lf", 2.0))  # weight of a band's points in the fit
    min_channels: int = 8             # minimal spatial extent of an object [channels]
    # A weak object must be long OR clearly above the noise: short AND faint objects are
    # typical of the oblique noise texture (Stage 7 statistics, docs/experiments_log.md).
    weak_min_support_m: float = 60.0  # weak objects: cable length covered by candidates ...
    weak_short_min_score: float = 5.0 # ... or else a mean response of at least this [sigma]
    v_min_kmh: float = 5.0            # plausibility filter (same fan as Stage 5)
    v_max_kmh: float = 150.0
    dup_overlap: float = 0.5          # min. overlap (fraction of the shorter) for a duplicate
    fit_iterations: int = 2           # robust re-weighting passes of the group line fit


@dataclass
class TrackingParams:
    """Stages 8-9: ridge tracking, track velocity and v(t) (tracking.track_objects)."""
    # --- Stage 9: ridge tracking (one passage time per cable channel) ---------------------
    track_map: str = "fk"             # "fk": f-k filtered direction map (no angular smoothing)
                                      # "response": Stage-5 filter-bank response
    track_band: str = "sum"           # "sum" of the HF and LF z maps | "hf" | "lf" | "auto"
                                      # (auto = LF if the object was detected in LF, else HF)
    smooth_rows: float = 1.0          # Gaussian smoothing of the map along time [rows] (0 = off)
    search_halfwidth_s: float = 1.0   # pass 1: window around the averaged line (Stage 7)
    refine_halfwidth_s: float = 0.5   # pass 2: window around the robust polynomial of pass 1
    peak_method: str = "parabola"     # sub-row peak: "parabola" | "centroid" (upper half of peak)
    centroid_rows: int = 6            # half-width of the centroid window [rows]
    extend_m: float = 30.0            # track this far beyond the detected extent
    min_peak_z: float = 1.5           # a channel's peak must exceed this robust z
    edge_margin_s: float = 0.5        # peaks this close to the first / last row are biased
                                      # (ridge continues beyond the image) -> dropped
    max_gap_channels: int = 3         # longest run of channels with gaps <= this is kept
    min_points: int = 6               # fewer tracked channels -> no track (line only)
    ransac_residual_s: float = 0.3    # inlier distance of the trajectory RANSAC [s]
    ransac_trials: int = 200
    poly_order: int = 2               # t(x) polynomial of the trajectory RANSAC
    seed: int = 0
    # --- velocity profile v(t) and speed variability ---------------------------------------
    velocity_window_channels: int = 9 # local regression window (9 ch = 46 m), Stage 9 study
    varying_dv_rel: float = 0.15      # varying speed: |v_end - v_start| >= 15 % of |v| ...
    varying_rms_gain: float = 0.3     # ... and the quadratic x(t) cuts the RMS residual by 30 %


@dataclass
class FeatureParams:
    """Stage 10: per-object descriptors (features.object_features)."""
    profile_halfwidth_s: float = 2.0  # straightened cross-profile: +-window around t(x) [s]
    peak_search_rows: int = 3         # ridge maximum searched +-rows around the trajectory
    baseline_quantile: float = 0.2    # per-channel baseline of the cross-profile
    subline_rel_height: float = 0.5   # sub-line = local maximum >= this x main peak ...
    subline_min_dip: float = 0.2      # ... separated from it by a dip of >= this x main peak
    subline_smooth_rows: float = 1.0  # Gaussian smoothing of the mean cross-profile [rows]
    spec_window_s: float = 0.8        # raw-signal window on the trajectory for the spectrum
    spec_lf_band: tuple = (1.0, 8.0)      # [Hz] same bands as the two image paths (Stage 3)
    spec_hf_band: tuple = (60.0, 312.5)
    min_channels: int = 4             # fewer channels on the trajectory -> NaN features


# Features used for clustering, in groups = the properties named in the brief (names from
# features.FEATURE_NAMES): (group, features, name of the high end, name of the low end).
# Chosen with the per-feature separability study on synthetic data (Stage 10): line
# energy / HF SNR separate trams from cars (AUC 0.97-0.99) much better than the FWHM
# thickness (0.64) - a line *looks* thick because it is strong; dv_rel / |a| separate
# varying from constant speed (AUC 0.90 / 0.80) and are used as logarithms (heavy tails).
# Spectrum shape (LF/HF ratio, centroid) and sub-lines carry little information.
CLUSTER_FEATURE_GROUPS: tuple = (
    ("thickness", ("log_peak_z", "log_area_zs", "snr_hf_db"), "thick", "thin"),
    ("speed change", ("log_dv_rel", "log_abs_accel", "log_v_std"), "varying speed",
     "constant speed"),
)
_EXTRA_GROUPS: tuple = (
    ("velocity", ("abs_v_kmh",), "fast", "slow"),
    ("spectrum", ("lf_hf_db", "spec_centroid_hz"), "low-frequency", "high-frequency"),
    ("shape", ("thickness_s", "tail_asym", "n_sublines"), "wide", "narrow"),
)
# feature-set / method study: name -> (groups, mode, equal group weights, forced k)
CLUSTER_STUDY: dict = {
    "factorized: thickness x speed change (default)": (CLUSTER_FEATURE_GROUPS, "factorized", True, 0),
    "factorized, k = 2 per property": (CLUSTER_FEATURE_GROUPS, "factorized", True, 2),
    "joint k-means, k by silhouette": (CLUSTER_FEATURE_GROUPS, "joint", True, 0),
    "joint k-means, k = 4": (CLUSTER_FEATURE_GROUPS, "joint", True, 4),
    "joint k = 4, linear speed-change features": (
        (CLUSTER_FEATURE_GROUPS[0], ("speed change", ("dv_rel", "abs_accel_ms2", "v_std_kmh"),
                                     "varying speed", "constant speed")), "joint", True, 4),
    "joint k = 4, + FWHM thickness": (
        (("thickness", ("thickness_s", "log_peak_z", "log_area_zs", "snr_hf_db"), "thick",
          "thin"), CLUSTER_FEATURE_GROUPS[1]), "joint", True, 4),
    "joint, + velocity, spectrum, shape": (CLUSTER_FEATURE_GROUPS + _EXTRA_GROUPS, "joint",
                                           True, 0),
    "thickness only": (CLUSTER_FEATURE_GROUPS[:1], "joint", True, 0),
    "speed change only": (CLUSTER_FEATURE_GROUPS[1:], "joint", True, 0),
}


@dataclass
class ClusteringParams:
    """Stage 10: k-means of the object features (cv2.kmeans)."""
    feature_groups: tuple = CLUSTER_FEATURE_GROUPS
    # "factorized": k-means separately on every property group (thick/thin, constant/
    # varying speed), clusters = combinations of the group labels; "joint": one k-means
    mode: str = "factorized"
    group_weights: bool = True        # joint mode: equal total weight of every group
    k_min: int = 2
    k_max: int = 6                    # joint mode: k in [k_min, k_max] by silhouette
    group_k_max: int = 3              # factorized mode: k per group in [k_min, group_k_max]
    k: int = 0                        # forced k (joint) / k per group (factorized); 0 = silhouette
    robust_scale: bool = False        # z-score (the log features are close to symmetric)
    clip_z: float = 4.0               # clip standardised features to +-clip_z (0 = off)
    attempts: int = 10                # cv2.kmeans restarts (KMEANS_PP_CENTERS)
    max_iter: int = 100
    seed: int = 0
    name_z: float = 0.5               # joint mode: |mean z of a group| above this -> named


@dataclass
class EvaluationParams:
    """Stage 11: object matching and trajectory overlap."""
    tol_t_s: float = 1.5              # matching gate: time gap at the detection centre [s]
    tol_v_kmh: float = 20.0           # matching gate: velocity difference [km/h]
    iou_tol_s: float = 0.5            # trajectory IoU: tolerance along time [s]
    # manual labels of real data (rough clicks): a label is *evaluable* if its velocity is
    # inside the fan searched by the detector (Stage 5) and it is long enough
    ref_v_range_kmh: tuple[float, float] = (5.0, 150.0)
    ref_min_extent_m: float = 30.0
    # a false positive "near a label" runs within fp_near_s of a labelled line on at least
    # fp_min_channels channels (fragment / duplicate of a vehicle or a gate failure)
    fp_near_s: float = 1.0
    fp_min_channels: int = 3


@dataclass
class PipelineParams:
    """All parameters of the method; the defaults are the final values of Stage 13."""
    pre: PreprocessingParams = field(default_factory=PreprocessingParams)
    enh: EnhancementParams = field(default_factory=EnhancementParams)
    det: DetectionParams = field(default_factory=DetectionParams)
    grp: GroupingParams = field(default_factory=GroupingParams)
    trk: TrackingParams = field(default_factory=TrackingParams)
    feat: FeatureParams = field(default_factory=FeatureParams)
    clu: ClusteringParams = field(default_factory=ClusteringParams)
    ev: EvaluationParams = field(default_factory=EvaluationParams)
