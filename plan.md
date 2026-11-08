# Implementation plan – Detecting Moving Objects in DAS Recordings (grade 5 version)

> Code and report deadline: **22.11.2026, 23:59**. Presentation: **26.11.2026 / 01.12.2026**.
> The plan covers the full grade-5 scope: real data, detection of **thick and thin lines**, and velocities.
> It also covers both extra points: the **v(t) plot** and **clustering of objects**.

---

## 0. Grade-5 requirements and where they are implemented

| # | Requirement from the project brief | Stage | Module / file |
|---|---|---|---|
| 1 | Load **real data**; 3 continuous 2-min segments | 1 | `das/io.py`, `config.SEGMENTS` |
| 2 | Exploratory analysis: statistics, distributions, variability in time and space | 2 | `das/stats.py`, `scripts/run_eda.py` |
| 3 | Frequency analysis: **FFT applied independently to each channel**, STFT | 3 | `das/spectral.py`, `scripts/run_spectral.py` |
| 4 | Noise filtering: frequency band, downsampling, filters, standardization | 4 | `das/preprocessing.py` |
| 5 | Processing so that **only moving objects** are visible | 5 | `das/enhancement.py` |
| 6 | Detection of **thick lines (tram/truck)** | 6A | `das/detection.py` |
| 7 | Detection of **thin, weak lines (cars)** – **grade-5 condition** | 6B | `das/detection.py` |
| 8 | Grouping of candidates and averaging the line within a group | 7 | `das/grouping.py` |
| 9 | Velocity of each object (km/h, direction) | 8 | `das/velocity.py` |
| 10 | (Extra 1) plot of **velocity over time** | 9 | `das/tracking.py` |
| 11 | (Extra 2) **clustering** of objects (thick/thin, constant/varying speed, shape) | 10 | `das/features.py`, `das/clustering.py` |
| 12 | Assessment of the number and accuracy of detected objects | 11 | `das/evaluation.py` |
| 13 | Block diagram of the algorithm with step descriptions | 14 | `docs/flowchart.md` → `report/figures/flowchart.png` |
| 14 | "Before/after" visualizations for each step, including removed noise | 4–6, 14 | `das/visualization.py` |
| 15 | Detailed results for 3 two-minute segments | 12 | `scripts/run_pipeline.py` |
| 16 | Description of experiments (successful and unsuccessful), errors and limitations | 13 | `docs/experiments_log.md` |
| 17 | Sources of data, code and inspiration | 14 | `docs/references.md` |
| 18 | Code without errors or warnings, vectorized, readable | all | `tests/`, `pyproject.toml` (`filterwarnings=error`) |

**Technology constraints.** The algorithm may only use **NumPy, OpenCV, scikit-image, Matplotlib and Pillow**.
Neural networks are not allowed.

- `scipy` is **not** on the list of allowed libraries. Band-pass filters, STFT and Welch are written with `numpy.fft`.
  Alternative: ask the instructor by e-mail for permission to use `scipy.signal` and record the answer in the report.
- `pandas` and `sklearn` are not on the list either. Tables are saved with the `csv`/`json` modules.
  K-means uses `cv2.kmeans`, and clustering metrics are computed in NumPy.
- Jupyter, nbconvert and pytest are tooling only, not part of the algorithm.

---

## 1. Data facts (verified on the files in `data/`)

| | Real data | Synthetic data |
|---|---|---|
| Archive | `data/20240507.rar` (259 files) | `data/Synthetic Dataset.zip` (6 samples × 3 files) |
| File | `HHMMSS.npy`, shape **(6250, 52)**, float32 | `{id}_signal.npy` **(75000, 59)**, float32 + `{id}_ground_truth.npy` int32 + `{id}_metadata.json` |
| Layout | axis 0 = time, axis 1 = space | same (`matrix_layout: "time_index, space_index"`) |
| dt / fs | 0.0016 s / **625 Hz** (Nyquist 312.5 Hz) | same (`axes.dt`) |
| dx | 5.1065 m → 52 channels ≈ 260 m | same (`axes.dx`) |
| Values | order of 1e-6 (std ≈ 1e-6, extremes ≈ ±3e-5) | – |
| Continuous runs | **090017–090227** (14 files, 140 s) and **090252–094332** (245 files, 2450 s) | each sample = 2 min |
| GT | none | mask with vehicle markers + `type` (car/tram/van…), `direction`, `lane_number` |

Design conclusions:

- A 2-minute segment is **12 consecutive files** = 75000 × 52 samples.
  Each of the 3 segments must lie entirely within one continuous run. `io.load_segment` checks this and raises an exception.
- For vehicle speeds of 10–150 km/h (2.8–42 m/s), crossing one channel takes 0.12–1.8 s.
  A car at 50 km/h crosses the whole cable section in about 19 s.
  The full 1.6 ms time resolution is therefore unnecessary for the trajectory image.
  Decimation ×25 gives 0.04 s per row, i.e. 3000 rows per 2 minutes.
- The image has only 52 columns, so lines are almost vertical. We therefore interpolate the space axis ×8 (416 columns, 0.64 m/px).
  Then 50 km/h gives an angle of about 40–45°, which greatly improves the angular resolution of Hough and the directional filters.
- Image convention as in the brief's figure: **time vertical (downwards), space horizontal**.
  A line with velocity v has slope Δcol/Δrow = v·row_dt/col_dx.

---

## 2. Code conventions (apply at every stage)

- All logic lives in `src/das/`. Scripts in `scripts/` only parse arguments, call the library and save results.
- **Every parameter** goes into a dataclass in `config.py`. Results are saved together with the parameters (JSON) so that every experiment is reproducible.
- **No per-pixel loops.** We use vectorized NumPy operations, `cv2.filter2D`, `np.fft`, `sliding_window_view` and `np.add.reduceat`/`reshape` for decimation.
  Loops are allowed only over objects, kernels and files.
- Processing functions return a **dictionary of intermediate stages** (`stages`), because the report requires "before/after" visualizations.
- Type hints, docstrings and comments in English. The plan and the experiments log are also in English.
- No runtime warnings:
  - pytest runs with `filterwarnings=error`;
  - final script runs use `python -W error`;
  - watch out for division by zero (MAD=0), `float64`→`uint8` in skimage and log of zero (add eps).
- The random seed (`seed`) for RANSAC and k-means is kept in the config.
- The environment is managed with **uv**: Python 3.12 (`.python-version`), dependencies in `pyproject.toml`, versions locked in `uv.lock` (versioned).
  - `uv sync` creates `.venv` with dependencies, the `dev` group and the `das` package (editable).
  - Running: `uv run python scripts/<script>.py`, `uv run pytest`.
  - New dependencies: `uv add <pkg>` (allowed libraries only!) or `uv add --dev <pkg>` for tooling.
  - After every dependency change, refresh `requirements.txt` for people without uv: `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`.

---

## 3. Implementation stages

Each stage has: **goal → files → tasks → artifacts → definition of done (DoD)**.

### Stage 0 – Environment and data

**Files:** `requirements.txt`, `pyproject.toml`, `scripts/extract_data.py`, `data/README.md`.

1. ✅ uv environment initialized (`uv sync`): numpy, opencv-python, scikit-image, matplotlib, pillow + dev: jupyter, nbconvert, pytest.
2. Implement `extract_data.py`:
   - extract the ZIP with `zipfile` into `data/synthetic/`;
   - extract the RAR with the system `tar.exe` (libarchive on Windows supports RAR, verified) into `data/real/`;
   - call it via `subprocess.run([...], check=True)`.
3. Verify the result: 259 `.npy` files, each (6250, 52); 6 synthetic samples, each (75000, 59); valid metadata JSON.
4. **Fix the segments.** Read the assigned file range from the Google spreadsheet (link in the brief) and enter it in `config.SEGMENTS`.
   - Working in a pair: 2 assigned segments + 1 own. Working individually: 1 assigned + 2 own. Always 3 in total.
   - Choose the own segments from the 090252–094332 run after the EDA (Stage 2). They must not overlap the assigned ones.
   - Choose them so that traffic differs: one with trams/trucks, one with many weak car lines.

**DoD:** `uv run python scripts/extract_data.py` finishes with an OK message; `config.SEGMENTS` has 3 entries.

### Stage 1 – Data loading

**Files:** `src/das/io.py`, `tests/test_io.py`.

1. `list_real_files`, `parse_timestamp`: name `HHMMSS` + date 2024-05-07 → `datetime`.
2. `find_continuous_runs`: neighbouring files must differ by 10 s. The function must return the two runs described in section 1.
3. `load_segment(start, n_files=12)`:
   - `np.concatenate` along the time axis;
   - returns a (75000, 52) matrix and the start time;
   - raises `ValueError` if the range contains a gap;
   - has a `pad_files=1` option that loads one extra margin file on each side (if it exists), so that FFT filtering does not corrupt the segment edges. The margin is cropped after processing.
4. `load_synthetic(id)`: returns (signal, gt, meta); dt and dx are taken from `meta["axes"]`.
5. Tests: continuity on an artificial list of names, shape of the concatenated segment, exception on a gap.

**DoD:** tests pass; `load_segment` returns (75000, 52) for every segment in the config.

### Stage 2 – Exploratory data analysis (EDA)

**Files:** `src/das/stats.py`, `scripts/run_eda.py`, `src/das/visualization.py` (`plot_waterfall`).

1. **Global statistics over all 259 files**, computed in streaming fashion (`RunningStats`, Welford), without loading everything into RAM:
   - mean, variance and RMS per channel;
   - global histogram in fixed bins (log scale), percentiles, skewness and kurtosis.
   High kurtosis indicates impulses, which justifies robust normalization (median/MAD) instead of mean/std.
2. **Per-segment statistics:** `describe`, `per_channel_stats` (RMS, std, MAD per channel), `per_time_stats` (e.g. 1 s window: RMS averaged over channels).
3. **Variability:**
   - RMS map (time × channel) in 1 s windows;
   - RMS profile along the cable, which shows dead or noisy channels, i.e. vertical stripes;
   - RMS over time, which shows global events, i.e. horizontal stripes;
   - variability of channel RMS between files over the 40 minutes.
4. `detect_bad_channels`: robust z-score of channel RMS (threshold ≈ 3.5). The result allows masking or interpolating bad channels in Stage 4.
5. Plots:
   - waterfall of the raw segment (|x|, clipped at the 99th percentile), visually like the figure in the brief;
   - histograms, per-channel boxplots, RMS map, correlation of neighbouring channels (spatial coherence).
6. Output: `outputs/results/eda_global.json`, `eda_<segment>.json`, figures in `outputs/figures/eda/`.

**DoD:** statistics table and at least 6 report-ready figures; list of bad channels (may be empty); own segments chosen.

### Stage 3 – Frequency analysis

**Files:** `src/das/spectral.py`, `scripts/run_spectral.py`.

1. `channel_fft`: `np.fft.rfft(x * window, axis=0)`, i.e. **independently for each channel** (requirement from the brief).
   Output: amplitude spectrum plot – a frequency × channel map in log scale and the spectrum averaged over channels.
2. `welch_psd`: segmentation via `sliding_window_view`, Hann window, averaging. Gives a smoother spectrum for choosing the band.
3. `stft`: compute a spectrogram in dB for 2–3 channels (the middle one and one with a clear tram passage).
   Show how a vehicle passage changes the frequency content: a tram has a wide band and low frequencies, a car a weaker, higher band.
4. `band_energy`: energy in bands, e.g. 0–1, 1–5, 5–10, 10–20, 20–50, 50–100, 100–312 Hz.
   Compute it separately for "vehicle" and "background" fragments, selected automatically by RMS: top and bottom quantile of windows.
   The vehicle/background energy ratio in each band is the **objective criterion for choosing `f_low`/`f_high`**.
5. `fk_spectrum`: 2D FFT of the (time × space) image after decimation.
   - Overlay constant-velocity lines at ±20, ±50, ±100 km/h on the f-k plot.
   - Show that vehicle energy lies inside the velocity "fan".
   - Show that noise concentrates near k≈0 (simultaneous events along the whole cable, i.e. infinite apparent velocity) and near f≈0 (static channel stripes).
6. Check for tonal interference (e.g. 50 Hz mains and harmonics) and add notch filters in the FFT domain if needed.

**DoD:**
- frequency band chosen and justified numerically (entered in `PreprocessingParams`);
- FFT/PSD/STFT/f-k figures in the report;
- note in `experiments_log.md`.

### Stage 4 – Noise removal and building the energy image

**Files:** `src/das/preprocessing.py`, `tests/test_preprocessing.py`.

Order of steps (each one stored in `stages`):

1. `detrend_channels`: remove the mean and linear trend of each channel.
2. `remove_common_mode`: subtract the **median across channels** at each time sample. Removes horizontal stripes, i.e. interference common to the whole cable.
   Also store the removed component for the "what was removed" visualization.
3. `bandpass_fft`:
   - `rfft` along time, band-pass mask with raised-cosine edges (`taper_hz`), `irfft`;
   - the filter is zero-phase, so it does not shift trajectories in time;
   - optionally add the notch from Stage 3.
4. `envelope_downsample`: RMS in blocks of 25 samples (`x.reshape(-1, 25, n_ch)`, then mean of squares and square root).
   Gives the energy envelope and decimation at once: 75000 → 3000 rows, 0.04 s/row.
   Compare with the mean-|x| variant and with plain decimation without an envelope (experiment for the report).
5. Interpolate bad channels from Stage 2 linearly from their neighbours.
6. `robust_normalize`: (x − median)/(1.4826·MAD) per channel. Equalizes channel sensitivity, i.e. removes vertical stripes.
7. `log_compress_and_clip`: `log1p` of the positive part, then clip at the 99.5th percentile.
   Reduces the dominance of trams, so the weak car lines become visible. **This is key for grade 5.**
8. Anisotropic smoothing: a light `cv2.GaussianBlur` (σ_t > σ_x) or `cv2.bilateralFilter`. Compare with a median filter.
9. `upsample_space`: `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` ×8 along the space axis.
10. Optional parallel variant: `skimage.exposure.equalize_adapthist` (CLAHE). Compare the contrast of weak lines before and after.

Tests:

- an in-band sine passes with < 1% error, out-of-band is attenuated by > 40 dB;
- common-mode removal removes a signal identical on all channels;
- shapes after decimation are correct.

**DoD:** a 3000 × 416 energy image on which both thick and thin lines are visible to the naked eye; a figure of the full chain of stages and the removed components.

### Stage 5 – Enhancing slanted lines (moving objects only)

**Files:** `src/das/enhancement.py`, `tests/test_enhancement.py`.

1. **f-k fan filter (`fk_velocity_filter`):**
   - Work on the image after step 7 (before upsampling, i.e. with the real dx), after subtracting the median.
   - Spatial padding 52 → 128 with a Tukey window to limit edge leakage.
   - The mask passes only |v| = |f/k| ∈ [v_min, v_max] = [10, 150] km/h, with tapered transitions (`fk_taper`).
   - **A separate mask for each direction**: the sign of f·k selects movement towards increasing or decreasing positions.
     This produces two images (+1 and −1), which separates **crossing lines** (visible in the brief's figure) and simplifies detection.
   - Physical justification:
     - a horizontal line has v→∞; it is a simultaneous event or a fast seismic wave with velocities of hundreds of m/s;
     - a vertical line has v→0; it is static channel noise;
     - both lie outside the fan and are removed.
   - Store the image of the removed component (input − output).
2. `subtract_background`: white top-hat (`cv2.morphologyEx(MORPH_TOPHAT)`) or subtraction of a moving median. Removes slowly varying background.
3. **Directional filter bank (`oriented_line_kernels` + `directional_filter_bank`):**
   - Kernels are line segments of length L (e.g. 41 px) with a Gaussian cross-profile, zero mean and unit norm, i.e. a matched filter.
   - Angles are taken only from the range corresponding to [v_min, v_max] for the given direction (`velocity_to_angle`), e.g. 24 angles.
   - Convolution via `cv2.filter2D`. Output: maximum response and angle index (local orientation and velocity).
   - Justification: integrating along the line increases SNR by roughly √L. **This is the main mechanism for extracting thin, weak lines.**
4. Alternative path for comparison in the report:
   - `ridge_filter`, i.e. `skimage.filters.sato`/`meijering` (Hessian ridge detector);
   - `skimage.filters.gabor` as an alternative to line kernels.
5. `enhance()` returns a dictionary: `fk_pos`, `fk_neg`, `removed_fk`, `bg_removed_*`, `response_*`, `angle_*`.

Test: a synthetic image contains a 50 km/h line, a horizontal line, a vertical line and noise. After filtering only the slanted line remains, in the correct direction channel.

**DoD:**
- on a real segment the response image shows clear thin and thick lines, and horizontal and vertical stripes are removed;
- a "before/after/removed" figure for the f-k filter and the directional filter.

### Stage 6 – Line detection: two processing paths

**Files:** `src/das/detection.py`, `tests/test_detection.py`.

Each path is run separately for direction +1 and −1.

**6A. Thick, strong lines (`detect_strong_lines`)**

1. Otsu thresholding (`skimage.filters.threshold_otsu`) on the response image. Alternatively a k·σ_robust threshold (`robust_noise_sigma`).
2. Morphological opening with a linear structuring element oriented in the dominant direction (several angles, then max).
   Removes specks and preserves elongated structures.
3. `skimage.measure.label` + `regionprops`. Reject objects with small area, low eccentricity, wrong orientation or too short a spatial extent (< `min_channels`).
4. For each component fit a line with RANSAC (`skimage.measure.ransac` + `LineModelND`) on pixels weighted by the response.
   Additionally compute thickness as area/length.
5. Return `LineSegment(source="strong")` segments and the **mask of strong objects** (for path 6B).

**6B. Thin, weak lines (`detect_weak_lines`) – grade-5 condition**

1. Mask out the strong objects. The mask is dilated by `strong_mask_dilation_px`, and the response in the masked area is set to 0.
   This way a tram's "halo" does not create false weak lines, and the threshold for weak lines can be lowered.
2. Thresholds are derived from the data: σ = 1.4826·MAD of the background response.
   `skimage.filters.apply_hysteresis_threshold(low=2σ, high=4σ)` follows a weak but continuous ridge.
3. `skimage.morphology.remove_small_objects`, then `skeletonize`.
4. `cv2.HoughLinesP` on the skeleton (`threshold`, `minLineLength`, `maxLineGap` from the config).
   Reject segments with an angle outside the velocity range or with the wrong direction.
5. Complementary `hough_global`, i.e. `skimage.transform.hough_line` + `hough_line_peaks` with `min_distance`/`min_angle`.
   Works well for long, straight and very weak lines because it accumulates votes along the whole line.
   Compare both methods in the report (recall of thin lines on synthetic data).
6. Optionally, for performance and better locality: process in 30 s time windows with 50% overlap and merge the segments in Stage 7.

Test: an image with one thick and one thin line (SNR ≈ 1.5) + noise. Both lines must be detected, the thin one via path 6B.

**DoD:**
- on synthetic data recall ≥ ~0.8 for `car` and ≈ 1.0 for `tram` (indicative targets; actual values are reported);
- on real segments most thin lines are visibly detected.

### Stage 7 – Grouping candidates and averaging lines

**Files:** `src/das/grouping.py`.

1. Describe each segment in physical space by the pair **(v [km/h], t_ref [s])**, where t_ref is the time of crossing the middle of the cable (`segment_to_params`).
   This parametrization does not depend on segment length.
2. `group_segments`:
   - pairwise distance matrix (vectorized); link when |Δv| < `tol_velocity_kmh` (or relatively < 15%) and |Δt_ref| < `tol_time_s`;
   - groups are connected components (union-find or `skimage.measure.label` on the adjacency matrix).
   This merges:
   - double edges of a thick line;
   - broken fragments of a thin line;
   - duplicates from time windows;
   - results of `HoughLinesP` and `hough_line`.
3. `merge_group`: weighted average (weight = length × mean response) or refitting a line to all endpoints and pixels of the group.
   The result is one `DetectedObject` with an extent (x_start..x_end, t_start..t_end).
4. `remove_duplicates`: if a "weak" object overlaps a "strong" one, the strong one is kept.
   Reject objects spanning < `min_channels` channels or with velocity outside 5–150 km/h (plausibility filter).
5. `kind` label (strong/weak): based on the source path, corrected by thickness and energy (Stage 10).

**DoD:** one line per vehicle; no multiple lines on a tram; a "candidates → groups → mean line" figure.

### Stage 8 – Velocity estimation

**Files:** `src/das/velocity.py`, `tests/test_velocity.py`.

1. `slope_to_velocity`: v = (Δcol·col_dx)/(Δrow·row_dt) [m/s], then ×3.6 → km/h.
   The sign gives the direction (+ towards increasing cable position).
2. `fit_constant_velocity`: fit a least-squares line to the tracking points (Stage 9) and report R².
   This velocity is more accurate than Hough alone and is the one reported in the tables.
3. Visualization as in the brief's figure: red lines on the waterfall, label "xx.xx km/h".
   Strong lines are drawn solid, weak lines dashed. Time axis in HH:MM:SS format, space axis in metres.
4. Per-segment object table (CSV): id, kind, direction, v [km/h], t_start, t_end, x_start, x_end, score, R².

Test: a line with a known slope gives a known velocity; the sign convention matches `direction` from the synthetic metadata.

**DoD:** for synthetic data, velocity MAE reported per vehicle type; for real data, tables and figures.

### Stage 9 – (Extra 1) Tracking and velocity over time

**Files:** `src/das/tracking.py`, `visualization.plot_velocity_profiles`.

1. `track_ridge`:
   - for each object and each channel in its extent, compute the predicted time from the mean line;
   - find the energy maximum in a ±`search_halfwidth_s` window (vectorized argmax on the cut-out band);
   - refine the position by parabolic interpolation (sub-sample).
   The output is a set of points (x_m, t_s).
2. `fit_trajectory_ransac`: reject outliers, e.g. a crossing with another vehicle.
3. `velocity_profile`: local slope dx/dt in a sliding window of `velocity_window_channels` (smoothed gradient or local linear regression).
   The output is v(t) in km/h.
4. `speed_variability`:
   - comparison of residuals of linear and quadratic fits (x(t) = a t² + b t + c);
   - standard deviation of v(t) and mean acceleration.
   Used for the "constant / varying speed" label.
5. v(t) plot: separate panels for several objects and a combined plot per segment coloured by cluster.
   Draw non-linear (curved) trajectories on the waterfall.

**DoD:** a v(t) plot for every object in every segment; identified braking and acceleration cases (e.g. at an intersection or a stop).

### Stage 10 – (Extra 2) Features and clustering of objects

**Files:** `src/das/features.py`, `src/das/clustering.py`, `scripts/run_clustering.py`.

1. Object features (`object_features`):
   - **geometry:** |v|, direction, std of v(t), curvature coefficient (a from the quadratic fit), R², duration, spatial extent;
   - **line thickness** (`line_thickness`): FWHM of the profile perpendicular to the trajectory [s], averaged over channels;
   - **energy:** maximum and mean response along the trajectory, total energy;
   - **spectrum:** spectral centroid and low/high band energy ratio of the raw signal along the trajectory (a window around t(x) for each channel);
   - **shape:** number of "sub-lines" (e.g. a tram = several bogies or axles, i.e. parallel fringes); transverse autocorrelation profile.
2. `standardize` (z-score), `choose_k` for k = 2..6 by silhouette (`silhouette_score` in NumPy) and the elbow method (compactness from `cv2.kmeans`).
3. `kmeans` via `cv2.kmeans` (`KMEANS_PP_CENTERS`, `attempts=10`).
4. Interpretation:
   - cluster centroids in physical units;
   - cluster names, e.g. "thick/fast/constant", "thin/constant", "varying speed";
   - a gallery of image crops (trajectory thumbnails) for each cluster, i.e. the "shape of the lines".
5. `pca_2d` (SVD) gives a 2D scatter coloured by cluster.
   Additionally show simple rules (thickness/energy thresholds) as a baseline for comparison.
6. Validation on synthetic data: `cluster_purity` against `type` (car/tram/van).
   On real data: agreement with the manual thick/thin labels (Stage 11).
7. Clustering is computed on objects from **all 3 segments together** (more samples). Other segments from the continuous run can also be added, which the brief allows.

**DoD:** PCA plot, centroid table, galleries, description of the clusters in the report.

### Stage 11 – Quantitative evaluation

**Files:** `src/das/evaluation.py`, `scripts/evaluate_synthetic.py`, `scripts/annotate_real.py`, `data/annotations/manual_labels.json`.

1. **Synthetic data:**
   - `gt_objects_from_mask`: for each `marker_value` take the (x, t) points from the mask, fit a line and derive v_true, t_ref, type and direction;
   - run the pipeline (dt and dx from metadata; 59 channels, so the code must not assume 52);
   - `match_detections`: greedy one-to-one matching by cost (|Δt_ref|/tol_t + |Δv|/tol_v) with gating;
   - `detection_metrics`: TP, FP, FN, precision, recall, F1 **separately for car (thin) and tram (thick)**, as evidence of meeting the grade-5 requirement;
   - `velocity_errors`: MAE, RMSE, MAPE;
   - `trajectory_iou` between the mask of detected trajectories and the GT mask (with tolerance).
2. **Real data (no GT):**
   - `annotate_real.py`: interactively (`plt.ginput`) mark the start and end of every visible line in the 3 segments, with a thick/thin label;
   - save to `manual_labels.json` (versioned);
   - the same metrics as above computed against the annotations. State in the report that this is a subjective assessment (a limitation of the method).
3. Save results to `outputs/results/eval_*.json` and to tables in the report.

**DoD:** a metrics table for the 6 synthetic samples and the 3 real segments, split into thick and thin.

### Stage 12 – End-to-end pipeline and results for 3 segments

**Files:** `src/das/pipeline.py`, `scripts/run_pipeline.py`.

1. `run_pipeline` chains Stages 4→5→6A/6B→7→8→9→10 (features) and times each stage with `time.perf_counter`.
2. `save_result`:
   - CSV of objects and JSON with parameters and timings;
   - optional cache of stages in `.npy`, so the report notebook does not recompute everything.
3. `run_pipeline.py --segment all` produces, for each segment in the config:
   - raw waterfall, image after preprocessing, image after f-k (both directions), filter response;
   - candidates, final result with velocities, v(t) plot, object table;
   - number of strong/weak objects per direction.
4. The same script works with `--synthetic <id>` and any `--start HHMMSS`, which shows that the approach is general and nothing is hardcoded.
5. Performance target: a 2-min segment in ≲ 20–30 s on a laptop. A table of stage timings is reported.

**DoD:** one command generates the full set of results for all 3 segments without errors or warnings (`uv run python -W error scripts/run_pipeline.py --segment all`).

### Stage 13 – Parameter tuning, ablation, experiments

**Files:** `scripts/tune_parameters.py`, `docs/experiments_log.md`.

1. Grid over the key parameters (`f_low`/`f_high`, decimation, hysteresis thresholds, kernel length, `tol_*`).
   We optimize F1 (car + tram) on synthetic data and F1 on the real-data annotations.
   **We do not tune for individual lines**, because the brief forbids hardcoding.
2. **Ablation:** switch off in turn common-mode removal, f-k, log compression, the filter bank and masking of strong lines.
   The result is a table of changes in thin-line recall, i.e. evidence of which step is responsible for what.
3. **Method comparison:**
   - HoughLinesP vs `hough_line` (global) vs RANSAC on components;
   - directional filter vs Sato/Meijering vs Gabor;
   - RMS vs |x|.
4. Sensitivity analysis: plot of F1 as a function of the hysteresis threshold.
5. Every attempt (including unsuccessful ones) is recorded in `experiments_log.md`: what, parameters, result, decision.
   Typical unsuccessful attempts to describe:
   - Canny + Hough on the raw image (many false lines from stripes);
   - a single global threshold (misses thin lines);
   - f-k without padding (edge artifacts).

**DoD:** ablation table, method comparison and the final parameter set saved as the defaults in `config.py`.

### Stage 14 – Report and block diagram

**Files:** `report/report.ipynb` (section skeleton already created), `scripts/make_report_figures.py`, `docs/flowchart.md`, `docs/references.md`.

1. Submission format: **`report.ipynb` + `report.html`**, via `uv run jupyter nbconvert --to html --execute report/report.ipynb`.
   Optionally a PDF by printing the HTML to PDF. The `.py` source code from `src/` and `scripts/` is included in full.
2. Block diagram: export the mermaid diagram from `docs/flowchart.md` to PNG (mermaid.live or draw.io) as `report/figures/flowchart.png`.
   Add a description for each block: what it does and why.
3. `make_report_figures.py` regenerates all figures at reduced resolution (dpi ≈ 100–110, waterfall downscaled with `cv2.INTER_AREA`) so the report is not too large.
4. Report content (the sections already exist in the notebook):
   1. data;
   2. EDA;
   3. frequency analysis;
   4. block diagram;
   5. preprocessing with "before/after/removed" images;
   6. enhancement;
   7. strong/weak detection and grouping;
   8. velocities;
   9. v(t);
   10. clustering;
   11. **detailed results for the 3 segments**;
   12. evaluation, ablation and unsuccessful experiments;
   13. conclusions, factual and methodological errors, and limitations;
   14. sources.
5. Limitations to discuss:
   - no GT for real data;
   - vehicles driving side by side or in a convoy merge into one line;
   - crossing lines;
   - unknown exact cable geometry (the cable may not run parallel to the road along its entire length, which causes velocity error);
   - the short cable section (≈260 m) limits the accuracy of v(t);
   - differences between synthetic and real data.
6. Sources: data, the course example notebook, library documentation, literature (list in `docs/references.md`; verify bibliographic details before citing).
   AI tools are also listed if used, with a description of what for.

**DoD:** the notebook runs from start to finish without errors; HTML < ~20 MB; every item in the brief's "Report" section is covered.

### Stage 15 – Quality control and submission

1. `uv run pytest` with `filterwarnings=error`: all tests pass (remove `skip` from the test templates).
2. Clean clone of the repo: `uv sync`, `uv run python scripts/extract_data.py`, `uv run python scripts/run_pipeline.py --segment all`. Everything passes without manual fixes.
   Also check the path without uv: `pip install -r requirements.txt && pip install -e .`.
3. Code review:
   - no per-pixel loops, no dead code;
   - docstrings and comments in place;
   - no hardcoded line positions;
   - parameters only in `config.py`.
4. Submission package: `src/`, `scripts/`, `tests/`, `report/report.ipynb`, `report/report.html`, `pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt`, `README.md`.
   No data, unless the instructor requires otherwise.
5. Presentation preparation (26.11 / 01.12): a short demo of `run_pipeline.py` on one segment, key figures, and the ability to justify every step and parameter.

---

## 4. Schedule (08.11 – 22.11.2026)

| Day | Stages | Result |
|---|---|---|
| 08–09.11 | 0, 1, 2 | data extracted, loader, EDA, segments chosen |
| 10.11 | 3 | filter band justified |
| 11–12.11 | 4, 5 | energy image + f-k + filter bank |
| 13–15.11 | 6, 7, 8 | strong + weak detection, grouping, velocities (**core of grade 5**) |
| 16.11 | 11 | metrics on synthetic data + annotations of the 3 real segments |
| 17.11 | 9 | v(t) |
| 18.11 | 10 | clustering |
| 19.11 | 12, 13 | full runs, tuning, ablation |
| 20–21.11 | 14 | report + block diagram |
| 22.11 | 15 | quality control, **submission before 23:59** |
| 23–25.11 | – | presentation preparation |

Buffer: if Stage 6B (thin lines) is delayed, it takes priority over Stages 9 and 10, because it is the grade-5 condition, whereas 9 and 10 are extra points.

---

## 5. Risks and plan B

| Risk | Symptom | Response |
|---|---|---|
| Thin lines get lost in noise | low `car` recall | longer kernels, log compression, masking of strong lines, global `hough_line`, less decimation |
| Many false lines | low precision | stricter f-k filter, minimum channel extent, velocity plausibility filter, higher `high` hysteresis threshold |
| Thick lines split into many | multiple objects on one tram | larger `tol_*` in grouping, morphological closing in 6A |
| Crossing lines | merged objects | direction separation in f-k (already planned), RANSAC during tracking |
| FFT edge artifacts | false lines at segment edges | `pad_files=1`, Tukey window, spatial padding |
| No permission for scipy | – | everything in `numpy.fft` (already planned this way) |
| Performance | > 1 min per segment | directional filter on the decimated image, windowed processing, stage cache |

---

## 6. Repository structure

```
Object_detection_from_DAS/
├── plan.md                      # this plan
├── README.md                    # installation, usage, directory layout
├── pyproject.toml, uv.lock, .python-version   # uv: dependencies, lockfile, Python version
├── requirements.txt             # export from uv (for installation with pip)
├── .gitignore                   # incl. .venv/, data, outputs/
├── Project 1 2026.pdf           # project brief
├── data/
│   ├── README.md                # data description and extraction
│   ├── 20240507.rar, Synthetic Dataset.zip   # (ignored by git)
│   ├── real/                    # 259 × HHMMSS.npy        (ignored)
│   ├── synthetic/               # {id}_signal/_ground_truth/_metadata (ignored)
│   └── annotations/manual_labels.json        # manual annotations (versioned)
├── src/das/
│   ├── config.py                # constants, paths, SEGMENTS, parameter dataclasses  (Stage 1)
│   ├── io.py                    # loading, continuity, segments                      (Stage 1)
│   ├── stats.py                 # EDA, streaming statistics                          (Stage 2)
│   ├── spectral.py              # per-channel FFT, Welch, STFT, f-k                  (Stage 3)
│   ├── preprocessing.py         # filtering, envelope, decimation, normalization     (Stage 4)
│   ├── enhancement.py           # f-k fan filter, directional filter bank            (Stage 5)
│   ├── detection.py             # strong (6A) and weak (6B) paths, Hough             (Stage 6)
│   ├── grouping.py              # candidate grouping, DetectedObject                 (Stage 7)
│   ├── velocity.py              # velocity, unit conversions                         (Stage 8)
│   ├── tracking.py              # ridge tracking, v(t)                               (Stage 9)
│   ├── features.py              # object features                                    (Stage 10)
│   ├── clustering.py            # cv2.kmeans, silhouette, PCA                        (Stage 10)
│   ├── evaluation.py            # metrics vs GT / annotations                        (Stage 11)
│   ├── visualization.py         # all plots                                          (whole project)
│   └── pipeline.py              # end-to-end orchestration                           (Stage 12)
├── scripts/                     # extract_data, run_eda, run_spectral, run_pipeline,
│                                # evaluate_synthetic, annotate_real, tune_parameters,
│                                # run_clustering, make_report_figures
├── tests/                       # unit tests per module
├── notebooks/                   # exploratory notebooks (01_eda, 02_spectral, 03_detection_dev …)
├── report/
│   ├── report.ipynb             # final report (section skeleton ready)
│   └── figures/                 # report figures (downscaled)
├── docs/
│   ├── flowchart.md             # block diagram (mermaid)
│   ├── experiments_log.md       # experiments log
│   └── references.md            # sources
└── outputs/                     # generated results (ignored): figures/, results/, cache/
```
