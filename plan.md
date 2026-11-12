# Implementation plan – Detecting Moving Objects in DAS Recordings (grade 5 version)

> Code and report deadline: **22.11.2026, 23:59**. Presentation: **26.11.2026 / 01.12.2026**.
> The plan covers the full grade-5 scope: real data, detection of **thick and thin lines**, and velocities.
> It also covers both extra points: the **v(t) plot** and **clustering of objects**.

---

## 0. Grade-5 requirements and where they are implemented

| # | Requirement from the project brief | Stage | Module / file |
|---|---|---|---|
| 1 | Load **real data**; 3 continuous 2-min segments (2 assigned + 1 own, see §1.1) | 1 ✅ | `das/io.py`, `config.SEGMENTS` |
| 2 | Exploratory analysis: statistics, distributions, variability in time and space | 2 ✅ | `das/stats.py`, `scripts/run_eda.py` |
| 3 | Frequency analysis: **FFT applied independently to each channel**, STFT | 3 ✅ | `das/spectral.py`, `scripts/run_spectral.py` |
| 4 | Noise filtering: frequency band, downsampling, filters, standardization | 4 ✅ | `das/preprocessing.py` |
| 5 | Processing so that **only moving objects** are visible | 5 ✅ | `das/enhancement.py`, `scripts/run_enhancement.py` |
| 6 | Detection of **thick lines (tram/truck)** | 6A ✅ | `das/detection.py`, `scripts/run_detection.py` |
| 7 | Detection of **thin, weak lines (cars)** – **grade-5 condition** | 6B ✅ | `das/detection.py` |
| 8 | Grouping of candidates and averaging the line within a group | 7 ✅ | `das/grouping.py` |
| 9 | Velocity of each object (km/h, direction) | 8 ✅ | `das/velocity.py`, `scripts/run_velocity.py` |
| 10 | (Extra 1) plot of **velocity over time** | 9 ✅ | `das/tracking.py` |
| 11 | (Extra 2) **clustering** of objects (thick/thin, constant/varying speed, shape) | 10 ✅ | `das/features.py`, `das/clustering.py`, `scripts/run_clustering.py` |
| 12 | Assessment of the number and accuracy of detected objects | 11 ✅ | `das/evaluation.py`, `scripts/evaluate_*.py`, `scripts/annotate_real.py` |
| 13 | Block diagram of the algorithm with step descriptions | 14 ✅ | `docs/flowchart.md` → `report/figures/flowchart.png` |
| 14 | "Before/after" visualizations for each step, including removed noise | 4–6, 14 ✅ | `das/visualization.py` |
| 15 | Detailed results for 3 two-minute segments | 12 ✅ | `scripts/run_pipeline.py` |
| 16 | Description of experiments (successful and unsuccessful), errors and limitations | 13 ✅ | `docs/experiments_log.md` |
| 17 | Sources of data, code and inspiration | 14 ✅ | `docs/references.md` |
| 18 | Code without errors or warnings, vectorized, readable | all, 15 ✅ | `tests/`, `pyproject.toml` (`filterwarnings=error`) |

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
| GT | none | mask with vehicle markers + `type` (car/tram/van/truck), `direction`, `lane_number`; 18–30 vehicles per sample |
| Data quality | 0 NaN/inf, all 259 files have 52 channels | ids 40–45; in sample 43 one vehicle (marker 25) is missing from the mask |

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

### 1.1 Analysed segments (pair work: Kuba + Mateusz)

The project is done in a **pair**. According to the brief, each student analyses their assigned segment, and the pair additionally selects **one** own 2-minute segment that does not overlap either assigned one. Total: **3 segments**.

| Name | Owner | Files (inclusive) | # files | Time span (2024-05-07) | Kind | Continuous run |
|---|---|---|---|---|---|---|
| `seg_kuba` | Kuba | `090452.npy` – `090642.npy` | 12 | 09:04:52 – 09:06:52 | assigned | 090252–094332 ✅ |
| `seg_mateusz` | Mateusz | `093222.npy` – `093412.npy` | 12 | 09:32:22 – 09:34:22 | assigned | 090252–094332 ✅ |
| `seg_own` | Kuba + Mateusz | `091152.npy` – `091342.npy` | 12 | 09:11:52 – 09:13:52 | own (chosen in Stage 2 ✅) | 090252–094332 ✅, no overlap |

Verified in the archive: both assigned ranges are complete (12 consecutive files, no gaps).
Both are entered in `config.SEGMENTS` and load as (75000, 52) via `io.load_named_segment` (Stage 1 ✅).

Rules for choosing `seg_own`:
- 12 consecutive files inside one continuous run, with no overlap with 09:04:52–09:06:52 or 09:32:22–09:34:22.
- Candidate areas: 090252–090442 (exactly 12 files before `seg_kuba`, adjacent but not overlapping), 090652–093212 (between the assigned segments), 093422–094332 (after `seg_mateusz`), or the separate run 090017–090227 (14 files).
- Pick it based on the EDA, so that its traffic complements the assigned segments: e.g. many thin car lines, or crossing lines in both directions, or a tram with braking.
- Prefer a segment with at least 1 file of margin on each side (for `pad_files=1` in `load_segment`).

Corresponding entries for `config.SEGMENTS`:

```python
SEGMENTS = [
    {"name": "seg_kuba",    "start": "090452", "n_files": 12, "kind": "assigned", "owner": "Kuba"},
    {"name": "seg_mateusz", "start": "093222", "n_files": 12, "kind": "assigned", "owner": "Mateusz"},
    {"name": "seg_own",     "start": "091152", "n_files": 12, "kind": "own",      "owner": "Kuba + Mateusz"},
]
```

**Why `seg_own` = 091152–091342** (decided 08.11.2026 from the EDA ranking in `outputs/results/eda_candidates.csv`):
- It has the highest traffic activity of all 2-minute windows that do not overlap the assigned segments: 25.8% of (1 s × channel) cells are above 3× the channel reference, with 10 traffic episodes.
  For comparison, `seg_kuba` has 11.8% / 6 episodes and `seg_mateusz` 9.9% / 2 episodes.
- It contains thick lines, many thin lines, and lines crossing in **both directions** (around 09:12:00–09:12:30 and 09:12:50).
  This complements the assigned segments, which are dominated by one direction.
- One file of margin is available on each side (needed for `pad_files=1`).
- Runners-up, for the report's discussion: 092052–092242, 092552–092742, 093852–094042.

Report implications: section 11 of the report presents detailed results for all three segments.
It should state clearly which segment was analysed by whom, and how and why `seg_own` was chosen.

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

### Stage 0 – Environment and data ✅ DONE (08.11.2026)

**Files:** `requirements.txt`, `pyproject.toml`, `scripts/extract_data.py`, `data/README.md`, `src/das/config.py`.

1. ✅ uv environment initialized (`uv sync`): numpy, opencv-python, scikit-image, matplotlib, pillow + dev: jupyter, nbconvert, pytest.
2. ✅ `extract_data.py` implemented:
   - the ZIP is extracted with `zipfile` into `data/synthetic/`, flattened to base names (also blocks path traversal);
   - the RAR is extracted into `data/real/` with the first available tool: Windows `tar.exe` (libarchive), `bsdtar`, `unrar` or `7z`; called via `subprocess.run([...], check=True)`;
   - extraction is skipped when the data is already present; `--force` re-extracts and `--verify-only` only verifies;
   - archive paths and expected counts live in `config` (`REAL_ARCHIVE`, `SYNTH_ARCHIVE`, `EXPECTED_REAL_FILES`, `EXPECTED_SYNTH_SAMPLES`).
3. ✅ Verification (part of `extract_data.py`, exit code 1 on any error). Result:
   - real data: 259 files, all (6250, 52) float32, 0 NaN/inf values, 2 continuous runs (090017–090227, 090252–094332);
   - synthetic data: 6 samples (ids 40–45), each (75000, 59), with metadata dt/dx equal to the real dataset;
     - shapes match the metadata, and every GT marker is described in the metadata;
     - vehicle types: car, tram, van, truck.
   - ⚠️ Sample **43** lists 25 vehicles, but only 24 appear in the GT mask: `vehicle_id` 24 (car, marker 25) is missing.
     Stage 11 must build the reference objects **from the mask**, not from the metadata list.
4. ✅ **Segments fixed** (see section 1.1). `config.SEGMENTS` contains `seg_kuba` (090452–090642) and `seg_mateusz` (093222–093412).
   Both are verified as continuous and loadable.
   The third, own segment `seg_own` (091152–091342) was added in Stage 2.

**DoD:** ✅ `uv run python -W error scripts/extract_data.py` finishes with `OK - data extracted and verified.` (≈10 s); `config.SEGMENTS` has all 3 entries.

### Stage 1 – Data loading ✅ DONE (08.11.2026)

**Files:** `src/das/io.py`, `tests/test_io.py`.

1. ✅ `list_real_files` lists only `HHMMSS.npy` names and returns them sorted.
   `parse_timestamp` turns the name `HHMMSS` plus the date 2024-05-07 into a `datetime`; invalid names and times raise `ValueError`.
2. ✅ `find_continuous_runs` splits files into runs where neighbours differ by exactly 10 s. Input order does not matter.
   On the real data it returns exactly the two runs from section 1.
3. ✅ `load_segment(start, n_files=12, pad_files=0, directory=None, name="")` returns a **`Segment` dataclass** instead of a tuple. It holds:
   - `data`: the concatenated (time, space) float32 matrix, including padding;
   - `start`: the start `datetime` of the core segment;
   - `files`, `pad_files`, `pad_before`, `pad_after` (padding in samples), `dt`, `dx`, `name`;
   - `core`: a view of the requested segment, e.g. (75000, 52), without padding;
   - `crop(arr, factor)`: removes the padding from any derived array, including arrays decimated by `factor`, e.g. 25 after the RMS envelope;
   - helpers `end`, `duration_s`, `n_samples`, `n_channels`, `time_axis()`, `distance_axis()`.
   
   Behaviour:
   - `ValueError` if the segment does not fit into one continuous run (the message names the run);
   - `FileNotFoundError` if the start file does not exist;
   - `pad_files` takes neighbours only from the same continuous run, so padding is shorter at run edges;
   - `resolve_segment_files()` resolves the file list without loading (used by the verification);
   - `get_segment_spec(name)` and `load_named_segment(name, pad_files)` load segments defined in `config.SEGMENTS`.
4. ✅ Synthetic data:
   - `list_synthetic_ids` returns ids in numeric order;
   - `load_synthetic(id)` returns (signal float32, gt, meta) and validates shapes against the metadata;
   - `load_synthetic_metadata` loads only the metadata;
   - `synthetic_axes(meta)` returns (dt, dx).
5. ✅ Tests in `tests/test_io.py`:
   - 23 unit tests on fake files in `tmp_path`: timestamps, filtering, continuity across a gap and a minute boundary, concatenation order, padding and `crop`, errors, synthetic loading and shape mismatch;
   - 4 integration tests on the real data, skipped if the data is not extracted: both runs (14 and 245 files), both configured segments → (75000, 52) with padding on both sides, synthetic (75000, 59).

**DoD:** ✅ `uv run pytest tests/test_io.py` → 27 passed (with `filterwarnings=error`); `load_named_segment` returns (75000, 52) for `seg_kuba` and `seg_mateusz`.

### Stage 2 – Exploratory data analysis (EDA) ✅ DONE (08.11.2026)

**Files:** `src/das/stats.py`, `scripts/run_eda.py`, `src/das/visualization.py` (`plot_waterfall`, `plot_overview_panels`, `hms_formatter`, `save_figure`), `tests/test_stats.py`.

Run: `uv run python -W error scripts/run_eda.py` (≈25 s, no warnings).

1. ✅ **Global statistics over all 259 files**, computed by streaming (`RunningStats`), without loading the whole dataset into RAM:
   - per-channel moments are merged batch-wise (Welford/Chan/Pébay formulas), so skewness and kurtosis are numerically stable;
   - signed histogram (800 bins, data-driven range ±40σ of the first file) and log-binned |x| histogram;
   - |x| percentiles approximated from the histogram (`hist_percentiles`).
2. ✅ **Per-segment statistics:**
   - `describe`: n, mean, std, rms, min/max, median, MAD, robust σ, percentiles, skewness, excess kurtosis;
   - `per_channel_stats`: mean, std, RMS, median, MAD, max|x|, p99|x|, skewness, kurtosis;
   - `per_time_stats` (1 s windows, built on `block_rms`);
   - `neighbour_correlation`, `channel_correlation_matrix`;
   - `activity_metrics`: active / strong / busy fractions and the number of traffic episodes, relative to the per-channel reference level.
3. ✅ **Variability:**
   - per-file × channel RMS heatmap over 40 min;
   - activity timeline;
   - 1 s RMS map with time and channel profiles for every segment;
   - a 2-minute-panel overview of the **whole** recording.
4. ✅ `detect_bad_channels`: robust z-score of log10 of the median per-file RMS, threshold 3.5 (see findings below).
5. ✅ Figures in `outputs/figures/eda/`:
   - global: `global_histograms`, `global_channel_profile`, `global_file_channel_rms`, `global_activity_timeline`, `global_overview`;
   - per segment (`seg_*/`): `waterfall` (linear + log RMS), `raw_zoom` (busiest 10 s, signed raw data + traces), `variability`, `channel_boxplots`, `histogram` (segment vs global), `correlation`.
6. ✅ Results in `outputs/results/`: `eda_global.json`, `eda_seg_kuba.json`, `eda_seg_mateusz.json`, `eda_seg_own.json`, `eda_candidates.csv` (activity of every admissible 2-min window).
7. ✅ `seg_own` chosen = **091152–091342** (section 1.1).
8. ✅ Tests: `tests/test_stats.py` (11 tests). For example, `RunningStats` matches NumPy to 1e-10 for unequal batches, and bad-channel detection finds both dead and noisy channels.

**Key EDA findings** (they drive Stages 3–6; describe them in report section 2):

| Finding | Numbers | Consequence |
|---|---|---|
| Extremely heavy-tailed distribution | global std 4.85e-7 vs robust σ ≈ 1.15e-7, excess kurtosis 435, skewness 0.90, max\|x\| 7.2e-5 (≈150 σ) | Use median/MAD normalisation, not mean/std. Use log compression (Stage 4, step 7). Percentile clipping for display. |
| Data are **quantised** | step q = 2.507e-9, 1–4% exact zeros (more in quiet files), robust σ ≈ 46 q | Never take log/division of raw values without eps. The MAD of very quiet channels can be only a few q. |
| Channel ~25 (≈128 m) is flagged as "bad" (z = 4.4) | RMS and kurtosis rise smoothly over channels 22–28 (≈110–145 m). Short **horizontal dashes** (≈5–8 channels wide, <0.5 s) appear there whenever vehicles pass. | A **physical hotspot**, probably a road/track feature such as a joint, manhole or tram crossing, **not a faulty channel**. Do **not** interpolate it. Robust per-channel normalisation equalises it, and the f-k fan filter (Stage 5) removes the quasi-horizontal events (apparent velocity → ∞). |
| Raw signal is oscillatory and broadband | signed raw zoom shows ± vibrations under a moving envelope | Trajectories are visible only in the **energy envelope** (RMS), which confirms Stage 4 step 4. |
| Weak (car) lines are barely visible in linear RMS but clear in log(RMS / channel median) | compare the left and right panels of `seg_*/waterfall.png` | Log compression + per-channel normalisation are essential for grade 5. |
| Spatial coherence alternates | neighbour correlation zig-zags between ≈0.3 and ≈0.9; the correlation matrix has a block structure | Channels are not uniformly coupled. Do not rely on a single neighbour; use smoothing/filters spanning several channels. |
| Most traffic moves towards **decreasing** x: in the waterfall, lines rise from lower-left to upper-right. Some lines go the opposite way, and the lines cross. *(Corrected in Stage 3, where the f-k energy confirmed v < 0 for 50–60% of the envelope energy.)* | overview panels, f-k | Direction-separated f-k filtering (Stage 5) is justified. |
| Traffic is bursty | per-file median RMS varies 5× over 40 min; quiet intervals of 10–60 s | Detection thresholds must be relative to the robust noise level, not absolute. |
| Short run 090017–090227: last 2 files (090217, 090227) are much quieter, with vertical stripes | overview panel 09:02:17 | Not used in the analysed segments; mention as a data limitation. |

Segment comparison (activity relative to the dataset per-channel reference):

| Segment | std | robust σ | kurtosis | active fraction | strong fraction | episodes |
|---|---|---|---|---|---|---|
| `seg_kuba` | 5.02e-7 | 1.15e-7 | 974 | 0.118 | 0.030 | 6 |
| `seg_mateusz` | 4.84e-7 | 1.12e-7 | 324 | 0.099 | 0.032 | 2 |
| `seg_own` | 7.00e-7 | 1.86e-7 | 194 | 0.258 | 0.050 | 10 |

No channel is flagged as bad inside any of the three segments.

**DoD:** ✅ statistics tables (JSON plus the tables above) and 5 global + 6 × 3 segment figures, report-ready; bad-channel list = [25] globally (a physical hotspot, kept); `seg_own` chosen and entered in `config.SEGMENTS`.

### Stage 3 – Frequency analysis ✅ DONE (08.11.2026)

**Files:**
- `src/das/spectral.py`;
- `scripts/run_spectral.py`;
- implemented early because the band choice needs them: `preprocessing.bandpass_fft`, `bandpass_response`, `remove_common_mode` (Stage 4) and `enhancement.fk_velocity_filter`, `fk_fan_mask`, `line_snr` (Stage 5);
- tests: `tests/test_spectral.py`, `test_preprocessing.py`, `test_enhancement.py`.

Run: `uv run python -W error scripts/run_spectral.py`.
The first run takes ≈2 min because of the band grid; it is then cached in `outputs/cache/spectral_fan_grid.npz` (≈25 s; `--recompute` refreshes it).

1. ✅ `channel_fft` applies `rfft` **independently to each channel** (Hann window, amplitude-normalised).
   Figure `<seg>/fft.png`: frequency × channel map (0–150 Hz) and the median spectrum (0–312 Hz).
2. ✅ `welch_psd`: Hann window, frames from `sliding_window_view`, one-sided density, Parseval verified in a test.
   The global PSD (all 259 files) is in `global_psd.png`.
3. ✅ `stft` for channels 8 (41 m), 25 (128 m hotspot) and 42 (214 m), 1 s windows (`<seg>/stft.png`):
   - trams are broadband 20–150 Hz for several seconds;
   - the hotspot shows short broadband impulses (wheel impacts) peaking around 100 Hz;
   - there are stationary local lines at ≈12 Hz and ≈31–35 Hz.
4. ✅ Band energies and vehicle-vs-background contrast (`band_contrast.png`), using 2 s cells, per-channel top 10% vs bottom 30% RMS:
   - the contrast is 26 dB at 1 Hz, has a minimum of 15 dB at 5–10 Hz and **peaks at 38 dB around 80 Hz** (> 28 dB over 50–200 Hz);
   - power share of the whole dataset: 50–200 Hz ≈ 58%, 1–5 Hz ≈ 7%.
5. ✅ `fk_spectrum`, `apparent_velocity`, `fk_velocity_energy`, `fk_fan_fraction` (`<seg>/fk.png`: raw signal, LF and HF envelopes, with ±20/50/100 km/h lines).
   - **Sign convention** (numpy FFT): a pattern moving with velocity v lies on **f = −v·k**, so v = −f/k. Tested on synthetic lines.
   - Raw signal: the vehicles appear as a slow ridge at f < 1 Hz (≈ −20…−50 km/h). The energy at 1–8 Hz is mostly fast waves (|v| > 150 km/h, seismic/surface waves).
   - Envelopes: the energy concentrates in the vehicle fan.
     In `seg_kuba` and `seg_mateusz`, 52–59% of the HF-envelope energy is at −150…−10 km/h and only 8–10% at +10…+150 km/h.
     In `seg_own`, 49% vs 24%, i.e. both directions are present.
     There is also a vertical band at k ≈ 0 (simultaneous events), which the Stage 5 fan filter removes.
6. ✅ Tonal interference:
   - there are **no** lines > 10 dB on the channel-median PSD, so no mains 50 Hz and **no notch filter is needed**;
   - single channels show weak local lines (1–3 Hz on ≤ 5 channels; 10.5, 26.5, 30, 33.5, 35.5, 48, 113.5 Hz on 1–2 channels);
   - these are removed implicitly by per-channel normalisation and the fan filter. `bandpass_response` still supports `notches` if needed.
7. ✅ **Band choice – objective criterion.** For a grid of 16 × 19 bands, each of the 3 segments goes through common-mode removal → band-pass → log RMS envelope (0.04 s) → f-k fan filter (10–150 km/h). The **line SNR** is the p99 of the robust z-score of the result (`band_selection.png`).
   - **HF path: 60–312.5 Hz**, line SNR **5.84**. The top-10 bands are all within 50–80 Hz × 120–312 Hz (5.5–5.8).
   - **LF path: 1–8 Hz**, line SNR **5.39**. The best band with f_high ≤ 10 Hz; 0.5–5 Hz gives 4.9, and 0.2–1 Hz only 2.9.
   - Broadband 0.2–312.5 Hz: 4.88, i.e. **both paths beat a single broad band**.
   - Rejected criteria (kept in the figures as documented experiments):
     - d' of log band power: the cells were selected by broadband RMS, so it always favours the widest band;
     - fraction of envelope f-k energy inside the fan: energy-weighted and penalised by short horizontal streaks that the fan filter removes anyway. It picked 0.2–1 Hz, which is visually the worst LF band.
8. ✅ Visual check (`<seg>/band_comparison.png`, 8 bands side by side):
   - **LF (1–8 Hz)** shows the **sharpest thin car lines** (the quasi-static load of the vehicle), but:
     - it is **blind for x < 55 m** (no LF response there);
     - **some lanes/directions are missing**: e.g. the tram towards +x in `seg_kuba` at 09:06:20–09:06:45 is visible only in HF;
     - short horizontal streaks remain.
   - **HF (60–312 Hz)** has a clean background and shows all thick and most thin lines, but the lines are wider (vibration footprint).
     It contains the hotspot dashes at ≈128 m (removed by the fan filter).
   - ⇒ **Two complementary processing paths (LF + HF) are required**, as the brief explicitly allows. Entered in `PreprocessingParams`: `hf_low/hf_high = 60/312.5`, `lf_low/lf_high = 1/8`.

**DoD:** ✅ two bands chosen and justified numerically (line-SNR grid) and visually; 2 global figures + 4 figures × 3 segments + `band_selection.png` and `band_contrast.png` (report section 3); `spectral_summary.json`; experiments logged in `docs/experiments_log.md`; 55 tests pass.

### Stage 4 – Noise removal and building the energy image ✅ DONE (08.11.2026)

**Files:** `src/das/preprocessing.py`, `tests/test_preprocessing.py` (14 tests).
The figures and the variant study for Stages 4 and 5 come from `scripts/run_enhancement.py` (see Stage 5).

Entry point: `preprocess(x, params, fs, dx, crop, bad_channels, bands)` or `preprocess_segment(seg, params)`.
It returns a `Preprocessed` object with `row_dt` (0.04 s), `col_dx` (5.1 m), `common` stages, `bands['hf'|'lf']` stages and `images`.
Preprocessing runs for **two bands**: HF 60–312.5 Hz and LF 1–8 Hz (Stage 3). Steps 1–2 are shared.
Segments are loaded with `pad_files=1`, and the padding is cropped after the envelope (`crop` argument). One segment takes ≈0.55 s for both bands.

Order of steps (each one stored as a stage, at image resolution):

1. ✅ `detrend_channels`: removes each channel's mean and linear trend (vectorised least squares).
   *Study: no effect, because the band-pass removes the trend anyway. It is kept because it is cheap and safe.*
2. ✅ `remove_common_mode` / `common_mode_signal`: the median across channels at each time sample.
   The removed component is stored (`common['common_mode']`).
   *Study: without it the HF AUC for light vehicles drops 0.830 → 0.810.*
3. ✅ `bandpass_fft` (+ `bandpass_response`): zero-phase FFT filter with raised-cosine edges, called for HF and LF.
   The removed out-of-band part is stored (`removed_by_bandpass`). No notch is needed.
4. ✅ `envelope_downsample`: RMS of blocks of 25 samples, so 75000 → 3000 rows at 0.04 s.
   The variants `abs_mean` and `decimate` (no envelope) are implemented.
   *Study: `abs_mean` is equal to RMS. `decimate` is clearly worse (HF light AUC 0.729 vs 0.830), so the envelope is necessary.*
5. ✅ `interpolate_channels`: linear interpolation of bad channels. Implemented but **off by default** (`interpolate_bad_channels=False`), because channel 25 is a physical hotspot.
6. ✅ `robust_normalize`: (x − median)/(1.4826·MAD) per channel; MAD = 0 is handled.
   Used by the alternative normalisation `robust_z`.
7. ✅ Normalisation and log compression (`normalize_envelope`):
   - default **`log_ratio`** = log10(envelope / channel median), clipped to [0, p99.5].
     This is a per-channel gain normalisation and log compression in one step, and it is independent of the channel gain;
   - alternative `robust_z` = log1p(max(z, 0)) (`log_compress_and_clip`).
   *Study:*
   - `robust_z` gives +0.017 light AUC in HF, the same in LF and in the fused map, but a lower SNR on real data, so `log_ratio` stays;
   - **without the lower clip, the LF AUC drops 0.960 → 0.938**, so the clip stays.
8. ✅ `smooth_anisotropic`: Gaussian with separate σ_t and σ_x, via `cv2.GaussianBlur`.
   *Study: no gain (AUC ±0.001), so it is off by default (`smooth_sigma_* = 0`).*
9. ✅ `upsample_space`: `cv2.resize(INTER_LINEAR)` ×8 along space. It is called in **Stage 5 after the f-k filter**, which needs the real dx; `space_upsample` is in `EnhancementParams`.
10. ✅ `clahe` (skimage `equalize_adapthist`), optional.
    *Study: +0.003 fused AUC, but it lowers the SNR on real data and adds a parameter, so it is off.*

Tests (all pass):
- in-band < 1% error, out-of-band > 40 dB attenuation, zero phase;
- common mode and detrend;
- envelope methods (RMS/abs-mean/decimate);
- gain equalisation by `robust_normalize` and `log_ratio`, interpolation, anisotropic smoothing, upsampling;
- full chain: shapes, crop and padding validation, and a moving 80 Hz burst that stands out in the HF image.

**DoD:** ✅
- two 3000 × 52 energy images per segment (3000 × 416 after the ×8 upsampling in Stage 5) in which thick and thin lines are visible;
- `outputs/figures/enhancement/<seg>/stage4_hf.png` and `stage4_lf.png` show the full chain with the removed components (common mode, out-of-band part).

### Stage 5 – Enhancing slanted lines (moving objects only) ✅ DONE (08.11.2026)

**Files:**
- `src/das/enhancement.py`;
- `src/das/evaluation.py`: pixel-level helpers `gt_envelope_mask`, `pixel_auc`, `enhancement_auc`;
- `src/das/visualization.py`: `plot_stages`;
- `scripts/run_enhancement.py`;
- `tests/test_enhancement.py` (17 tests).

Run: `uv run python -W error scripts/run_enhancement.py`. The figures take ≈1 min; the variant study takes ≈14 min, and `--skip-variants` skips it.

Entry point: `enhance(image, EnhancementParams, row_dt, col_dx)` returns an `Enhanced` object:
- `response[+1 | −1]`: robust-z maps, 3000 × 416;
- `angle_index[d]` and `velocities[d]`: the local orientation and velocity of the strongest kernel, useful in Stages 6–9;
- `stages`; `combined` (the maximum over directions); and `col_dx` after upsampling (0.64 m).

It runs on both band images, so a segment produces **4 maps**: HF±, LF±. Time ≈1.1 s per band.

1. ✅ **f-k fan filter** (`fk_velocity_filter`, `fk_fan_mask`, `tukey_window`):
   - tapered raised-cosine fan with |v| ∈ [`v_min`, `v_max`] = **[5, 150] km/h**, a **separate mask per direction**, and k = 0 rejected;
   - it runs on the Stage-4 image with the real dx, before upsampling;
   - it stores `fk_removed` (input − fan output) for the "removed" figure.
   - Physical justification: a horizontal line has v → ∞ (a simultaneous event or a fast seismic wave); a vertical line has v → 0 (static channel noise). Both lie outside the fan.
   - **Edge leakage, three padding modes compared:**
     - `reflect` removes the leakage of full-width horizontal stripes completely (tested: residual < 1e-6, vs 1.6% for `zero`);
     - but after common-mode removal such stripes no longer exist. On synthetic data `zero` was slightly better (light AUC 0.838 vs 0.830), and `tukey` was clearly worse (0.821, heavy 0.989);
     - ⇒ **`zero` is the default**, and `reflect` stays available.
   - `v_min` was lowered from 10 to 5 km/h so that slow vehicles, braking or starting, are kept. In synthetic sample 40 a vehicle at < 10 km/h was removed entirely. Light AUC +0.006, and the SNR on real data is unchanged.
2. ✅ `subtract_background`: white top-hat with a horizontal line element of 101 px (≈65 m).
   *Study: worse on synthetic data (light 0.957 vs 0.962), so it is off by default (`background=False`).*
   The f-k filter already removes the slowly varying background.
3. ✅ **Directional matched-filter bank:**
   - functions: `bank_angles`, `velocity_to_angle`, `angle_to_velocity`, `oriented_line_kernels`, `directional_filter_bank`;
   - kernels are segments with a Gaussian cross-profile (σ = 2 px) and a Hann taper, zero mean and unit norm;
   - 24 angles per direction, uniform between the slopes of `v_min` and `v_max`;
   - the correlation uses `cv2.filter2D`; the output is the maximum response and the argmax index, so each pixel also gets a local velocity estimate.
   - **Kernel length 81 px** (≈3.2 s along a typical line). 41 px gave light HF AUC 0.830, 81 px gives 0.846, 121 px is the same after fusing but blurs the line ends.
   - Test: on a weak line with SNR 0.7, the z-score on the line rises more than 2.5×.
4. ✅ Alternatives, compared on the same data (`ridge_filter` = skimage `sato`/`meijering`, `gabor_kernels` = cv2 Gabor bank):

   | Enhancer | light AUC HF / LF / fused | heavy AUC fused | time |
   |---|---|---|---|
   | **filter bank (initial, L=41)** | 0.830 / 0.960 / 0.962 | 1.000 | 26 s |
   | f-k only (no enhancer) | 0.815 / 0.966 / 0.956 | 1.000 | 9 s |
   | Sato | 0.812 / 0.936 / 0.948 | 0.996 | 81 s |
   | Meijering | 0.759 / 0.898 / 0.917 | 0.985 | 55 s |
   | Gabor | 0.841 / 0.943 / 0.949 | 0.997 | 31 s |
   | no f-k filter (ablation) | 0.790 / 0.925 / 0.933 | 0.999 | 24 s |
   | **FINAL: bank L=81 + zero pad + fan 5–150** | **0.867 / 0.972 / 0.974** | **1.000** | 27 s |

   Times are for 3 synthetic samples + 3 real segments.
   ⇒ The **matched filter bank is the best enhancer**. The Hessian filters and Gabor lose on thin lines, and the f-k filter is essential.
5. ✅ `enhance()` stores the stages `input`, `fk_both`, `fk_removed`, `fk_pos`/`fk_neg`, `upsampled_*`, `background_*` (if enabled) and `response_*`.

**Evaluation method (used from now on):**
- **primary: pixel AUC against the synthetic ground truth** (`enhancement_auc`), on samples 40, 42 and 44, separately for light (car/van, thin) and heavy (tram/truck, thick) vehicles.
  Negatives are pixels more than 1 s and 2 channels away from any vehicle.
- **secondary: line SNR on the real segments.**
  It is **not comparable between methods** that change the output distribution. For example, without the f-k filter the HF SNR *rises* to 13.4 because the hotspot dashes inflate p99, even though the AUC falls.
- Heavy vehicles are separated almost perfectly (AUC ≈ 1.000) by every reasonable variant. **The grade-5 difficulty is the light vehicles**, and the fused HF+LF map gives the best light AUC (0.974).

**Observations on the real segments** (`<seg>/final_maps.png`):
- crossing lines are separated by direction: in `seg_kuba` the tram towards +x appears only in the `v > 0` maps;
- the horizontal hotspot dashes at ≈128 m are removed;
- thin lines are clearest in LF `v < 0`;
- maps of the direction with little traffic show an oblique noise texture with z ≈ 2–4, so the Stage 6 thresholds must be data-driven (robust σ, hysteresis).

**Limitation for the report:** a vehicle that stops completely (v < 5 km/h) is suppressed by the fan filter.

**DoD:** ✅
- the response maps of the 3 real segments show clear thin and thick lines, with horizontal and vertical stripes removed (`final_maps.png`);
- before/after/removed figures `stage5_hf.png` and `stage5_lf.png` for each segment;
- `synthetic_40.png` compares the result with the GT;
- the variant study is in `variant_study.png`, `enhancement_variants.csv` and `enhancement_summary.json`.

### Stage 6 – Line detection: two processing paths ✅ DONE (09.11.2026)

**Files:**
- `src/das/detection.py`;
- `src/das/evaluation.py`: object-level helpers implemented early from Stage 11 (`gt_objects_from_mask`, `match_detections`, `detection_metrics`, `velocity_errors`);
- `src/das/velocity.py`: `slope_to_velocity`; `src/das/visualization.py`: `plot_segments`, `plot_detections`;
- `scripts/run_detection.py` (Stages 6 and 7 together);
- tests: `tests/test_detection.py` (13), `tests/test_evaluation.py` (3), `tests/test_velocity.py` (1).

Run: `uv run python -W error scripts/run_detection.py` (≈1 min of figures + ≈6 min variant study; `--skip-variants` skips it).
Detection takes ≈1 s per 2-minute segment (all 4 maps); grouping ≈0.01 s.

Entry point: `detect_candidates(enhanced, DetectionParams, v_range_kmh)` runs both paths on every map (HF±, LF±) and returns `Detections`:
- `segments`: `LineSegment` list in pixel coordinates, with `source` (`strong` / `weak` = HoughLinesP / `hough_global`), `direction`, `band`, `score` (mean response) and `thickness_px` (strong only);
- `stages`: binary maps of every step, for the before/after figures;
- `thresholds`: the data-driven thresholds of every map.

**6A. Thick, strong lines (`detect_strong_lines`), HF maps**
1. ✅ Threshold = Otsu of the positive response, **floored at 4 σ** (robust σ). Without the floor, Otsu in a direction without trams separates only noise texture.
   The floor is 4 σ instead of 6 σ so that wide, diffuse lines on real data (z ≈ 4–6) reach this path. Their skeleton in 6B is branched, and Hough fails on it (log #21, #27).
2. ✅ Union of binary openings with line structuring elements of 9 admissible slopes (41 px). This removes specks and horizontal/vertical streaks.
3. ✅ `label` + `regionprops`: area ≥ 1500 px and extent ≥ 40 m.
4. ✅ **Ridge** (local maxima along time, NMS ±5 rows) + **sequential RANSAC** (`LineModelND`, up to 4 lines per component), so two trams touching in the mask give two lines.
   A later line that overlaps no accepted line in x is the bent continuation of a line (near the cable ends). It only extends that line.
   The first version, with per-column centroids, failed for touching trams (log #19).
5. ✅ Thickness = mean number of component pixels per column assigned to the line. The kept components form the **strong mask**.

**6B. Thin, weak lines (`detect_weak_lines`), HF and LF maps – grade-5 condition**
1. ✅ The strong mask is dilated by an ellipse of ±12 rows (0.5 s) × ±6 columns (4 m) and replaced by the background median.
2. ✅ Hysteresis `low = med + 2σ`, `high = med + 4σ` (σ of the background outside the mask) → `remove_small_objects` (150 px) → `skeletonize`.
3. ✅ `cv2.HoughLinesP` (threshold 30, min length 60 px, gap 20 px, 0.5°) **and** global `hough_line` + `hough_line_peaks`, restricted to the admissible angles.
   Global lines are cut into segments by their skeleton support (gaps ≤ 60 rows).
4. ✅ Filters:
   - slope inside the velocity range of the direction;
   - **local orientation** from the Stage-5 filter bank within 10° of the segment angle;
   - mean response ≥ the low threshold.

**Results (6 synthetic samples, objects after Stage 7):**

| Variant | recall light (car/van) | recall heavy (tram/truck) | precision | F1 |
|---|---|---|---|---|
| strong path only (grade 3/4) | 0.176 | 0.980 | 0.913 | 0.621 |
| HoughLinesP only | 0.765 | 1.000 | 0.927 | 0.887 |
| global `hough_line` only | 0.788 | 0.980 | 0.950 | 0.902 |
| no masking of strong objects | 0.800 | 0.980 | 0.935 | 0.899 |
| weak path HF only / LF only | 0.412 / 0.812 | 1.000 / 0.980 | 0.913 / 0.936 | 0.743 / 0.903 |
| **FINAL** | **0.882 (75/85)** | **1.000 (49/49)** | **0.932** | **0.929** |

- DoD targets met: car recall ≥ ~0.8 and tram recall 1.0.
- Missed cars: visible for < 5 s at the record end, very slow or with strongly varying speed (≤ 30 km/h), and one at 110 km/h.
- `kind` strong/weak = **detection path**, not vehicle type. In a direction without trams, a clear car line passes the strong threshold (13 of 70 matched cars).
  Neither ridge score nor thickness separates cars from trams at this stage (log #22). The thick/thin label is left to clustering (Stage 10); recall is always reported per GT type.

**Real segments:** `seg_kuba` 17 objects (14 strong, 3 weak), `seg_mateusz` 24 (22 + 2), `seg_own` 23 (18 + 5).
- Figures in `outputs/figures/detection/<seg>/`: `stage6_<band>_<pos|neg>.png` (every step of 6A and 6B), `candidates_groups_objects.png`, `objects.png`.
- Table: `outputs/results/detection_objects_<seg>.csv`.
- Limitation: in dense areas, crossings with a strong line of the opposite direction leave short fragments, because the f-k direction separation is imperfect there.

### Stage 7 – Grouping candidates and averaging lines ✅ DONE (09.11.2026)

**Files:** `src/das/grouping.py`, `tests/test_grouping.py` (15 tests).
Entry point: `build_objects(segments, GroupingParams, row_dt, col_dx, x_ref_m, channel_dx_m)` → (objects, groups).

1. ✅ Pixel → physical convention, as in `plot_waterfall`: t = (r + 0.5) row_dt, x = (c + 0.5) col_dx. A line is t(x) = t_c + p (x − x_c), and v = 1/p.
   `segment_to_params` gives (v, t_ref) at the cable middle, independent of which fragment was detected.
2. ✅ `link_matrix` (vectorised, N × N) links two candidates of the same direction when either holds:
   - **same line**: |Δv| < 8 km/h or < 15 %, and the time gap at the midpoint between the segments < tol;
   - **continuation**: the x-ranges follow each other (gap −20…+20 m), the lines meet at the junction (time gap < tol), and the velocity ratio is ≤ 2.
     This joins pieces of bent trajectories (braking, line ends). Without it, FP rises from 9 to 17.

   tol = max(1 s, ½ thickness of a strong candidate), so edge segments of a tram join the tram.
   *Stage 13: tol_time_s raised to 1.5 s (real fragments merged, synthetic unchanged, log #59).*
   Groups come from `connected_components` (min-label propagation, no scipy).
3. ✅ `merge_group`: robust weighted LSQ of t(x) on points sampled along all segments.
   - Weight = score × band weight, with **LF ×2**; points are re-weighted with a 3σ-MAD rule.
   - Object extent = inlier extent.
   - Stored: `support_m` (union of the candidate x-ranges), `fit_rms_s`, `bands`, `sources`, `thickness_s` and `kind` (strong if any candidate is strong).
4. ✅ `remove_duplicates`:
   - duplicates: same direction, overlap ≥ 50 % of the shorter object, and lines within tol at both ends of the overlap → keep the strong one, then the one with larger support × score;
   - plausibility: extent ≥ 8 channels and 5 ≤ |v| ≤ 150 km/h;
   - a **weak object must be long (support ≥ 60 m) OR clearly above the noise (score ≥ 5 σ)**.

   Statistics behind the rule (log #29), as median support / z:
   - synthetic TP 229 m / 8.9;
   - synthetic FP 43 m / 4.0;
   - real weak objects before the rule 49 m / 4.1 (noise texture of the quiet direction).
5. ✅ Objects are renumbered by t_ref, and the per-segment table goes to `outputs/results/detection_objects_<seg>.csv`.

Velocity |v| MAE of matched synthetic objects: **2.9 km/h** (MAPE ≈ 4 %). Band weight LF ×1 / ×2 / ×4 changes it by only 0.04 km/h.

**DoD:** ✅
- one line per vehicle (one per tram, also for touching trams);
- "candidates → groups → mean line" figure for every real segment and every synthetic sample;
- variant study in `outputs/results/detection_variants.csv`, `detection_summary.json` and `outputs/figures/detection/variant_study.png`;
- experiments in `docs/experiments_log.md` #18–#30.

Open point for Stage 11: tol_time 1.5 s merges 12 more real objects (synthetic unchanged). The annotations should show whether these are fragments or close cars.
→ Resolved (Stage 11, 11.11): no missed label lies near a matched one (≥ 4.9 s), so close cars are not merged. 19 of the 24 false positives are still fragments on a labelled line, so there is rather too little merging than too much.

### Stage 8 – Velocity estimation ✅ DONE (09.11.2026)

**Files:**
- `src/das/velocity.py`;
- `src/das/visualization.py`: `plot_detections`;
- `scripts/run_velocity.py` (Stages 8 and 9 together);
- `tests/test_velocity.py` (4 tests).

Run: `uv run python -W error scripts/run_velocity.py` (≈1.5 min of figures + ≈4 min variant study; `--skip-variants` skips it).

1. ✅ `slope_to_velocity`: v = (Δcol·col_dx)/(Δrow·row_dt) [m/s], then ×3.6 → km/h.
   The sign gives the direction (+ towards increasing cable position).
2. ✅ `fit_constant_velocity`: least-squares fit x(t) = v·t + b to the tracking points (Stage 9), returning (v, b, R²).
   - This **track velocity** is the one reported in the tables (`reported_velocity_kmh`). The Stage-7 line velocity is the fallback for objects without a track.
   - Synthetic: |v| MAE **2.89 → 1.86 km/h** (light 2.23, heavy 1.29; MAPE ≈ 3 %), all 124 matched objects tracked.
   - Real: 62 of 64 objects tracked, median R² 0.99.
3. ✅ Visualization as in the brief's figure (`plot_detections`, `outputs/figures/velocity/<seg>/velocities.png`):
   - red lines on the raw waterfall with the label "xx.xx km/h";
   - strong lines solid, weak lines dashed;
   - time axis in HH:MM:SS, space axis in metres.
4. ✅ Per-segment object table (`velocity.write_objects_csv` → `outputs/results/velocity_objects_<seg>.csv`; Stage 6–7 tables use the same writer).
   Columns: id, kind, direction, velocity (track fit), line velocity, **R²**, speed class, acceleration, v_start / v_end, v_std, t_ref (also as clock time), t/x extent, track points, support, score, thickness, bands, sources.

Tests: ✅ known slope → known velocity, sign convention, known line → v and R² = 1, noisy line → R² < 1, reported-velocity fallback, CSV round trip.
The sign convention also matches `direction` for all 134 synthetic vehicles.

**DoD:** ✅ velocity MAE per vehicle type on synthetic data; tables and figures for the 3 real segments.

### Stage 9 – (Extra 1) Tracking and velocity over time ✅ DONE (09.11.2026)

**Files:**
- `src/das/tracking.py`;
- `src/das/pipeline.py`: `run_pipeline`, `run_segment`, `retrack` (Stages 4 → 9; implemented early from Stage 12, completed in Stage 12 with features, `save_result`, `load_cache`);
- `src/das/evaluation.py`: `profile_errors`;
- `src/das/visualization.py`: `plot_velocity_profiles`, `plot_profile_gallery`, tracks in `plot_detections`;
- `tests/test_tracking.py` (10 tests, 11 cases).

Entry point: `track_objects(objects, enhanced, TrackingParams, row_dt)` fills, for every object, `track` (N × [x, t]), `v_track_kmh`, `r2`, `accel_ms2`, `v_start_kmh`, `v_end_kmh`, `v_std_kmh`, `speed_class` and `profile` (M × [t, x, v]).
It takes ≈0.05 s per segment.

1. ✅ `track_ridge`, two passes, one passage time per cable channel:
   - **map**: the f-k filtered map of the object's direction (Stage 5), robust z, sum of the HF and LF z maps, Gaussian along time (1 row).
     The filter-bank response straightens curved tracks: v(t) MAE 5.33 vs 2.29 km/h (log #32). HF only gives 3.41 and LF only 4.35 (log #33).
   - **pass 1** (`ridge_times`, vectorised over channels): argmax in ±1 s around the averaged Stage-7 line, refined by a parabola through the peak and its neighbours.
     Peaks on the window edge are invalid. A centroid refinement was tested and rejected (log #35).
   - channels within the object extent ± 30 m are searched; peaks below z 1.5 are dropped.
2. ✅ `fit_trajectory_ransac`: RANSAC of a **quadratic t(x)**, i.e. constant acceleration (residual 0.3 s, 200 trials).
   It is solved for all trials at once with batched Vandermonde solves. It rejects e.g. a crossing vehicle.
   **Pass 2** searches ±0.5 s around the curve and keeps the longest run of channels (gaps ≤ 3).
   Order 1 cannot follow bent tracks; order 3 over-follows noise on real data (log #36).
3. ✅ `velocity_profile`: local LSQ slope in a sliding window of **9 channels (46 m)** (`sliding_window_view`), reported at the mean time of the window.
   v(t) MAE against the GT v(t): 5 / 7 / **9** / 11 / 15 channels → 4.31 / 3.01 / **2.29** / 1.87 / 1.46 km/h.
   9 is a compromise, because the smooth synthetic dynamics always reward smoothing (log #37).
4. ✅ `speed_variability`: linear vs quadratic x(t). It returns:
   - the acceleration **along the motion** (+ faster, − braking, for both directions);
   - v_start / v_end, the RMS of both fits and the std of v(t);
   - a label: **varying** if |v_end − v_start| ≥ 15 % of |v| AND the quadratic cuts the RMS by ≥ 30 %, else **constant**.

   Synthetic results:
   - acceleration error (median) **0.07 m/s²**;
   - class agreement with the GT **0.85**; errors are mostly GT-varying → constant (17/64), i.e. conservative.
5. ✅ Plots (`outputs/figures/velocity/`):
   - `<seg>/tracks.png`: averaged lines vs measured curved tracks, coloured by speed class;
   - `<seg>/velocity_profiles.png`: **combined v(t) per segment**, coloured by speed class; Stage 10 can pass cluster colours;
   - `<seg>/profile_gallery.png`: v(t) per object with constant and quadratic fits;
   - synthetic: `synthetic_<id>_tracks.png` (with GT tracks) and `synthetic_<id>_profiles.png` (with the GT v(t));
   - `variant_study.png`.
   - Results: `velocity_summary.json`, `velocity_variants.csv`, `velocity_synthetic_objects.csv`.

**DoD:** ✅ a v(t) plot for every object in every segment. Braking and acceleration cases are identified:
- real: 6 / 10 / 14 varying-speed objects in `seg_kuba` / `seg_mateusz` / `seg_own`;
- e.g. `seg_own` #14: 95 → 40 km/h, #18: 40 → 57 km/h.

Limitations:
- near crossings and between two close vehicles the track can jump (`seg_kuba` #5, #12: ±20 km/h swings); weak objects have noisier profiles. R² and v_std flag them;
- the cable section is only ≈260 m, so a profile has 10–40 points.

### Stage 10 – (Extra 2) Features and clustering of objects ✅ DONE (09.11.2026)

**Files:**
- `src/das/features.py`, `src/das/clustering.py`;
- `src/das/visualization.py`: `plot_clusters`, `plot_k_selection`, `plot_cluster_features`, `plot_cluster_gallery`, cluster colours in `plot_detections` / `plot_velocity_profiles`;
- `scripts/run_clustering.py`;
- `tests/test_features.py` (7 tests, 9 cases), `tests/test_clustering.py` (10 tests).

Run: `uv run python -W error scripts/run_clustering.py` (≈2 min; `--mode joint`, `--k` to override).

1. ✅ Object features (`object_features`, vectorised over channels, ≈0.01 s per object):
   - **cross-profile of the line**: in every channel a ±2 s window of the f-k energy map (HF+LF z) is cut around t(x) and re-centred on the ridge maximum.
     From it: FWHM thickness `thickness_s` (sub-row interpolation), peak z, area under the profile, **number of sub-lines** (`count_sublines`), tail asymmetry, z in HF and LF;
   - **spectrum**: the raw signal in a 0.8 s window on the trajectory (all channels at once, rFFT) relative to the median background spectrum of the channel → SNR in LF / HF [dB], LF−HF ratio, spectral centroid of the excess power;
   - **motion**: |v|, dv_rel = |v_end − v_start| / |v|, std of v(t), |a|, R², plus their logarithms (heavy tails);
   - geometry / detection: duration, extent, number of channels, strong/weak, score.
2. ✅ **Feature choice from a separability study** (`clustering_feature_auc.csv`, `feature_separability.png`).
   AUC on synthetic data, heavy vs light: HF SNR 0.99, peak z 0.97, area 0.97, but FWHM only 0.64.
   A line *looks* thick because it is strong. Varying vs constant speed: log dv_rel 0.90, log |a| 0.80.
   LF/HF spectrum, sub-lines, tail asymmetry: ≈0.5, i.e. no information. Sub-lines are not resolved at 0.04 s rows, which is a stated limitation.
   Final groups (`config.CLUSTER_FEATURE_GROUPS`):
   - **thickness** = (log peak z, log area, HF SNR);
   - **speed change** = (log dv_rel, log |a|, log std v(t)).
3. ✅ Building blocks:
   - `standardize`: z-score, NaN → centre, clip ±4; robust median/MAD as an option;
   - `kmeans` = `cv2.kmeans` (k-means++, 10 attempts, seeded; labels sorted by size);
   - `silhouette_score`: vectorised (D @ one-hot);
   - `choose_k`: by silhouette, with the compactness for the elbow plot.
4. ✅ **Factorized clustering** (default, `factorized_kmeans`):
   - one k-means per property, k ∈ [2, 3] by silhouette;
   - levels ordered low → high and named (thin / medium thickness / thick, constant / varying speed);
   - a cluster = the combination of levels, so the names are directly the categories of the brief.
   A joint k-means on all features (equal group weights, k 2–6 by silhouette, names from mean z) is kept as the comparison (`--mode joint`).
5. ✅ Validation on synthetic data (133 objects, NumPy `cluster_purity` / `adjusted_rand_index`):
   - k = 2 × 2: **thin/constant 51, thin/varying 23, thick/constant 25, thick/varying 34**;
   - thickness level vs heavy/light: purity **0.92**, ARI 0.70; trams in the thick clusters 46/59;
   - speed level vs GT speed class: purity **0.78**;
   - 4 clusters vs the 4 GT classes: ARI 0.45, purity 0.73.
   Rule baseline (Otsu threshold of the line area × Stage-9 speed class): ARI 0.53, close to k-means.
   Method study (`clustering_study.csv`):
   - joint k = 4: ARI 0.48;
   - joint k by silhouette: k = 2, which only finds thick / thin;
   - adding velocity + spectrum + shape: ARI 0.12 (noise features destroy the structure);
   - linear speed-change features: ARI 0.41.
6. ✅ Real data (64 objects of the 3 segments together): silhouette chooses 3 thickness levels × 2 speed levels → 6 clusters:
   - thin/constant 4, thin/varying 13;
   - medium/constant 9, medium/varying 22;
   - thick/constant 6, thick/varying 10.
   The thick clusters have FWHM ≈1.8–1.9 s vs 1.1–1.3 s, and HF SNR 28–30 vs 14 dB for the thin ones. The speed level agrees with the Stage-9 class for 73 % of objects.
   Agreement with the manual thick/thin labels (Stage 11, 40 objects): the single features separate them (AUC z_hf 0.84, peak z 0.83, FWHM 0.78), but the k-means thickness level does not (ARI ≈ 0), because HF SNR is not informative on real data (AUC 0.54). See Stage 11.
7. ✅ Figures (`outputs/figures/clustering/{real,synthetic}/`):
   - `pca.png` (PCA via SVD + loadings);
   - `k_selection.png` (silhouette + elbow per property);
   - `features.png` (box plots per cluster);
   - `gallery.png`: straightened lines per cluster, the "shape of the lines";
   - `real/<seg>_clusters.png`: waterfall + v(t) coloured by cluster.
   Tables: `clustering_real_objects.csv`, `clustering_real_centroids.csv`, the synthetic versions, `clustering_summary.json`.

**DoD:** ✅ PCA plot, centroid table, galleries; cluster description ready for the report (above + log #42–#47).

Limitations:
- with only 64 real objects and a continuum of speed changes, the silhouette is low (0.21);
- the clusters describe the signal, not the vehicle type;
- on real data the thickness level does not match the manual thick/thin labels (ARI ≈ 0, Stage 11): HF SNR does not separate them there.

### Stage 11 – Quantitative evaluation ✅ DONE (11.11.2026)

**Files:**
- `src/das/evaluation.py`, `src/das/annotation.py`;
- `scripts/evaluate_synthetic.py`, `scripts/annotate_real.py`, `scripts/evaluate_real.py`;
- `data/annotations/manual_labels.json`;
- `tests/test_evaluation.py` (+8 tests), `tests/test_annotation.py` (5 tests).

1. **Synthetic data** ✅ (`uv run python -W error scripts/evaluate_synthetic.py`, ≈1 min):
   - ✅ `gt_objects_from_mask` (markers present in the mask), pipeline with dt/dx from the metadata (59 channels);
   - ✅ `match_detections` (one-to-one, gates 1.5 s / 20 km/h), `detection_metrics`, `velocity_errors` (now also of the reported track velocity);
   - ✅ `class_metrics`: P / R / F1 **separately for thick and thin**. Class precision uses the detection path (6A = thick, 6B = thin); a thick/thin confusion is computed for matched pairs;
   - ✅ `trajectory_iou`: detected vs GT trajectories rasterised one pixel per channel (`rasterize_tracks`), dilated ±0.5 s, then IoU; pooled over samples from the overlap counts;
   - ✅ `summarize_evaluations` (micro-averaged totals) and gate sensitivity (1.0–3.0 s, 10–30 km/h).

   Result (6 samples, 134 vehicles):

   | | recall | precision | F1 | \|v\| MAE |
   |---|---|---|---|---|
   | thick (tram/truck) | **1.000** (49/49) | 0.940 | **0.969** | 1.29 km/h |
   | thin (car/van) | **0.882** (75/85) | 0.924 | **0.903** | 2.23 km/h |
   | all | 0.925 | 0.932 | **0.929** | 1.86 km/h (MAPE 3.5 %) |

   - trajectory IoU 0.63;
   - 15/75 matched cars come from the strong path (loud cars);
   - per-sample F1 0.90–0.97;
   - stable for gates ≥ 1.5 s / 20 km/h; a 10 km/h gate drops thick recall to 0.88, because the line velocity of long trams is off by > 10 km/h before tracking.
2. **Real data (no GT)** ✅ (11.11.2026; `uv run python -W error scripts/evaluate_real.py`, ≈1 min):
   - ✅ `annotate_real.py`: a matplotlib GUI on the raw log-RMS waterfall (not the detector output), with `b` switching to the Stage-4 HF/LF images.
     - Clicks give a polyline; `t` / `n` = thick / thin; undo, delete, scrolling of a 30 s window.
     - Physical units (x m, t s from the segment start) are stored in `manual_labels.json` together with the annotator.
     - The labels are saved automatically after every finished / undone / deleted line and when the window is closed.
     - Fixed 11.11: the tool was not referenced, and matplotlib keeps only weak references to callbacks, so clicks were ignored. A regression test now covers this.
   - ✅ Labels: 52 lines marked by hand (Kuba) on `seg_kuba` 15, `seg_mateusz` 18, `seg_own` 19.
     The reference is **indicative only**: rough clicks on a much noisier waterfall than the synthetic one. Therefore:
     - `split_references` lists but does not score labels with |v| outside the detector fan 5–150 km/h, or shorter than 30 m (`EvaluationParams.ref_v_range_kmh`, `ref_min_extent_m`).
       This excludes 5 labels: 4 near-horizontal strokes of 187–3147 km/h, i.e. horizontal noise stripes or imprecise clicks, and 1 label of 13 m. **47 labels are scored** (23 thick, 24 thin).
       Scores with all 52 labels are reported as well;
     - `false_positive_kinds` splits the false positives into "near a label" (a fragment or duplicate of a labelled vehicle, or a pair rejected by the gates) and "no label" (noise or an unmarked vehicle);
     - the sensitivity to the matching gates is reported (1.0 s / 15 km/h … 4.0 s / 40 km/h);
     - the detector is **not** tuned on these labels.

   Result (64 detections, 47 scored labels):

   | | recall | precision | F1 | \|v\| difference |
   |---|---|---|---|---|
   | thick | **0.957** (22/23) | 0.630 | 0.759 | 2.97 km/h |
   | thin | 0.750 (18/24) | 0.600 | 0.667 | 5.37 km/h |
   | all | **0.851** (40/47) | 0.625 | **0.721** | 4.05 km/h (MAPE 6.6 %) |
   | all, incl. the 5 not scored labels | 0.769 | 0.625 | 0.690 | 4.05 km/h |

   - per segment F1: `seg_kuba` 0.67, `seg_mateusz` 0.72, `seg_own` 0.76;
   - trajectory IoU 0.38 (labels are short and drawn by eye);
   - the result is **independent of the matching gates** (P/R/F1 identical from 1.0 s / 15 km/h to 4.0 s / 40 km/h), so the inaccuracy of the clicks does not change the scores.

   Errors:
   - **false positives: 19 of 24 lie on a labelled line.** They are fragments or duplicates in the halo of strong lines, e.g. the "X" of crossing fragments next to trams. Only 5 have no label nearby.
     Counting only those 5, precision would be 40/45 = 0.89, so the low precision is mostly over-segmentation of strong vehicles (an input for Stage 13), not phantom objects;
   - **missed labels (7):** all are weak, isolated thin lines ≥ 4.9 s from any matched label, 2 of them at the segment edge.
     This answers the open point of Stage 6: tol_time 1.5 s does not merge close cars;
   - **thick / thin class:** the detection path (6A strong / 6B weak) puts 14 of 18 matched thin labels on the strong path.
     Real cars are much louder relative to the background than synthetic ones, so class precision is about 0.6.
     The class of an object should therefore come from the features or clusters (Stage 10), not from the path;
   - **velocity:** |v| of the labels differs from the reported velocity by 4.1 km/h (2.97 thick, 5.37 thin), close to the accuracy of a manual slope.

   Clusters vs labels (`run_clustering.py`, 40 matched objects):
   - **single features** separate the manual thick / thin well: AUC z_hf 0.84, log peak z 0.83, FWHM thickness 0.78;
   - but the HF SNR, the strongest synthetic thickness feature (AUC 0.99), gives only 0.54 on real data (median ≈ 27 dB for both classes);
   - so the **thickness level of the k-means does not reproduce the manual labels**: ARI ≈ 0.
     The thick level holds 7 thick + 2 thin, the medium level 10 + 10, and the thin level 5 thick + 6 thin;
   - this is a limitation, stated in the report: the thickness group was chosen on synthetic data.
     It is not re-tuned on 40 indicative labels, which would fit the reference itself.
3. ✅ Results:
   - synthetic: `outputs/results/eval_synthetic.{csv,json}`, figures `outputs/figures/evaluation/synthetic_*.png`;
   - real: `eval_real.{csv,json}` (incl. the not scored labels and the gate study), `eval_real_pairs.csv` (label vs detected velocity), `evaluate_real.log`;
   - real figures: `real_<segment>.png` and `real_summary.png`.
     Colours: red = matched; orange = FP near a label; magenta = FP without a label; lime / yellow = found / missed label; cyan = not scored.

**DoD:** ✅ synthetic metrics table split into thick and thin; ✅ the 3 real segments evaluated against the manual labels (indicative reference, see log #48–#52).

### Stage 12 – End-to-end pipeline and results for 3 segments ✅ DONE (11.11.2026)

**Files:**
- `src/das/pipeline.py`: `run_pipeline`, `run_segment`, `PipelineResult.counts`, `result_table`, `stage_images`, `save_result`, `load_cache`;
- `src/das/clustering.py`: `cluster_objects` (Stage 10 for a list of feature dicts);
- `scripts/run_pipeline.py`;
- `tests/test_pipeline.py` (4 tests), `tests/test_clustering.py` (+1 test).

1. ✅ `run_pipeline` chains Stages 4 → 5 → 6A/6B → 7 → 8/9 → 10 (features) and times each stage with `time.perf_counter`.
   - Timers: `preprocess_s`, `enhance_s`, `detect_s`, `group_s`, `track_s`, `features_s`, plus `total_s` = their sum. The script also measures `load_s`.
   - The spectral features use the raw samples of the cropped segment (`crop[0]` = padding offset).
   - `features=False` skips Stage 10, and `keep_stages=False` drops the large images after tracking and features.
   - Clustering needs many objects, so it runs over all objects of one run (`cluster_objects`, factorized k-means of `ClusteringParams`), not per segment.
     For `--segment all` it gives exactly the Stage-10 result on real data (k = 3 × 2: 4 / 13 / 9 / 22 / 6 / 10), and for `--synthetic all` the synthetic one (51 / 23 / 25 / 34).
2. ✅ `save_result(result, name, t0, params, out_dir, cache_dir, extra)`:
   - `outputs/results/pipeline_<name>_objects.csv`: one row per object with the Stage 7–9 fields (reported velocity, v(t) summary, speed class, times / positions, wall-clock time), 15 Stage-10 features and the cluster name (csv module);
   - `pipeline_<name>.json`: axes (dt, dx), image shape, timings, counts (strong / weak × direction, speed classes), cluster sizes, all parameters (`asdict(PipelineParams)`) and `extra` (kind, start, duration; on synthetic data the metrics vs GT);
   - optional `--cache`: `outputs/cache/<name>/<stage>_<band>_<direction>.npy` (raw log RMS, Stage-4 images, f-k maps, responses, float32, ≈ 26 MB per 2-min recording). `load_cache(name)` reads them back for the report notebook.
3. ✅ `run_pipeline.py --segment all` produces for each segment (`outputs/figures/pipeline/<name>/`):
   - `01_raw.png`: raw waterfall;
   - `02_preprocessed.png`: raw vs the HF / LF images of Stage 4;
   - `03_enhancement.png`: f-k per direction + filter-bank response, HF and LF;
   - `04_candidates.png`: candidates → groups → one line per vehicle;
   - `05_result.png`: velocities on the raw waterfall as in the brief, plus the measured tracks coloured by cluster;
   - `06_velocity.png`: v(t) of every object, coloured by cluster.
   The object table is printed and saved. The number of strong / weak objects per direction is in the console, the JSON and `pipeline_summary.csv`.
   The stage timing table is in the console, `pipeline_summary.csv` and `figures/pipeline/timings.png`. Log: `outputs/results/run_pipeline.log`.

   | segment | objects | strong v>0 / v<0 | weak v>0 / v<0 | varying speed |
   |---|---|---|---|---|
   | `seg_kuba` | 17 | 3 / 11 | 2 / 1 | 6 |
   | `seg_mateusz` | 24 | 5 / 17 | 0 / 2 | 10 |
   | `seg_own` | 23 | 3 / 15 | 0 / 5 | 14 |

   *After Stage 13 (tol_time_s = 1.5 s): `seg_kuba` 15 (strong 2 / 11, weak 2 / 0, 5 varying), `seg_mateusz` 19 (4 / 13, 0 / 2, 9), `seg_own` 18 (2 / 12, 0 / 4, 8); ≈8.6–10 s per segment.*
4. ✅ Generality: the same script works with `--start HHMMSS [--n-files N]` (any window, e.g. `--start 092002`: 25 objects) and `--synthetic <id>|all`.
   On synthetic data the GT is overlaid and the metrics are stored (F1 0.90–0.97, |v| MAE 0.9–2.6 km/h per sample, as in Stage 11).
   Every parameter comes from `config.PipelineParams` and the axes from the data / metadata; nothing is specific to the 3 segments.
5. ✅ Performance (2-min segment, laptop i5-1235U, 15 W):

   | CPU state | preprocess | enhance | detect | group + track | features | total |
   |---|---|---|---|---|---|---|
   | not throttled (6 synthetic + `start_092002`) | 0.8–1.0 s | 5.1–6.2 s | 1.2–1.4 s | 0.1 s | 0.3–0.4 s | **7.7–9.0 s** |
   | on battery, throttled (3 segments) | 0.9–2.7 s | 8.7–29.6 s | 1.3–5.4 s | 0.1–0.4 s | 0.3–1.0 s | 12.9–38.7 s |

   - The target ≲ 20–30 s is met with a few × margin when the CPU is not throttled.
   - On battery (balanced power plan, the same code) the CPU is throttled. A 24-kernel benchmark slowed from 0.9 s to 3–4.9 s after a few seconds of load, so the measured times vary 4× between runs.
   - ≈ 70 % of the time is the Stage-5 directional filter bank (2 bands × 2 directions × 24 kernels, `cv2.filter2D` on the 8× upsampled grid).
     Cheaper settings (fewer angles, smaller upsampling) change the results, so they are an ablation for Stage 13, not a Stage-12 change.

**DoD:** ✅ `uv run python -W error scripts/run_pipeline.py --segment all` generates the full set of results for the 3 segments (exit 0, no warnings).
`--synthetic all --no-figures --cache` and `--start 092002` also pass with `-W error`. 159 tests pass.

Notes for Stage 13:
- v(t) at the edge of an ad-hoc window: an object cut by the window end (`start_092002` #23, t_end = 120.1 s) has v_end 217 km/h. The 3 segments have no such case (padding).
- Clamping the local profile at the image edge is an option.

### Stage 13 – Parameter tuning, ablation, experiments ✅ DONE (12.11.2026)

**Files:**
- `scripts/tune_parameters.py` (studies `ablation`, `methods`, `sweep`, `hysteresis`, `grouping`, `combined`; `--quick`, `--jobs`);
- `src/das/pipeline.py`: `run_pipeline(pre=, enhanced=, detections=)` reuses earlier stages;
- new options for the ablation: `PreprocessingParams.normalization = "ratio"` (no log compression), `EnhancementParams.fk_pad_mode = "none"`;
- `src/das/tracking.py` + `TrackingParams.edge_margin_s` (fix of the Stage-12 edge artefact);
- `docs/experiments_log.md` #54–#60;
- tests: edge margin (`test_tracking.py`), `ratio` normalisation, f-k without padding, reused stages (`test_pipeline.py`) – 162 tests pass.

Run: `uv run python -W error scripts/tune_parameters.py` (83 variants, 78 distinct, × 9 recordings, ≈16 min with 9 processes on battery).
Outputs: `outputs/results/tuning_<study>.csv`, `tuning.json`, `tune_parameters.log`, figures `outputs/figures/tuning/`.

1. ✅ **Evaluation:** every variant changes the final parameters in one place. Stages 4 → 9 run on the 6 synthetic samples and the 3 labelled real segments, and the objects are scored with the Stage-11 evaluation.
   - Runs in parallel per recording. Inside a worker the results of Stages 4 / 5 / 6 are cached per upstream parameters, so a grouping grid costs only grouping + tracking.
   - **Selection rule:** the synthetic F1 decides. The real F1 only breaks ties between synthetic-equivalent settings (|ΔF1| < 0.005) and vetoes changes that fit only the simulator.
2. ✅ **Grid / sweep:** band edges, decimation, kernel length, number of angles, upsampling, strong floor, minimal extent / support, hysteresis grid (low × high) and grouping grid (tol_time × tol_v).
   - The synthetic F1 is maximal at every final value. The hysteresis optimum is 2 / 4 σ, and the real F1 is flat (0.73–0.81) over the grid.
   - Speed option: 12 angles or ×4 upsampling halve the filter-bank cost at −0.02…−0.03 synthetic F1. Not taken.
3. ✅ **Ablation at the detection level** (synthetic F1 / thin recall; final 0.929 / 0.882):

   | variant | synthetic F1 | synthetic thin R | real F1 |
   |---|---|---|---|
   | final | **0.929** | **0.882** | **0.788** |
   | no common-mode removal | 0.900 | 0.812 | 0.788 |
   | no log compression | 0.817 | 0.647 | 0.504 |
   | no f-k fan filter | 0.893 | 0.800 | 0.557 |
   | no filter bank | 0.859 | 0.765 | 0.708 |
   | no masking of strong lines | 0.899 | 0.800 | 0.780 |
   | no hysteresis (single 4 σ) | 0.903 | 0.812 | 0.783 |
   | weak path on HF only | 0.740 | 0.412 | 0.809 |
   | strong path only (grade 3/4) | 0.621 | 0.176 | 0.756 |
   | no ridge tracking | 0.929 (\|v\| MAE 2.89 vs 1.89) | 0.882 | 0.788 |

4. ✅ **Method comparison** (synthetic / real F1):
   - HoughLinesP only 0.887 / 0.768; global `hough_line` only 0.902 / 0.809;
   - Sato 0.663 / 0.337, Meijering 0.580 / 0.217, Gabor 0.476 / 0.257;
   - mean |x| 0.893, decimation 0.874; robust z 0.935 / 0.772 (loses 2 trams, rejected);
   - f-k padding reflect 0.914, Tukey 0.903 (real 0.661), none 0.908; fixed 6 σ 0.906 (real thick R 0.78);
   - **unsuccessful approaches, now measured:** Canny + HoughLinesP on the raw image 0.591 synthetic and **0 correct detections on real data**; on the HF image 0.040 synthetic. Also the single threshold and the f-k filter without padding (above).
5. ✅ **Final parameters:** the only change is **`GroupingParams.tol_time_s` 1.0 → 1.5 s**.
   - Synthetic F1 unchanged (0.929). Real F1 0.721 → **0.788**: P 0.625 → 0.750, FP near a label 19 → 10, thick recall unchanged; one thin label of `seg_mateusz` is merged with a neighbour.
   - This resolves the Stage-6 open point.
   - Rejected although better on the labels: a strong floor of 5 σ (real 0.817). It loses a real thick line, contradicts log #21 and rests only on the rough labels. Also rejected: 2.0 s and min extent 60 m (both worse on synthetic data).
   - All results of Stages 6–12 were regenerated with the final parameters. Counts per segment are now 15 / 19 / 18 objects, and clustering gives 6 clusters over 52 objects.
6. ✅ **Edge artefact of Stage 12 fixed:** peaks within 0.5 s of the first / last image row are dropped in both tracking passes (`edge_margin_s`).
   - Without the margin a ridge that continues beyond the window has its noisy flank maximum a few rows before the border. This compresses the passage times, and v(t) reached 351 km/h at the end of window 09:20:02.
   - Synthetic detections are unchanged; |v| MAE 1.86 → 1.89 km/h.

**DoD:** ✅ ablation table, method comparison (incl. the classical baselines) and the final parameter set saved as the defaults in `config.py`. All experiments are in `experiments_log.md`.

### Stage 14 – Report and block diagram ✅ DONE (12.11.2026)

**Files:**
- `report/report.ipynb` + `report/report.html`;
- `scripts/make_report_figures.py`;
- `docs/flowchart.md`, `docs/references.md`;
- `report/figures/` (67 figures + `flowchart.png`).

1. ✅ Format: `report.ipynb` (executed, with outputs) + `report.html` (15.7 MB < 20 MB).
   - `uv run jupyter nbconvert --to notebook --execute --inplace report/report.ipynb` and `uv run jupyter nbconvert --to html report/report.ipynb`.
   - The notebook runs from start to finish without errors.
   - It displays the figures and reads all tables from `outputs/results/` (no numbers typed into tables).
   - Section 11 **runs the whole pipeline live on `seg_kuba`** and checks that the result is identical to the saved table.
   - Appendix B lists the full source code of `src/das/` and `scripts/` (30 files, ≈9.5 k lines); Appendix A shows `experiments_log.md`.
2. ✅ Block diagram: drawn with matplotlib by `make_report_figures.py` (`report/figures/flowchart.png`), not with mermaid.live or draw.io, so it needs no external tool.
   - Each block notes what it removes. The mermaid source in `docs/flowchart.md` has the same blocks.
   - Report section 4 has a description table (what each block does and why).
3. ✅ `make_report_figures.py`:
   - `--regenerate` reruns all scripts with `-W error`; `--skip-studies` skips the slow variant studies.
   - Then it draws the flowchart and copies 67 curated figures downscaled with `cv2.INTER_AREA` to ≤ 1300 px. Files > 150 kB become JPEG (q 82), 8.9 MB in total.
4. ✅ Report content: all 14 sections of the plan.
   - Section 11 holds the per-segment results: waterfall, result with velocities and clusters, v(t), object table, counts strong / weak per direction, timings.
   - Section 12 covers evaluation and the Stage-13 studies, plus a table of unsuccessful approaches.
   - Section 13 gives conclusions, factual / methodological errors found, and limitations.
5. ✅ Limitations discussed:
   - no GT for real data and a subjective reference;
   - synthetic ≠ real;
   - convoy / side-by-side vehicles and the halo of trams;
   - crossings;
   - unknown cable geometry;
   - the 266 m section (few v(t) points, LF blind for x < 55 m);
   - the 5–150 km/h fan;
   - unresolved sub-lines;
   - timings depending on the power state.
6. ✅ Sources (`docs/references.md`, shown in report section 14): data, libraries with the functions used, literature with the place of use, and the AI assistant (Claude / Claude Code) with a description of what it was used for.

**DoD:** ✅ the notebook runs from start to finish without errors; HTML 15.7 MB < ~20 MB; every item of the brief's "Report" section is covered:
- dataset analysis → sections 2–3;
- block diagram → 4;
- before / after visualisations incl. removed components → 5–7;
- results for 3 segments → 11;
- analysis, conclusions, errors, limitations → 12–13;
- references → 14.

Open for Stage 15:
- the authors' full names on the title cell;
- a final read of the report text.

### Stage 15 – Quality control and submission ✅ DONE (12.11.2026)

1. ✅ `uv run pytest` with `filterwarnings=error`: **162 passed**.
   - No `skip` markers are left in the test templates. The only `skipif` markers are the three data-integrity tests of `test_io.py`, which run whenever the data are extracted.
   - Every script was run with `uv run python -W error` without warnings.
2. ✅ **Clean copy** of the versioned files plus the two archives in a scratch folder (all work is still uncommitted, so a copy of `git ls-files -co --exclude-standard` was used instead of a clone):
   - `uv sync` → `extract_data.py` (259 files + 6 samples verified) → `run_pipeline.py --segment all` (≈1 min) → `pytest`: everything passed without manual fixes.
   - The object tables of the three segments were **byte-identical** to the ones in the repository.
   - **Path without uv:** fresh `venv`, then `pip install -r requirements.txt && pip install -e .`, then `run_pipeline.py --segment seg_kuba`: OK.
   - Same versions in all environments: numpy 2.5.4, opencv 5.0.0, scikit-image 0.26.0, matplotlib 3.11.2. `requirements.txt` is identical to a fresh `uv export`.
3. ✅ **Code review:**
   - **Loops:** no per-pixel loops. Loops run only over bands, directions, kernels, components, objects, channels with an issue and feature columns.
   - **Dead code removed:**
     - stubs `plot_spectra`, `plot_stft`, `plot_fk` in `visualization.py` (they raised `NotImplementedError`);
     - `velocity.slope_to_velocity` and `ms_to_kmh`, and `grouping.segment_to_params`, which only tests used.
     - The tests now check `detection.velocity_to_slope` and the vectorised `grouping._segment_arrays` instead.
   - **Hidden parameter:** the slope tolerance of the velocity fan, `slope_ok(margin=0.1)`, is now `DetectionParams.slope_margin`. It is passed explicitly everywhere, including the Canny baseline of `tune_parameters.py`.
   - **Docstrings:** added to all parameter dataclasses, `PipelineResult`, `RunningStats.update`, the annotation event handlers, and 51 non-trivial script functions. Each new docstring was checked against the function body.
   - **No hardcoded lines:** `config.SEGMENTS` defines only the analysis windows. All thresholds are relative to the robust σ of each map, and the same code runs on 59-channel synthetic data and on any window (`--start`).
   - **Outputs unchanged:** after the changes the three segments give the same objects and velocities.
   - **Small fix:** `run_pipeline.py --no-figures` no longer prints a figure path.
   - **Reproducibility observation:**
     - One real object of `seg_kuba` (#3) lies at the thick / medium boundary of the factorized k-means. It changed its thickness level between two runs of identical code; everything else was identical.
     - The current code gives the same result in repeated runs (features compared bit by bit across processes).
     - Recorded as a limitation in report section 13 and in log #62.
4. ✅ **Submission package:** listed in `README.md` ("Submission package"), as decided with the user (a list, no zip script).
   - **Included:** `src/`, `scripts/`, `tests/`, `report/report.ipynb` + `report.html` + `report/figures/`, `docs/`, `data/README.md` + `data/annotations/manual_labels.json`, `pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt`, `README.md`, `plan.md`.
   - **Not included:** data, `outputs/`, `.venv/` and the brief PDF.
   - `docs/` and the labels are needed because the notebook and `evaluate_real.py` read them; `plan.md` because the report refers to "Stage N".
   - `data/README.md` now contains the download links of the brief.
5. ✅ **Presentation:** `docs/presentation.md` contains:
   - the live demo (`run_pipeline.py --segment seg_kuba`, alternatives `--start`, `--synthetic`);
   - a ≈8-minute outline with the figures in `report/figures/`;
   - a one-line justification with the measured numbers for every step and key parameter.

**Report final read:**
- The text agrees with the result tables.
- A note on the "Stage N" / "log #N" numbering was added to the title section, and the borderline-cluster limitation to section 13.
- The notebook was regenerated and executed: 0 errors, live pipeline identical to the saved table, Appendix B with the current code. `report.html` is 15.7 MB.
- Authors stay "Kuba and Mateusz" (the user's decision); full names can be added to cell 0 before submission.

**DoD:** ✅ tests and all scripts without errors / warnings; clean setup with uv and with pip; code review done; package list in the README; presentation notes. **Left to the user:** the authors' names (optional), committing / submitting before 22.11.2026 23:59.

---

## 4. Schedule (08.11 – 22.11.2026)

| Day | Stages | Result |
|---|---|---|
| 08–09.11 | 0 ✅, 1 ✅, 2 ✅ | data extracted, loader, EDA, segments chosen |
| 10.11 | 3 ✅ (08.11) | filter bands justified (LF 1–8 Hz + HF 60–312.5 Hz) |
| 11–12.11 | 4 ✅, 5 ✅ (08.11) | energy images (HF + LF) + f-k per direction + filter bank; variant study |
| 13–15.11 | 6 ✅, 7 ✅, 8 ✅ (09.11) | strong + weak detection, grouping, velocities (**core of grade 5**) |
| 16.11 | 11 ✅ (synthetic 09.11, real 11.11) | metrics on synthetic data + annotations of the 3 real segments |
| 17.11 | 9 ✅ (09.11) | v(t) |
| 18.11 | 10 ✅ (09.11) | clustering |
| 19.11 | 12 ✅ (11.11), 13 ✅ (12.11) | full runs, tuning, ablation |
| 20–21.11 | 14 ✅ (12.11) | report + block diagram |
| 22.11 | 15 ✅ (12.11) | quality control, **submission before 23:59** |
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
│   ├── presentation.md          # demo + talk outline + justification of every step
│   ├── experiments_log.md       # experiments log
│   └── references.md            # sources
└── outputs/                     # generated results (ignored): figures/, results/, cache/
```
