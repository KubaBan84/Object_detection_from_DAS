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

### Stage 6 – Line detection: two processing paths

**Files:** `src/das/detection.py`, `tests/test_detection.py`.

Each path is run separately for direction +1 and −1.
**Update after Stage 3:**
- 6A (strong lines) runs on the **HF** image.
- 6B (weak lines) runs on **both** the HF and the **LF** image. LF gives the sharpest car lines and the most precise velocities, but it does not see x < 55 m or some lanes.
- Stage 7 merges the candidates from both bands. When an object is found in both, keep the LF geometry (thinner line) and record which bands detected it; this is also a clustering feature.

**Inputs from Stage 5:** `Enhanced.response[±1]` for HF and LF.
- These are robust-z maps of 3000 × 416 px, with 0.04 s per row and 0.64 m per column. Thresholds can therefore be expressed directly in σ.
- `Enhanced.angle_index[d]` / `velocities[d]` give a per-pixel slope, i.e. a velocity estimate. Use it to reject Hough segments whose slope disagrees with the local orientation.
- Measure the AUC / recall on synthetic data with the helpers in `evaluation.py`.
- Note: in a direction with little traffic the maps contain a noise texture with z ≈ 2–4.

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
     iterate over markers **present in the mask**, not over the metadata list (sample 43 has a vehicle without a mask, see Stage 0);
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
   *The pixel-level ablation of Stages 4–5 is already done (`run_enhancement.py`, Stage 5 table). Stage 13 repeats it at the detection level (recall/F1).*
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
| 08–09.11 | 0 ✅, 1 ✅, 2 ✅ | data extracted, loader, EDA, segments chosen |
| 10.11 | 3 ✅ (08.11) | filter bands justified (LF 1–8 Hz + HF 60–312.5 Hz) |
| 11–12.11 | 4 ✅, 5 ✅ (08.11) | energy images (HF + LF) + f-k per direction + filter bank; variant study |
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
