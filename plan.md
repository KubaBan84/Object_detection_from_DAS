# Implementation plan – Detecting Moving Objects in DAS Recordings (grade-4 version)

> Code and report deadline: **22.11.2026, 23:59**. Presentation: **26.11.2026 / 01.12.2026**.
> Scope: the **"Version for 3/4" with grade 4** of the brief: real data, three continuous
> 2-minute segments, detection of the **thick, clearly visible lines** and their velocities.

## Scope change (19.11.2026)

The project was first developed for grade 5 (08–12.11): thick and thin lines, two frequency
paths, both extra points (v(t) and clustering). The plan changed to grade 4, so these parts
were **removed** from the code, the tests, the documentation and the report:

| Removed | Why | Was |
|---|---|---|
| detection of the thin, weak lines (hysteresis, skeleton, two Hough transforms, masking of the strong lines) | grade-5 condition ("To receive a grade of 5, you must also detect the thinner, weaker lines") | Stage 6B |
| the low-frequency path 1–8 Hz | used only for the thin car lines | Stages 3–7 |
| ridge tracking, v(t), acceleration, constant / varying speed | extra point 1 | Stage 9 |
| object features and k-means clustering | extra point 2 | Stage 10 |
| unused options (CLAHE, smoothing, top-hat, Gabor, Meijering, f-k padding modes, robust-z normalisation, channel interpolation) | only used by the old variant studies | Stages 4–5 |

**Additional requirements for teams of three** (interactive GUI with sliders for the filter
bands, thresholds, preprocessing and visualisation): **not applicable** - we are a pair, and
nothing of it was ever implemented. `scripts/annotate_real.py` is only the labelling tool that
produced our reference labels, not that application.

The regression check (experiments log #19) showed that the remaining thick-line detection
gives **identical** line candidates and objects before and after the clean-up.

---

## 0. Requirements of the grade-4 version and where they are implemented

| # | Requirement from the brief | Stage | Module / file |
|---|---|---|---|
| 1 | Load **real data**; 3 continuous 2-min segments (pair: 2 assigned + 1 own) | 1 | `das/io.py`, `config.SEGMENTS` |
| 2 | Exploratory analysis: statistics, distribution and variability in time and space | 2 | `das/stats.py`, `scripts/run_eda.py` |
| 3 | Frequency analysis: **FFT applied independently to each channel**, STFT | 3 | `das/spectral.py`, `scripts/run_spectral.py` |
| 4 | Noise filtering: frequency band, downsampling, filters, standardisation | 4 | `das/preprocessing.py` |
| 5 | Processing so that **only moving objects** (slanted lines) are visible | 5 | `das/enhancement.py` |
| 6 | Detection of the **thick lines** (tram / heavy truck) | 6 | `das/detection.py` |
| 7 | Grouping of line candidates and averaging the line of a group | 7 | `das/grouping.py` |
| 8 | **Velocity** of each object (km/h, direction), figure as in the brief | 8 | `das/velocity.py`, `visualization.plot_detections` |
| 9 | Number and accuracy of the detected objects | 9 | `das/evaluation.py`, `scripts/evaluate_*.py` |
| 10 | Detailed results for the 3 two-minute segments | 10 | `das/pipeline.py`, `scripts/run_pipeline.py` |
| 11 | Experiments (successful and unsuccessful) | 11 | `scripts/run_experiments.py`, `docs/experiments_log.md` |
| 12 | Block diagram with a description of every step; before / after visualisations incl. removed noise | 12 | `docs/flowchart.md` → `report/figures/flowchart.png`, `run_pipeline.py` figures |
| 13 | Report: analysis, conclusions, errors, limitations, sources | 12 | `report/report.ipynb` + `.html`, `docs/references.md` |
| 14 | Code without errors or warnings, vectorised, readable | all, 13 | `tests/`, `pyproject.toml` (`filterwarnings=error`) |

**Technology constraints.** The algorithm only uses **NumPy, OpenCV, scikit-image and
Matplotlib** (Pillow is allowed but not needed). No neural networks.

- `scipy` is not on the list: band-pass filters, STFT and Welch are written with `numpy.fft`.
- `pandas` / `sklearn` are not on the list: tables are written with `csv` / `json`.
- uv, pytest, Jupyter and nbconvert are tooling only.

---

## 1. Data facts (verified on the files in `data/`)

| | Real data | Synthetic data (validation only) |
|---|---|---|
| Archive | `data/20240507.rar` (259 files) | `data/Synthetic Dataset.zip` (6 samples × 3 files) |
| File | `HHMMSS.npy`, shape **(6250, 52)**, float32 | `{id}_signal.npy` **(75000, 59)** + `{id}_ground_truth.npy` + `{id}_metadata.json` |
| Layout | axis 0 = time, axis 1 = space | same |
| dt / fs | 0.0016 s / **625 Hz** (Nyquist 312.5 Hz) | same |
| dx | 5.1065 m → 52 channels ≈ 266 m | same (59 channels) |
| Continuous runs | **090017–090227** (14 files) and **090252–094332** (245 files) | each sample = 2 min |
| GT | none → manual labels (Stage 9) | mask with vehicle markers + type (car / tram / van / truck), direction |

Design conclusions:

- A 2-minute segment is **12 consecutive files** = 75 000 × 52 samples, inside one continuous
  run (`io.load_segment` checks this). One file of padding on each side is loaded and cropped
  after the envelope, so the FFT edge effects stay outside the segment.
- Vehicles at 5–150 km/h cross one channel in 0.12–3.7 s, so decimation ×25 (0.04 s per row,
  3000 rows per 2 min) loses nothing.
- 52 columns make the lines almost vertical: the space axis is interpolated ×8 (0.64 m / px)
  before the directional filters, so that the angles are resolved.
- Image convention as in the figure of the brief: **time vertical (downwards), space
  horizontal**. A line with velocity v has the slope Δcol/Δrow = v·row_dt/col_dx.
- The synthetic data are used only as a **validation with a true ground truth** (thick recall,
  velocity error). The brief allows this ("may still be used for initial development and
  testing"). All results of the report are on real data.

### 1.1 Analysed segments (pair work: Kuba + Mateusz)

| Name | Owner | Files (inclusive) | Time span (2024-05-07) | Kind |
|---|---|---|---|---|
| `seg_kuba` | Kuba | `090452.npy` – `090642.npy` | 09:04:52 – 09:06:52 | assigned |
| `seg_mateusz` | Mateusz | `093222.npy` – `093412.npy` | 09:32:22 – 09:34:22 | assigned |
| `seg_own` | Kuba + Mateusz | `091152.npy` – `091342.npy` | 09:11:52 – 09:13:52 | own |

All three lie in the run 090252–094332 and do not overlap. **Why `seg_own` = 091152–091342**
(EDA ranking, `outputs/results/eda_candidates.csv`):
- it has the highest traffic activity of all admissible windows: 25.8 % of the (1 s × channel)
  cells are above 3× the channel reference, 10 traffic episodes (`seg_kuba` 11.8 % / 6,
  `seg_mateusz` 9.9 % / 2);
- it has traffic in **both directions** with crossing lines, while the assigned segments are
  dominated by one direction;
- one file of margin is available on each side (padding).

---

## 2. Code conventions

- All logic lives in `src/das/`; the scripts in `scripts/` parse arguments, call the library and
  save results and figures.
- **Every parameter** is a field of a dataclass in `config.py`; results are saved together with
  the parameters (JSON).
- **No per-pixel loops**: vectorised NumPy, `cv2.filter2D`, `np.fft`, reshapes for the envelope.
  Loops only over kernels, components, objects and files.
- Processing functions keep their intermediate images (`stages`) for the before / after figures.
- Type hints, docstrings and comments in English.
- No warnings: pytest runs with `filterwarnings=error`, the scripts with `python -W error`.
- Environment: **uv** (Python 3.12, `pyproject.toml`, `uv.lock`); `requirements.txt` for pip
  (`uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`).

---

## 3. Stages

Each stage: goal → files → method → result / DoD. Numbers are from the current code.

### Stage 0 – Environment and data ✅
`scripts/extract_data.py` extracts the RAR (real) and ZIP (synthetic) archives into `data/real/`
and `data/synthetic/` and verifies them: 259 real files, all (6250, 52) float32, no NaN / inf,
two continuous runs; 6 synthetic samples (ids 40–45) with consistent metadata. Sample 43 lists a
vehicle that is missing from its mask, so the references are built from the mask.

### Stage 1 – Data loading ✅
`src/das/io.py`, `tests/test_io.py`.
- `list_real_files`, `parse_timestamp`, `find_continuous_runs` (neighbours 10 s apart).
- `load_segment(start, n_files, pad_files)` → `Segment` (data, start time, padding, dt, dx,
  `core`, `crop`); a segment that does not fit into one run raises `ValueError`.
- `load_named_segment(name)` for `config.SEGMENTS`; `load_synthetic(id)` → (signal, gt, meta).

### Stage 2 – Exploratory data analysis ✅
`src/das/stats.py`, `scripts/run_eda.py` (≈25 s).
- Streaming global statistics over all 259 files (`RunningStats`, merged moments, histograms).
- Per segment: descriptive statistics, per-channel / per-second statistics, correlation of the
  channels, activity metrics, figures (waterfall linear vs log, raw zoom, variability, box plots,
  histograms, correlation), ranking of the candidate own segments.

Key findings (they drive the method):

| Finding | Numbers | Consequence |
|---|---|---|
| Extremely heavy-tailed distribution | std 4.85e-7 vs robust σ 1.15e-7, excess kurtosis 435 | median / MAD statistics, log compression, robust thresholds |
| Quantised values | step 2.5e-9, 1–4 % exact zeros | eps in every log / division |
| Channel 25 (≈128 m) flagged by the bad-channel test | RMS bump over channels 22–28 with short horizontal dashes | a physical hotspot, not a fault: kept; the per-channel normalisation and the f-k filter handle it |
| Raw signal is oscillatory | ± vibration under a moving envelope | lines are visible only in the energy envelope (RMS) |
| Lines are clear only after log10(RMS / channel median) | waterfall linear vs log | log compression + per-channel normalisation |
| Most traffic moves towards decreasing x, some lines cross | waterfalls, f-k | direction-separated f-k filtering |
| Traffic is bursty | per-file median RMS varies 5× | thresholds relative to the robust noise level |

### Stage 3 – Frequency analysis ✅
`src/das/spectral.py`, `scripts/run_spectral.py` (≈25 s, the band grid is cached).
- `channel_fft`: rfft **independently for every channel** (Hann window).
- `welch_psd` (global PSD of all files), `stft` of three channels (41 m, 128 m hotspot, 214 m).
- f-k spectra of the raw signal and of the band envelopes (sign convention: v = −f/k).
- No tonal lines > 10 dB on the channel median (no 50 Hz) → **no notch filter**.
- Vehicle / background contrast: maximum 38 dB around 80 Hz.
- **Band choice by an objective criterion**: for a grid of 16 × 19 bands, the line SNR (p99 of
  the robust z of the fan-filtered log envelope). Best band **60–312.5 Hz (5.84)**; the best
  band with f_high ≤ 10 Hz (1–8 Hz) gives 5.39 and broadband 4.88. Visual check: 1–8 Hz is
  blind for x < 55 m and for some lanes, 60–312.5 Hz has a clean background and all thick lines.
  → `PreprocessingParams.band_low / band_high`.

### Stage 4 – Noise removal, energy image ✅
`src/das/preprocessing.py`, `tests/test_preprocessing.py`.
`preprocess(x, params, fs, dx, crop)` → `Preprocessed` (image + all stages), ≈0.9 s per segment:
1. `detrend_channels` – mean and linear trend of every channel.
2. Common-mode removal – the median over channels at every sample (cable-wide events).
3. `bandpass_fft` – zero-phase FFT band-pass 60–312.5 Hz with raised-cosine edges.
4. `envelope_downsample` – RMS of 25-sample blocks → 0.04 s rows; padding cropped.
5. `normalize_envelope` – log10(env / channel median) (gain equalisation + log compression),
   clipped to [0, p99.5].

### Stage 5 – Only moving objects visible ✅
`src/das/enhancement.py`, `tests/test_enhancement.py`.
`enhance(image, params, row_dt, col_dx)` → `Enhanced` (response map per direction), ≈3–4 s:
1. `fk_velocity_filter` – 2-D FFT, tapered fan 5–150 km/h, **separately for v > 0 and v < 0**,
   k = 0 rejected, zero padding of the space axis to 128 columns.
   Removes horizontal stripes (v → ∞), vertical stripes (v → 0), the hotspot dashes and the
   traffic of the other direction.
2. `upsample_space` ×8 (cv2.resize, linear).
3. `directional_filter_bank` – 24 zero-mean, unit-norm line kernels (81 px, Gaussian
   cross-profile σ = 2 px) between the slopes of 5 and 150 km/h, `cv2.filter2D`, maximum over
   the kernels (matched filter).
4. `robust_z` – (response − median) / (1.4826 MAD).

### Stage 6 – Detection of the thick lines ✅
`src/das/detection.py`, `tests/test_detection.py`. `detect_lines(enhanced, params, v_range)`,
≈1 s per segment, on the map of every direction:
1. Threshold: Otsu of the positive response, **at least median + 4 σ** (robust σ).
2. Connected components (`label`, `regionprops`): area ≥ 1500 px, extent ≥ 40 m.
   (An oriented morphological opening between steps 1 and 2 was removed on 20.11 - neutral in
   the ablation, experiments log #26.)
3. `ridge_along_time` – local maxima along time (NMS ±5 rows), one ridge pixel per column.
4. `sequential_ransac` – up to 4 lines per component (two touching trams → two lines; a
   non-overlapping later line only extends the line it continues).
6. Line thickness = mean number of component pixels per column assigned to the line.

### Stage 7 – Grouping and averaging ✅
`src/das/grouping.py`, `tests/test_grouping.py`. `build_objects(...)` → (objects, groups):
1. `link_matrix` (vectorised): same direction AND (same line: |Δv| < 8 km/h or 15 % and time
   gap at the midpoint < 1.5 s, widened to half the line thickness; OR continuation: x-ranges
   follow each other within ±20 m, the lines meet, velocity ratio ≤ 2).
2. `connected_components` (min-label propagation) → groups.
3. `merge_group` – robust weighted least squares t(x) on points along all segments of a group
   (weight = score, 3σ-MAD re-weighting) → one line per vehicle.
4. `remove_duplicates` – overlapping lines of one direction; plausibility (≥ 8 channels,
   5–150 km/h).

### Stage 8 – Velocity ✅
`src/das/velocity.py`, `tests/test_velocity.py`.
- v = 1 / slowness of the averaged line [m/s] × 3.6 → km/h; sign = direction
  (+ towards increasing cable position).
- `plot_detections` – red lines with "xx.xx km/h" labels on the raw waterfall (as in the brief).
- `write_objects_csv` – object table: id, direction, velocity, passage time (also as clock
  time), extent, support, score, thickness, fit RMS, number of candidates.

### Stage 9 – Evaluation ✅
`src/das/evaluation.py`, `src/das/annotation.py`, `scripts/evaluate_real.py`,
`scripts/evaluate_synthetic.py`, `scripts/annotate_real.py`, `data/annotations/manual_labels.json`.
- **References.** Real: 52 lines marked by hand (Kuba) on the raw waterfall (`annotate_real.py`)
  with a thick / thin label; 5 strokes outside the detector fan or shorter than 30 m are not
  scored (47 scored: 23 thick, 24 thin). Synthetic: the ground-truth mask (49 thick, 85 thin).
- **Grade-4 scoring.** One-to-one matching with all references (gates 1.5 s / 20 km/h).
  - **thick recall** is the main metric;
  - **precision** = matched detections / detections (a loud car matched to a thin label is a
    correct vehicle, not a false positive);
  - F1 of the two; thin labels found are a bonus;
  - |v| error of the matched pairs; false positives split into "near a label" / "no label".

Results (final parameters):

| | thick recall | precision | F1 | thick \|v\| error | thin found (bonus) |
|---|---|---|---|---|---|
| real, 3 segments (manual labels) | **0.913** (21/23) | 0.791 (34/43) | **0.847** | 3.6 km/h | 13/24 |
| synthetic, 6 samples (GT) | **0.980** (48/49) | 0.913 (63/69) | **0.945** | 2.9 km/h (MAPE 3.8 %) | 15/85 |

- 7 of the 9 real false positives lie on a labelled line (fragments in the halo of trams).
- The two missed real thick labels are in `seg_own` (dense traffic in both directions).
- The result does not depend on the matching gates from 1.5 s / 20 km/h upwards.

### Stage 10 – End-to-end pipeline and results for the 3 segments ✅
`src/das/pipeline.py`, `scripts/run_pipeline.py`, `tests/test_pipeline.py`.
- `run_pipeline` chains Stages 4 → 8 and times every stage; `save_result` writes the object
  table (CSV), a JSON summary (parameters, timings, counts, thresholds) and optionally the stage
  images (`--cache`).
- `run_pipeline.py --segment all` writes for every segment the figures of every step:
  `01_raw`, `02_preprocessing`, `03_enhancement`, `04_detection_pos|neg`, `05_grouping`,
  `06_result`.
- Real segments: `seg_kuba` 13 objects (v > 0: 2, v < 0: 11), `seg_mateusz` 17, `seg_own` 13.
- Time ≈ 3.5–6 s per 2-min segment (≈ 70 % of it is the filter bank).
- Generality: `--start HHMMSS` (any window) and `--synthetic <id>` run the same code; nothing is
  specific to the 3 segments.

### Stage 11 – Experiments ✅
`scripts/run_experiments.py` (29 variants × 9 recordings, ≈3.5 min with 9 processes),
`docs/experiments_log.md` (#19–#24).
- **Ablation**, **method comparison** (Sato, fixed 6 σ, other bands, Canny + HoughLinesP
  baseline) and **sweeps**.
- **Selection rule**:
  - the synthetic F1 decides, and a gain must exceed 0.01 (one object ≈ 0.007);
  - such a gain is vetoed if the real F1 drops by more than 0.01;
  - between synthetic ties, the real F1 decides only for a gain > 0.02.
- Result: **no variant passes the rule, so the final parameters are unchanged.**
  - Every step matters, and the real data show it most clearly: without log compression the
    real F1 drops 0.85 → 0.38, without the f-k filter → 0.67, without the filter bank → 0.68,
    and without the 4 σ floor → 0.56.
  - 1–8 Hz, 20–312.5 Hz and 121-px kernels are better only on the simulator (vetoed).
  - The Canny + Hough baseline finds no real thick line correctly.

### Stage 12 – Report and block diagram ✅
`report/report.ipynb` + `report/report.html`, `scripts/make_report_figures.py`,
`docs/flowchart.md`, `docs/references.md`.
- The notebook explains every step with its formula, the key code (`inspect.getsource`) and
  before / after figures. It reads all numbers from `outputs/results/` and runs the pipeline
  live on `seg_kuba`, checking it against the saved table.
- `make_report_figures.py` draws the block diagram and copies the downscaled figures
  (`--regenerate` reruns every script first).

### Stage 13 – Quality control and submission ✅
- `uv run pytest` (warnings are errors) passes; all scripts run with `python -W error`.
- The submission package is listed in `README.md`. **Left to the user:** the authors' full names
  (optional) and committing / submitting before 22.11.2026 23:59.

---

## 4. Risks and limitations (for the report)

| Risk / limitation | Effect | Response |
|---|---|---|
| No ground truth for real data | the scores depend on rough manual labels | synthetic validation; matching gates studied; labels never used for tuning beyond tie-breaks |
| Synthetic ≠ real | settings better on the simulator lose real lines | selection rule with a real-data veto |
| Loud cars pass the thick-line threshold | some "thick" detections are cars | counted as correct vehicles; the brief asks to detect the thick lines, not to classify |
| Halo / crossings of trams | fragments → false positives near labelled lines | grouping (continuation, thickness-widened tolerance), duplicate removal |
| Short cable section (266 m) | few points per line, curved lines approximated by one straight line | velocity = mean velocity over the section |
| Fan 5–150 km/h | stopped or very fast objects are removed | physically plausible for street traffic |

---

## 5. Repository structure

```
Object_detection_from_DAS/
├── plan.md                      # this plan
├── README.md                    # installation, usage, layout, submission package
├── pyproject.toml, uv.lock, .python-version, requirements.txt
├── Project 1 2026.pdf           # project brief
├── data/
│   ├── README.md                # data description and download links
│   ├── real/, synthetic/        # extracted data (ignored by git)
│   └── annotations/manual_labels.json        # manual labels of the 3 real segments
├── src/das/
│   ├── config.py                # constants, paths, SEGMENTS, parameter dataclasses
│   ├── io.py                    # loading, continuity, segments                (Stage 1)
│   ├── stats.py                 # EDA, streaming statistics                    (Stage 2)
│   ├── spectral.py              # per-channel FFT, Welch, STFT, f-k            (Stage 3)
│   ├── preprocessing.py         # detrend, common mode, band-pass, envelope    (Stage 4)
│   ├── enhancement.py           # f-k fan filter, directional filter bank      (Stage 5)
│   ├── detection.py             # thick-line detection                         (Stage 6)
│   ├── grouping.py              # candidate grouping, DetectedObject            (Stage 7)
│   ├── velocity.py              # velocity, object table                       (Stage 8)
│   ├── evaluation.py, annotation.py   # metrics, manual labels             (Stage 9)
│   ├── pipeline.py              # end-to-end chain, saving                     (Stage 10)
│   └── visualization.py         # plots
├── scripts/                     # extract_data, run_eda, run_spectral, run_pipeline,
│                                # evaluate_real, evaluate_synthetic, annotate_real,
│                                # run_experiments, make_report_figures
├── tests/                       # unit tests per module
├── report/                      # report.ipynb, report.html, figures/
├── docs/                        # flowchart.md, experiments_log.md, references.md, presentation.md
└── outputs/                     # generated results (ignored): figures/, results/, cache/
```
