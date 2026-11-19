# Presentation notes (26.11 / 01.12.2026)

A short live demo, the key figures, and a one-line justification for every step and parameter.
All numbers come from the report (`report/report.html`); "log #N" is `docs/experiments_log.md`.

## 1. Live demo (≈2 min)

Before class: `uv sync`, data extracted (`uv run python scripts/extract_data.py`), laptop on
mains power (on battery the CPU throttles and the timings vary).

```bash
uv run python -W error scripts/run_pipeline.py --segment seg_kuba   # ≈5 s, prints the object table
```

Then show `outputs/figures/pipeline/seg_kuba/`, step by step from `01_raw` to `06_result`
(velocities on the raw waterfall, style of the brief).

Alternatives if time allows:
- `--start 092002`: any other 2-min window; nothing is tied to the three segments.
- `--synthetic 42`: a synthetic sample with the ground truth overlaid.

## 2. Talk outline (≈8 min) and figures (`report/figures/`)

| # | Topic | Figure | Message |
|---|---|---|---|
| 1 | Data | `eda_global_overview`, `eda_seg_kuba_waterfall` | heavy tails (kurtosis 435); lines visible only in the log energy |
| 2 | Frequencies | `spectral_seg_kuba_fft`, `spectral_band_selection` | FFT per channel; band 60–312.5 Hz chosen by the line SNR (5.84, best of 16 × 19 bands) |
| 3 | Block diagram | `flowchart` | band-pass → envelope → log → f-k fan per direction → filter bank → threshold + opening → RANSAC → grouping → velocity |
| 4 | Noise removal | `step_seg_kuba_preprocessing`, `step_seg_kuba_enhancement` | before / after / removed for every step; the f-k fan leaves only slanted lines |
| 5 | Detection | `step_seg_kuba_detection_neg`, `step_seg_own_grouping` | Otsu ≥ 4 σ, oriented opening, components, ridge + sequential RANSAC; one averaged line per vehicle |
| 6 | Results | `result_seg_*_result` | 13 / 17 / 13 objects with velocities |
| 7 | Evaluation | `eval_real_summary`, `eval_synthetic_summary` | real thick recall 21/23, precision 0.79; synthetic thick 48/49, \|v\| error 2.9 km/h |
| 8 | Experiments | `experiments` | every step is needed; better-on-simulator settings lose real lines; Canny + Hough fails |
| 9 | Limitations | section 13 | no real GT, loud cars, tram halo / crossings, 266 m section |

## 3. Why each step (likely questions)

**Pre-processing** (`PreprocessingParams`)
- *Common-mode median*: subtracts events that the whole cable sees at once. Without it the real
  F1 is 0.823 instead of 0.847.
- *Band 60–312.5 Hz*: the best of 16 × 19 bands by line SNR after the fan filter (5.84; broadband
  4.88; 1–8 Hz 5.39). 1–8 Hz is blind for x < 55 m and loses real thick lines (recall 0.57).
- *RMS of 25 samples (0.04 s rows)*: 150 km/h needs 0.12 s per channel, so nothing is lost. Plain
  decimation aliases the oscillation (real F1 0.770).
- *log10(env / channel median)*: the most important step. Without it the real F1 drops to 0.376.

**Enhancement** (`EnhancementParams`)
- *f-k fan 5–150 km/h per direction*:
  - removes horizontal (v → ∞) and vertical (v → 0) structures and the hotspot dashes at 128 m;
  - separates crossing lines.
  - Without it: real F1 0.673.
- *×8 upsampling + 24 line kernels of 81 px*: 52 columns are too few to resolve the angles; the
  matched filter integrates the energy along the line. Without it: real F1 0.684. Sato instead:
  0.605 (and 2× slower).

**Detection** (`DetectionParams`)
- *Otsu with a 4 σ floor*: Otsu adapts to the image; the floor stops it from splitting pure noise
  in a direction without trams. No floor: real precision 0.39; 5 σ / 6 σ lose real thick lines
  (recall 0.83 / 0.57).
- *Oriented opening (41 px, 9 slopes)*: keeps only elongated structures in the admissible slopes.
  It is neutral after the f-k filter and the filter bank, but guards against streaks.
- *Sequential RANSAC on the ridge*: splits two touching trams; per-column centroids failed there
  (log #16).

**Grouping** (`GroupingParams`)
- *Link if |Δv| < 8 km/h or 15 % and time gap < 1.5 s, or continuation*: merges the fragments of a
  tram. Without linking: synthetic precision 0.84, real F1 0.758.

**Velocity**
- *v = 1 / slope of the robust mean line*: synthetic |v| error 2.9 km/h for trams / trucks
  (MAPE 3.8 %), real difference to the slope of the labels 3.6 km/h.

**General**
- *Nothing hardcoded*: every parameter is a field of `src/das/config.py`, all thresholds are
  relative to the robust σ of each image, and the same code runs on 59-channel synthetic data and
  on any window.
- *Vectorised*: no per-pixel loops. ≈ 3.5–6 s per 2-min segment, ≈ 70 % of it is the filter bank.
- *Selection of the parameters*: synthetic F1 decides; the real labels only veto or break ties,
  so the method is not tuned on its own test reference (log #24).
- *Why the real precision is lower*: the labels are indicative, and the real data have a hotspot,
  tram halos and crossings that the simulator does not have.
