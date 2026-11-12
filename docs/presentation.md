# Presentation notes (26.11 / 01.12.2026)

A short live demo, the key figures, and a one-line justification for every step and parameter.
All numbers come from the report (`report/report.html`); "log #N" is `docs/experiments_log.md`.

## 1. Live demo (≈2 min)

Before class: `uv sync`, data extracted (`uv run python scripts/extract_data.py`), laptop **on
mains power** (on battery the CPU throttles and timings vary up to 4×).

```bash
uv run python -W error scripts/run_pipeline.py --segment seg_kuba   # ≈6-10 s, prints the object table
```

Then show `outputs/figures/pipeline/seg_kuba/`. The result figure has velocities on the raw
waterfall (style of the brief) and the tracks coloured by cluster; there is also v(t).

Alternatives if time allows:
- `--start 092002`: any other 2-min window; this shows nothing is tied to the three segments.
- `--synthetic 42`: a synthetic sample with metrics against the ground truth.

## 2. Talk outline (≈8 min) and figures (`report/figures/`)

| # | Topic | Figure | Message |
|---|---|---|---|
| 1 | Data | `eda_global_overview`, `eda_seg_kuba_waterfall` | heavy tails (kurtosis 435), thin lines only visible after log compression |
| 2 | Frequencies | `spectral_seg_kuba_fk`, `spectral_band_selection` | FFT per channel; bands chosen by line SNR: HF 60-312.5 Hz + LF 1-8 Hz beat one broad band |
| 3 | Block diagram | `flowchart` | two frequency paths → f-k fan → filter bank → strong / weak detection → grouping → tracking → clustering |
| 4 | Noise removal | `stage4_seg_kuba_hf`, `stage5_seg_kuba_hf`, `stage5_seg_own_final_maps` | before / after / removed for every step; the f-k fan leaves only slanted lines |
| 5 | Detection | `stage6_seg_own_lf_neg`, `stage7_seg_own_candidates` | 6A thick (Otsu + oriented opening + RANSAC), 6B thin (hysteresis + skeleton + 2× Hough), one averaged line per vehicle |
| 6 | Results | `result_seg_*_result`, `result_seg_*_velocity`, `stage10_seg_own_clusters` | 15 / 19 / 18 objects; velocities and v(t) |
| 7 | Evaluation | `eval_synthetic_summary`, `eval_real_summary` | synthetic F1 0.93 (thick R 1.00, thin R 0.88, \|v\| error 1.9 km/h); real F1 0.79 vs our labels |
| 8 | Experiments | `tuning_ablation`, `tuning_methods` | ablation; Canny + Hough baseline fails (0 correct on real data) |
| 9 | Limitations | section 13 | no real GT, convoys / crossings, unknown cable geometry, 266 m section |

## 3. Why each step (likely questions)

**Pre-processing** (`PreprocessingParams`)
- *Common-mode median*: subtracts events that the whole cable sees at once. Without it the
  synthetic F1 is 0.900 instead of 0.929.
- *HF 60-312.5 Hz / LF 1-8 Hz*: grid of 16 × 19 bands scored by line SNR after the fan filter.
  - HF scores 5.84 and LF 5.39; a single broad band scores 4.88.
  - The paths complement each other: LF gives sharp car lines but is blind for x < 55 m; HF has a clean background and all trams.
- *RMS of 25 samples (0.04 s rows)*: 150 km/h needs 0.12 s per channel, so nothing is lost.
  - Plain decimation aliases (F1 0.874).
  - mean |x| gives 0.893.
- *log10(env / channel median)*: the most important step.
  - Without it: F1 0.82 synthetic / 0.50 real; thin recall 0.65 / 0.42.
  - Robust z: same synthetic F1, but it loses 2 trams.

**Enhancement** (`EnhancementParams`)
- *f-k fan 5-150 km/h per direction*:
  - removes horizontal (v → ∞) and vertical (v → 0) structures and the hotspot dashes at 128 m;
  - separates crossing lines.
  - The lower limit is 5 km/h because 10 km/h removed a slow synthetic vehicle.
  - Without the filter: real F1 0.56.
- *Zero padding to 128 columns*: limits the wrap-around of the circular FFT.
  - Alternatives: reflect 0.914, none 0.908, Tukey 0.903.
- *×8 upsampling + 24 line kernels of 81 px*: 52 columns are too few for angles; the matched filter integrates along the line.
  - Sweep: 41 px 0.895, 121 px 0.876, 12 angles 0.905.
  - Sato / Meijering / Gabor: object precision ≤ 0.53.

**Detection** (`DetectionParams`)
- *Otsu with a 4 σ floor* (6A): Otsu adapts to the image. The floor stops it from splitting pure
  noise in a direction without trams.
  - 6 σ missed wide, diffuse real tram lines.
  - 5 σ was better on the labels but lost a thick line, so it was rejected.
- *Oriented opening (41 px, 9 slopes)*: removes specks and horizontal / vertical streaks. It only
  keeps elongated structures in the admissible slopes.
- *Sequential RANSAC*: splits two touching trams. Per-column centroids failed there (log #19).
- *Masking strong lines ±0.5 s × ±4 m* (6B): the halo of a tram would produce false thin lines
  (+3 F1 points).
- *Hysteresis 2 / 4 σ*: a clear synthetic optimum, and the real F1 is flat (0.73-0.81) over the
  grid. A single threshold gives thin recall 0.81 instead of 0.88.
- *HoughLinesP + global `hough_line`*: they find different lines (each alone: 0.887 / 0.902,
  both: 0.929).
- *Local-orientation check 10°*: rejects Hough segments that cut across the noise texture.

**Grouping** (`GroupingParams`)
- *Link if |Δv| < 8 km/h or 15 % and time gap < 1.5 s*: this was the only change in Stage 13.
  - It is identical on synthetic data and gives real F1 0.721 → 0.788.
  - It merges fragments of strong vehicles.
- *Weak object ≥ 60 m or ≥ 5 σ*: short and faint objects are typical of the noise texture (log #29).

**Velocity / v(t)** (`TrackingParams`)
- *Ridge tracking per channel + RANSAC quadratic*: lowers the |v| error from 2.89 to 1.89 km/h.
  It rejects peaks of crossing vehicles.
- *v(t) from a 9-channel (46 m) sliding fit*: the error is 2.29 km/h. Wider windows smooth more,
  but real braking can be abrupt.
- *Edge margin 0.5 s*: a ridge cut by the window end produced up to 350 km/h.

**Clustering** (`ClusteringParams`)
- *Factorized k-means* (`cv2.kmeans`): thickness × speed change, k ∈ {2, 3} by silhouette, so the
  cluster names are the categories of the brief.
  - A joint k-means on all features collapses (ARI 0.12).
  - Synthetic thickness purity is 0.92.
  - On real data the clusters describe loudness, not vehicle type.

**General**
- *Nothing hardcoded*: every parameter is a field in `src/das/config.py`, and all thresholds are
  relative to the robust σ of each image. The same code runs on 59-channel synthetic data and on
  arbitrary windows.
- *Vectorised*: no per-pixel loops. Loops run only over bands, kernels, objects and channels.
  - Run time ≈ 6-10 s per 2-min segment, ≈12× real time.
  - The filter bank is ≈70 % of the time.
- *Tests*: 162 unit tests, warnings are errors.
- *Why the real F1 is lower*: the labels are indicative, and real data has a hotspot, a noise
  texture and crossings that the simulator does not have.
