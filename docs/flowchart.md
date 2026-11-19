# Algorithm flowchart

Mermaid source of the block diagram. The report figure `report/figures/flowchart.png` is drawn
with matplotlib by `scripts/make_report_figures.py` (same blocks, plus what each block removes),
so no external tool is needed.

```mermaid
flowchart TD
    A[Load 12 consecutive files + 1 file padding each side -> raw matrix 87500 x 52] --> B[Stage 4: detrend + common-mode removal - median across channels]
    B --> C[Stage 4: zero-phase FFT band-pass 60-312.5 Hz]
    C --> D[Stage 4: RMS envelope of 25-sample blocks -> 0.04 s rows, crop padding]
    D --> E[Stage 4: log10 envelope / channel median, clip 0..p99.5 -> image 3000 x 52]
    E --> F[Stage 5: f-k fan filter 5-150 km/h, separately for v > 0 and v < 0]
    F --> G[Stage 5: x8 upsampling along the cable + matched line-filter bank, 24 angles, 81 px -> robust z map per direction]
    G --> H[Stage 6: threshold Otsu, at least median + 4 sigma -> oriented opening with 9 line elements]
    H --> I[Stage 6: connected components, area and extent filter -> ridge along time + sequential RANSAC -> line candidates]
    I --> J[Stage 7: link candidates - same line or continuation - connected components]
    J --> K[Stage 7: robust weighted mean line per group, duplicate removal, plausibility 5-150 km/h and >= 8 channels]
    K --> L[Stage 8: velocity v = 1 / slope of the mean line, sign = direction]
    L --> M[Figures: waterfall with red lines and km/h labels, object table, timings]
    L --> N[Stage 9: evaluation - one-to-one matching with manual labels / synthetic GT, thick recall, precision, velocity error]
```
