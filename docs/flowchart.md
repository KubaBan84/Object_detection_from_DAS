# Algorithm flowchart

Mermaid source of the block diagram. The report figure `report/figures/flowchart.png` is drawn
with matplotlib by `scripts/make_report_figures.py` (same blocks, plus what each block removes),
so no external tool is needed.

```mermaid
flowchart TD
    A[Load 12 consecutive files + 1 file padding each side -> 2-min matrix 75000 x 52] --> B[Detrend + common-mode removal: median across channels]
    B --> HF1[HF path: band-pass 60-312.5 Hz - vibration energy]
    B --> LF1[LF path: band-pass 1-8 Hz - quasi-static load]
    HF1 --> HF2[RMS envelope + decimation x25 -> 0.04 s rows, crop padding]
    LF1 --> LF2[RMS envelope + decimation x25 -> 0.04 s rows, crop padding]
    HF2 --> HF3[log10 RMS / channel median, clip 0..p99.5]
    LF2 --> LF3[log10 RMS / channel median, clip 0..p99.5]
    HF3 --> F[f-k fan filter 5-150 km/h, zero padding, separately for direction +1 / -1]
    LF3 --> F
    F --> H[Spatial upsampling x8 + matched line-filter bank: 24 angles, 81 px -> robust z maps HF+/HF-/LF+/LF-]
    H --> S[Path A on HF: strong lines - Otsu threshold floored at 4 sigma, oriented opening, components, ridge NMS + sequential RANSAC]
    S -->|strong mask, dilated| W
    H --> W[Path B on HF and LF: weak lines - mask strong, hysteresis 2/4 sigma, skeleton, HoughLinesP + global hough_line, slope + local-orientation check]
    S --> M[Grouping: link same line - velocity + time gap - or continuation, connected components]
    W --> M
    M --> MA[Robust weighted average line per group, LF weight x2]
    MA --> MD[Clean-up: duplicates, min extent, velocity range, weak = long or clearly above noise]
    MD --> T[Ridge tracking on the f-k map HF+LF: peak per channel around the line, RANSAC quadratic t of x, refine around the curve, drop peaks within 0.5 s of the window edges]
    T --> V[Velocity: LSQ x of t = v t + b with R2, label km/h]
    T --> VT[v of t: sliding 9-channel regression; quadratic fit -> acceleration, constant / varying speed]
    V --> FE[Features per object: line energy + HF SNR on the cross-profile, FWHM, raw spectrum on the track, log speed change]
    VT --> FE
    FE --> K[Factorized k-means cv2.kmeans: thickness levels x speed-change levels, k by silhouette -> e.g. thick, varying speed]
    K --> R[Figures, tables]
    MD --> EV[Evaluation: one-to-one matching with synthetic GT / manual labels, P/R/F1 thick vs thin, velocity error, trajectory IoU]
    V --> EV
```
