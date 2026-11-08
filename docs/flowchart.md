# Algorithm flowchart (source for report/figures/flowchart.png)

```mermaid
flowchart TD
    A[Load 12 consecutive files -> 2-min matrix 75000 x 52] --> B[Detrend + common-mode removal]
    B --> C[Band-pass FFT f_low..f_high]
    C --> D[RMS envelope + time decimation x25]
    D --> E[Robust per-channel normalisation + log + clip]
    E --> F[Spatial upsampling]
    F --> G1[f-k fan filter, direction +1]
    F --> G2[f-k fan filter, direction -1]
    G1 --> H[Background removal + directional filter bank]
    G2 --> H
    H --> S[Path A: strong lines\nOtsu -> oriented opening -> components -> RANSAC]
    H --> W[Path B: weak lines\nmask strong -> hysteresis -> skeleton -> HoughLinesP]
    S --> M[Grouping of candidates + average line]
    W --> M
    M --> V[Velocity km/h per object]
    V --> T[Ridge tracking -> v t profile]
    T --> K[Features -> k-means clustering]
    K --> R[Figures, tables, metrics]
```
