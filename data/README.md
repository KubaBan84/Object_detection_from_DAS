# Data

Raw archives are kept here and are **not** versioned (see `.gitignore`).

| Archive                 | Extract to         | Content                                                  |
|-------------------------|--------------------|----------------------------------------------------------|
| `20240507.rar`          | `data/real/`       | 259 files `HHMMSS.npy`, shape (6250, 52), float32        |
| `Synthetic Dataset.zip` | `data/synthetic/`  | `{id}_signal.npy`, `{id}_ground_truth.npy`, `{id}_metadata.json` (75000 x 59) |

Download (links from the project brief):
- real recordings: https://drive.google.com/file/d/1lJKLz3LsQmnAf9q5GGi6arEYBz-3CMhx/view?usp=sharing
- synthetic dataset: https://drive.google.com/file/d/1llkbY0JH7IEq__io5C4cx7WLeR1T4gvh/view?usp=sharing

Extraction: `python scripts/extract_data.py` (uses the Windows/libarchive `tar`, which reads RAR).

Real data metadata: dt = 0.0016 s (fs = 625 Hz), dx = 5.106500953873407 m, 10 s per file,
date 2024-05-07. Continuous runs: `090017-090227` (14 files) and `090252-094332` (245 files).

`annotations/manual_labels.json` - manual reference trajectories for the analysed real segments
(created with `scripts/annotate_real.py`) - **versioned**.
