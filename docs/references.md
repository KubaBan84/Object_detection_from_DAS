# Sources, references and inspiration

For every entry: WHAT was used and WHERE (module / report section).

## Data
- **Real DAS recordings**, fibre-optic cable along Jana Pawła II street, 2024-05-07, 259 files
  of 10 s (`HHMMSS.npy`, 6250 × 52, dt = 0.0016 s, dx = 5.1065 m). Course materials, Google Drive
  link in the project brief. Used for: everything on real data (report sections 1–3, 5–12).
- **Synthetic DAS dataset** (course materials, 6 samples `40`–`45`: signal, ground-truth mask,
  metadata with vehicle type / direction / lane). Used for: development and the quantitative
  evaluation against the ground truth (Stages 5–13, report section 12).
- **Course example notebook** (loading and displaying the data, link in the brief). Used for:
  the waterfall convention (time vertical, space horizontal) and the "velocity label on the
  line" figure style reproduced in `visualization.plot_detections`.
- **Manual labels** of the three real segments (`data/annotations/manual_labels.json`), drawn by
  the authors with `scripts/annotate_real.py`. Indicative reference only (report section 12).

## Libraries (allowed by the brief)
- **NumPy**: `numpy.fft` (`rfft`, `fft2`, per-channel FFT, FFT band-pass, Welch PSD, STFT and f-k
  transform - no SciPy), `lib.stride_tricks.sliding_window_view` (envelopes, STFT frames, v(t)
  windows), `linalg` (batched RANSAC solves, SVD of features), robust statistics. All modules.
- **OpenCV**: `filter2D` (directional matched-filter bank), `resize` (`INTER_LINEAR` spatial
  upsampling, `INTER_AREA` report figures), `HoughLinesP` (weak path), `morphologyEx` / `dilate`
  (oriented opening, strong-line mask), `GaussianBlur`, `getGaborKernel` (Gabor comparison),
  `kmeans` with k-means++ initialisation (clustering), `Canny` (unsuccessful baseline, Stage 13).
- **scikit-image**: `filters.threshold_otsu`, `filters.apply_hysteresis_threshold`,
  `morphology.skeletonize` / `remove_small_objects`, `transform.hough_line` /
  `hough_line_peaks` (global Hough), `measure.label` / `regionprops` / `ransac` / `LineModelND`
  (strong path), `filters.sato` / `meijering` (comparison), `exposure.equalize_adapthist`
  (CLAHE comparison).
- **Matplotlib**: all figures and the annotation GUI (`annotation.py`).
- Pillow is allowed but was not needed.
- Tooling only (not part of the algorithm): uv, pytest, Jupyter / nbconvert.

Library documentation used: numpy.org/doc, docs.opencv.org (4.x), scikit-image.org/docs,
matplotlib.org - function signatures, conventions (e.g. the `hough_line` angle convention, the
`numpy.fft` frequency ordering) and examples of the functions listed above.

## Literature (methods used, and where)
- Hough transform: P. V. C. Hough, *Method and means for recognizing complex patterns*, US Patent
  3 069 654 (1962); R. O. Duda, P. E. Hart, *Use of the Hough transformation to detect lines and
  curves in pictures*, Communications of the ACM 15(1), 1972. → global `hough_line` (Stage 6B).
- Progressive probabilistic Hough (`cv2.HoughLinesP`): J. Matas, C. Galambos, J. Kittler,
  *Robust detection of lines using the progressive probabilistic Hough transform*, Computer
  Vision and Image Understanding 78(1), 2000. → Stage 6B.
- RANSAC: M. A. Fischler, R. C. Bolles, *Random sample consensus*, Communications of the ACM
  24(6), 1981. → sequential RANSAC of strong lines (6A), trajectory RANSAC (Stage 9).
- Otsu threshold: N. Otsu, *A threshold selection method from gray-level histograms*, IEEE
  Trans. Systems, Man, and Cybernetics 9(1), 1979. → strong threshold (6A).
- Hysteresis thresholding / edge detection: J. Canny, *A computational approach to edge
  detection*, IEEE Trans. PAMI 8(6), 1986. → hysteresis of the weak path (6B), Canny baseline.
- Thinning: T. Y. Zhang, C. Y. Suen, *A fast parallel algorithm for thinning digital patterns*,
  Communications of the ACM 27(3), 1984 (algorithm of `skimage.morphology.skeletonize`). → 6B.
- Oriented matched filters for line detection: S. Chaudhuri et al., *Detection of blood vessels
  in retinal images using two-dimensional matched filters*, IEEE Trans. Medical Imaging 8(3),
  1989 - the idea of a bank of rotated line kernels with a Gaussian cross-profile (Stage 5).
- Hessian ridge filters used for comparison: Y. Sato et al., Medical Image Analysis 2(2), 1998
  (`sato`); E. Meijering et al., Cytometry A 58(2), 2004 (`meijering`); A. F. Frangi et al.,
  MICCAI 1998 (background).
- f-k (frequency-wavenumber) fan / velocity filtering: Ö. Yilmaz, *Seismic Data Analysis*,
  Society of Exploration Geophysicists, 2001 (chapter on f-k filtering). → Stage 5.
- Welch PSD: P. D. Welch, IEEE Trans. Audio and Electroacoustics 15(2), 1967. → Stage 3.
- Running moments: B. P. Welford, Technometrics 4(3), 1962; T. F. Chan, G. H. Golub, R. J.
  LeVeque, *Updating formulae and a pairwise algorithm for computing sample variances*, 1979.
  → streaming global statistics (Stage 2).
- k-means++: D. Arthur, S. Vassilvitskii, *k-means++: the advantages of careful seeding*,
  SODA 2007; silhouette: P. J. Rousseeuw, Journal of Computational and Applied Mathematics 20,
  1987; adjusted Rand index: L. Hubert, P. Arabie, Journal of Classification 2, 1985. → Stage 10.
- DAS background: A. H. Hartog, *An Introduction to Distributed Optical Fibre Sensors*, CRC
  Press, 2017; N. J. Lindsey et al., *City-scale dark fiber DAS measurements of infrastructure
  use during the COVID-19 pandemic*, Geophysical Research Letters 47(16), 2020 (vehicles as
  slanted lines in DAS waterfalls, velocity from the slope). → report sections 1 and 13.

## Code and inspiration
- No code was copied from external projects; all algorithms are implemented in `src/das/` on
  top of the libraries above.
- **AI assistant: Claude (Anthropic), used through Claude Code.** Used as a programming
  assistant for: proposing and implementing parts of the code (modules in `src/das/`, scripts,
  tests), refactoring, running the experiments and summarising their results, and drafting the
  documentation and report text. The method choices were discussed and verified by the authors
  on the data (figures, metrics, tests); the authors are responsible for the content.
