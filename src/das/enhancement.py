"""Enhancement of slanted lines and suppression of everything else (plan: Stage 5).

For every band image (Stage 4) and every direction of travel (+1 / -1)::

    image --f-k fan filter (10-150 km/h, one direction)--> upsample x8 along space
          --[optional white top-hat]--> directional matched-filter bank --> robust z-score

The f-k filter removes everything that does not move like a vehicle: horizontal stripes
(simultaneous events, v -> inf), vertical stripes (static channel noise, v -> 0) and
fast seismic waves. The filter bank integrates the energy along short line segments of
all admissible slopes, which raises the SNR of thin car lines (matched filter).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import EnhancementParams
from .preprocessing import upsample_space

MAD_TO_SIGMA = 1.4826


# ======================================================================================
# f-k velocity filter
# ======================================================================================
def _raised_cosine_step(v: np.ndarray, edge: float, width: float) -> np.ndarray:
    """0 below edge - width/2, 1 above edge + width/2, Hann transition in between."""
    t = np.clip((v - (edge - width / 2)) / max(width, 1e-12), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def tukey_window(n: int, alpha: float = 0.2) -> np.ndarray:
    """Tukey (tapered cosine) window of length n."""
    if alpha <= 0:
        return np.ones(n)
    x = np.linspace(0.0, 1.0, n)
    w = np.ones(n)
    edge = alpha / 2
    lo, hi = x < edge, x > 1 - edge
    w[lo] = 0.5 * (1 + np.cos(np.pi * (x[lo] / edge - 1)))
    w[hi] = 0.5 * (1 + np.cos(np.pi * ((x[hi] - 1) / edge + 1)))
    return w


def fk_fan_mask(f: np.ndarray, k: np.ndarray, v_min: float, v_max: float, direction: int,
                taper: float = 0.2) -> np.ndarray:
    """Tapered f-k velocity-fan mask on the (unshifted) numpy FFT grid.

    The apparent velocity of an (f, k) cell is v = -f / k (numpy FFT sign convention, see
    ``spectral``). Cells with v_min <= |v| <= v_max pass; transitions are raised-cosine
    over ``taper * edge``. ``direction`` +1 keeps v > 0 (towards increasing position),
    -1 keeps v < 0, 0 keeps both. k = 0 (simultaneous events, |v| = inf) and DC are rejected.
    """
    ff, kk = f[:, None], k[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        vel = np.where(kk != 0, -ff / np.where(kk != 0, kk, 1.0), np.inf)
    speed = np.abs(vel)
    mask = _raised_cosine_step(speed, v_min, taper * v_min) * \
        (1.0 - _raised_cosine_step(speed, v_max, taper * v_max))
    mask[~np.isfinite(speed)] = 0.0
    if direction > 0:
        mask *= vel > 0
    elif direction < 0:
        mask *= vel < 0
    return mask


def fk_velocity_filter(img: np.ndarray, dt: float, dx: float, v_min: float, v_max: float,
                       direction: int = 0, taper: float = 0.2, pad_x: int = 128,
                       pad_mode: str = "zero") -> np.ndarray:
    """Keep only energy moving with apparent velocity v_min <= |v| <= v_max [m/s].

    ``direction``: +1 increasing cable position, -1 decreasing, 0 both.
    ``pad_mode`` controls the spatial boundary (the FFT is circular in space):

    - ``zero``: zero-pad to ``pad_x`` columns;
    - ``reflect``: mirror the image (2 C columns) - a stripe spanning the whole cable then
      has no edges at all, so it does not leak into the fan; the mirrored copy of a line
      has the opposite velocity and is removed by the direction mask / cropped away;
    - ``tukey``: Tukey taper (alpha 0.2) along space, then zero padding.

    The image mean is removed first. Returns an array of img.shape.
    """
    v = np.asarray(img, dtype=np.float64)
    v = v - v.mean()
    n = v.shape[1]
    if pad_mode == "reflect":
        v, nx = np.concatenate([v, v[:, ::-1]], axis=1), 2 * n
    elif pad_mode == "tukey":
        v, nx = v * tukey_window(n)[None, :], max(pad_x, n)
    elif pad_mode == "zero":
        nx = max(pad_x, n)
    else:
        raise ValueError(f"Unknown pad_mode {pad_mode!r}")
    spec = np.fft.fft2(v, s=(v.shape[0], nx))
    mask = fk_fan_mask(np.fft.fftfreq(v.shape[0], dt), np.fft.fftfreq(nx, dx),
                       v_min, v_max, direction, taper)
    return np.real(np.fft.ifft2(spec * mask))[:, :n]


# ======================================================================================
# Image-domain enhancement
# ======================================================================================
def robust_z(img: np.ndarray) -> np.ndarray:
    """(img - median) / (1.4826 MAD) of the whole image."""
    med = np.median(img)
    sigma = MAD_TO_SIGMA * np.median(np.abs(img - med))
    return (img - med) / (sigma if sigma > 0 else 1.0)


def line_snr(img: np.ndarray, q: float = 99.0, thresh: float = 4.0) -> tuple[float, float]:
    """Robust line-to-noise measure: (q-th percentile of robust z, fraction of z > thresh)."""
    z = robust_z(np.asarray(img, dtype=np.float64))
    return float(np.percentile(z, q)), float(np.mean(z > thresh))


def velocity_to_angle(v_ms: float | np.ndarray, row_dt: float, col_dx: float):
    """Velocity [m/s] -> angle of the line from the vertical (time) axis [rad].

    A line moving with v advances v*row_dt/col_dx columns per row; positive v (towards
    increasing position) gives a positive angle (columns increase downwards).
    """
    return np.arctan(np.asarray(v_ms) * row_dt / col_dx)


def angle_to_velocity(angle: float | np.ndarray, row_dt: float, col_dx: float):
    """Inverse of :func:`velocity_to_angle` -> [m/s]."""
    return np.tan(np.asarray(angle)) * col_dx / row_dt


def bank_angles(v_min: float, v_max: float, direction: int, n_angles: int, row_dt: float,
                col_dx: float) -> np.ndarray:
    """Angles uniformly spaced between the slopes of v_min and v_max for one direction."""
    a0, a1 = velocity_to_angle(np.array([v_min, v_max]) * np.sign(direction), row_dt, col_dx)
    return np.linspace(a0, a1, n_angles)


def oriented_line_kernels(length: int, angles: np.ndarray, width_sigma: float = 1.5
                          ) -> list[np.ndarray]:
    """Zero-mean, unit-norm matched line kernels (one per angle).

    Each kernel is a segment of ``length`` px through the centre at ``angle`` (from the
    vertical axis), with a Gaussian cross-profile of ``width_sigma`` px and a Hann taper
    along the segment. Subtracting the mean over the kernel box turns it into a
    "line minus surroundings" detector; the unit L2 norm makes the responses of
    different angles comparable.
    """
    kernels = []
    half = length / 2.0
    for a in np.atleast_1d(angles):
        ca, sa = np.cos(a), np.sin(a)
        hr = int(np.ceil(half * abs(ca) + 3 * width_sigma))
        hc = int(np.ceil(half * abs(sa) + 3 * width_sigma))
        r, c = np.mgrid[-hr:hr + 1, -hc:hc + 1].astype(np.float64)
        along = r * ca + c * sa
        perp = -r * sa + c * ca
        k = np.exp(-0.5 * (perp / width_sigma) ** 2) * \
            np.where(np.abs(along) <= half, 0.5 + 0.5 * np.cos(np.pi * along / half), 0.0)
        k -= k.mean()
        kernels.append((k / np.linalg.norm(k)).astype(np.float32))
    return kernels


def gabor_kernels(angles: np.ndarray, wavelength: float, sigma: float, gamma: float = 0.3
                  ) -> list[np.ndarray]:
    """Even (cosine) Gabor kernels whose stripes run along the given line angles.

    In ``cv2.getGaborKernel`` the stripes run along (-sin theta, cos theta) in (x=col,
    y=row) coordinates, so a line at angle ``a`` from the vertical needs theta = -a.
    ``gamma`` < 1 elongates the envelope along the stripes. Zero-mean, unit norm.
    """
    size = int(2 * np.ceil(3 * sigma / gamma) + 1)
    out = []
    for a in np.atleast_1d(angles):
        k = cv2.getGaborKernel((size, size), sigma, -float(a), wavelength, gamma, 0.0,
                               ktype=cv2.CV_64F)
        k -= k.mean()
        out.append((k / np.linalg.norm(k)).astype(np.float32))
    return out


def directional_filter_bank(img: np.ndarray, kernels: list[np.ndarray]
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Correlate with every kernel (cv2.filter2D) -> (max response, argmax kernel index)."""
    src = np.asarray(img, dtype=np.float32)
    best = np.full(src.shape, -np.inf, dtype=np.float32)
    idx = np.zeros(src.shape, dtype=np.int16)
    for i, k in enumerate(kernels):
        r = cv2.filter2D(src, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
        better = r > best
        best[better] = r[better]
        idx[better] = i
    return best.astype(np.float64), idx


def subtract_background(img: np.ndarray, kernel_px: int) -> np.ndarray:
    """White top-hat with a horizontal line element of ``kernel_px`` columns.

    The grey opening along space keeps only structures wider than the element (slowly
    varying background along the cable); subtracting it keeps the narrower slanted lines.
    """
    se = np.ones((1, int(kernel_px)), dtype=np.uint8)
    return cv2.morphologyEx(np.asarray(img, dtype=np.float32), cv2.MORPH_TOPHAT, se
                            ).astype(np.float64)


def ridge_filter(img: np.ndarray, sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
                 method: str = "sato") -> np.ndarray:
    """Alternative enhancer: Hessian-based ridge filters from scikit-image (bright ridges)."""
    from skimage.filters import meijering, sato

    fn = {"sato": sato, "meijering": meijering}[method]
    return fn(np.asarray(img, dtype=np.float64), sigmas=sigmas, black_ridges=False)


# ======================================================================================
# Full Stage-5 chain
# ======================================================================================
@dataclass
class Enhanced:
    """Result of :func:`enhance` for one band image."""
    row_dt: float                                    # [s] per row
    col_dx: float                                    # [m] per column (after upsampling)
    response: dict[int, np.ndarray] = field(default_factory=dict)       # direction -> z map
    angle_index: dict[int, np.ndarray] = field(default_factory=dict)    # filter bank only
    velocities: dict[int, np.ndarray] = field(default_factory=dict)     # m/s per index
    stages: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def combined(self) -> np.ndarray:
        """Maximum over directions (for display / direction-agnostic measures)."""
        return np.maximum.reduce(list(self.response.values()))


def enhance(img: np.ndarray, params: EnhancementParams, row_dt: float, col_dx: float
            ) -> Enhanced:
    """Stage-5 chain for one band image (rows = time, columns = channels).

    Stages: ``input``, ``fk_both`` and ``fk_removed`` (what the fan filter removed, both
    directions), and per direction d in {pos, neg}: ``fk_d``, ``upsampled_d``,
    ``background_d`` (if enabled) and ``response_d`` (robust z-score).
    """
    v_min, v_max = params.v_min_kmh / 3.6, params.v_max_kmh / 3.6
    up = params.space_upsample
    out = Enhanced(row_dt=row_dt, col_dx=col_dx / up)
    def fk(d: int) -> np.ndarray:
        if not params.fk_filter:          # ablation: no velocity filtering at all
            return np.asarray(img, dtype=np.float64) - np.mean(img)
        return fk_velocity_filter(img, row_dt, col_dx, v_min, v_max, d, params.fk_taper,
                                  pad_mode=params.fk_pad_mode)

    both = fk(0)
    out.stages.update(input=np.asarray(img, dtype=np.float64), fk_both=both,
                      fk_removed=(img - np.mean(img)) - both)
    for d, name in ((+1, "pos"), (-1, "neg")):
        f = fk(d)
        u = upsample_space(f, up)
        out.stages[f"fk_{name}"] = f
        out.stages[f"upsampled_{name}"] = u
        if params.background:
            u = subtract_background(u, params.bg_kernel_px)
            out.stages[f"background_{name}"] = u
        angles = bank_angles(v_min, v_max, d, params.n_angles, row_dt, out.col_dx)
        if params.enhancer == "filterbank":
            resp, idx = directional_filter_bank(
                u, oriented_line_kernels(params.kernel_length_px, angles, params.kernel_width_sigma))
            out.angle_index[d] = idx
            out.velocities[d] = angle_to_velocity(angles, row_dt, out.col_dx)
        elif params.enhancer == "gabor":
            resp, idx = directional_filter_bank(
                u, gabor_kernels(angles, params.gabor_wavelength_px, params.gabor_sigma_px))
            out.angle_index[d] = idx
            out.velocities[d] = angle_to_velocity(angles, row_dt, out.col_dx)
        elif params.enhancer in ("sato", "meijering"):
            resp = ridge_filter(u, params.ridge_sigmas, params.enhancer)
        elif params.enhancer == "none":
            resp = u
        else:
            raise ValueError(f"Unknown enhancer {params.enhancer!r}")
        out.response[d] = robust_z(resp)
        out.stages[f"response_{name}"] = out.response[d]
    return out
