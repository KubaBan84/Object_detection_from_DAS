"""Clustering of detected objects (plan: Stage 10, extra 2). cv2.kmeans + NumPy metrics.

    feature matrix (objects x features, physical units), features in property groups
    (thickness: line energy; speed change: log dv, log |a|, log std v(t))
      --standardisation (z-score, NaN -> centre, clip +-4)--> Z
      factorized (default): per group cv2.kmeans (k-means++ init, 10 restarts), k by the
        silhouette -> levels ordered low -> high (thin / thick, constant / varying speed)
        -> cluster = combination of the levels, e.g. "thick, varying speed"
      joint (comparison): one k-means on all groups (equal group weights), k = 2..6 by
        the silhouette; names from the mean z of every group in the cluster
    labels --> centroids in physical units;  Z --SVD--> 2-D PCA projection (scatter plot)

Validation helpers: purity against known classes (synthetic vehicle types), the adjusted
Rand index between two partitions (k-means vs a simple rule baseline) - NumPy only, since
sklearn is not an allowed library.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_otsu

MAD_TO_SIGMA = 1.4826


# ======================================================================================
# Standardisation and k-means
# ======================================================================================
def standardize(X: np.ndarray, robust: bool = True, clip: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature standardisation -> (Z, centre, scale).

    ``robust``: median and 1.4826 x MAD (a few extreme objects - e.g. a very loud tram -
    do not compress all other objects into one point); else mean and std. Missing
    values (NaN, e.g. objects without a track) are set to the centre, i.e. z = 0. A
    constant feature gets scale 1. ``clip`` > 0 clips Z to +-clip.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 0:
        return X.copy(), np.zeros(X.shape[1]), np.ones(X.shape[1])
    finite = np.isfinite(X)
    centre = np.zeros(X.shape[1])
    scale = np.ones(X.shape[1])
    for j in range(X.shape[1]):                       # loop over features (columns) only
        col = X[finite[:, j], j]
        if col.size == 0:
            continue
        if robust:
            centre[j] = np.median(col)
            s = MAD_TO_SIGMA * np.median(np.abs(col - centre[j]))
            if s <= 0:                                # >50 % identical values (e.g. counts)
                s = col.std()
        else:
            centre[j], s = col.mean(), col.std()
        scale[j] = s if s > 0 else 1.0
    Z = np.where(finite, (X - centre) / scale, 0.0)
    if clip > 0:
        Z = np.clip(Z, -clip, clip)
    return Z, centre, scale


def flatten_groups(groups) -> list[str]:
    """Feature names of ((group, (features...), high name, low name), ...) in order."""
    return [f for g in groups for f in g[1]]


def group_weights(groups, equal: bool = True) -> np.ndarray:
    """Per-feature weight: 1/sqrt(group size) (each group has the same total variance
    contribution after standardisation) or 1 for all features."""
    return np.concatenate([np.full(len(g[1]), 1.0 / np.sqrt(len(g[1])) if equal else 1.0)
                           for g in groups])


def kmeans(X: np.ndarray, k: int, attempts: int = 10, seed: int = 0, max_iter: int = 100
           ) -> tuple[np.ndarray, np.ndarray, float]:
    """cv2.kmeans wrapper -> (labels, centers, compactness).

    k-means++ initialisation, ``attempts`` restarts (the best compactness is kept), the
    OpenCV RNG seeded for reproducibility. Labels are renumbered by decreasing cluster
    size (0 = largest), so runs are comparable.
    """
    data = np.ascontiguousarray(X, dtype=np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1e-4)
    compactness, labels, centers = cv2.kmeans(data, k, None, crit, attempts,
                                              cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    order = np.argsort(-np.bincount(labels, minlength=k), kind="stable")
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[labels], centers[order].astype(np.float64), float(compactness)


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix (vectorised, ||a||^2 + ||b||^2 - 2 a.b)."""
    sq = np.sum(X ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    return np.sqrt(np.clip(d2, 0.0, None))


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Silhouette coefficient of every sample (vectorised).

    a(i) = mean distance to the other members of its cluster, b(i) = smallest mean
    distance to another cluster, s = (b - a) / max(a, b); 0 for singleton clusters.
    The per-cluster sums come from one matrix product D @ onehot.
    """
    labels = np.asarray(labels)
    uniq, lab = np.unique(labels, return_inverse=True)
    n, k = labels.size, uniq.size
    if k < 2 or k >= n:
        return np.zeros(n)
    D = pairwise_distances(np.asarray(X, dtype=np.float64))
    onehot = np.zeros((n, k))
    onehot[np.arange(n), lab] = 1.0
    sums = D @ onehot                                  # (n, k) sum of distances to each cluster
    counts = onehot.sum(axis=0)
    own = counts[lab]
    a = sums[np.arange(n), lab] / np.maximum(own - 1, 1)
    mean_other = sums / counts[None, :]
    mean_other[np.arange(n), lab] = np.inf
    b = mean_other.min(axis=1)
    den = np.maximum(a, b)
    s = np.where(den > 0, (b - a) / np.where(den > 0, den, 1.0), 0.0)
    return np.where(own > 1, s, 0.0)


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (-1..1; higher = better separated clusters)."""
    return float(np.mean(silhouette_samples(X, labels))) if len(labels) else float("nan")


def choose_k(X: np.ndarray, k_range: range = range(2, 7), attempts: int = 10, seed: int = 0
             ) -> tuple[int, dict]:
    """Number of clusters with the highest silhouette -> (k, {k: {silhouette, compactness}}).

    The compactness (sum of squared distances to the centres) is returned for the elbow
    plot. k values >= the number of samples are skipped.
    """
    scores = {}
    for k in k_range:
        if k >= len(X):
            break
        labels, _, comp = kmeans(X, k, attempts, seed)
        scores[k] = {"silhouette": silhouette_score(X, labels), "compactness": comp}
    if not scores:
        return 1, scores
    best = max(scores, key=lambda kk: scores[kk]["silhouette"])
    return best, scores


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projection on the first 2 principal components (np.linalg.svd).

    Returns (scores (n, 2), explained variance ratio (2,), components (2, n_features)).
    The sign of each component is fixed (largest loading positive) for reproducible plots.
    """
    Xc = np.asarray(X, dtype=np.float64) - np.mean(X, axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:2].copy()
    sign = np.sign(comps[np.arange(comps.shape[0]), np.argmax(np.abs(comps), axis=1)])
    comps *= sign[:, None]
    var = s ** 2
    ratio = var[:2] / var.sum() if var.sum() > 0 else np.zeros(2)
    scores = Xc @ comps.T
    if scores.shape[1] < 2:                            # a single feature
        scores = np.column_stack([scores, np.zeros(len(scores))])
        ratio = np.append(ratio, 0.0)
    return scores, ratio, comps


# ======================================================================================
# Validation metrics
# ======================================================================================
def contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Contingency table n_ij = #(a == i-th label, b == j-th label)."""
    _, ia = np.unique(np.asarray(a), return_inverse=True)
    _, ib = np.unique(np.asarray(b), return_inverse=True)
    table = np.zeros((ia.max() + 1, ib.max() + 1), int)
    np.add.at(table, (ia, ib), 1)
    return table


def cluster_purity(labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Purity vs. known classes: fraction of objects in the majority class of their cluster."""
    if len(labels) == 0:
        return float("nan")
    return float(contingency(labels, true_labels).max(axis=1).sum() / len(labels))


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two partitions (1 = identical, ~0 = random agreement)."""
    n = len(a)
    if n < 2:
        return float("nan")
    t = contingency(a, b)

    def c2(x):
        return x * (x - 1) / 2.0

    sum_ij = c2(t).sum()
    sum_a, sum_b = c2(t.sum(axis=1)).sum(), c2(t.sum(axis=0)).sum()
    expected = sum_a * sum_b / c2(n)
    max_idx = 0.5 * (sum_a + sum_b)
    if max_idx == expected:
        return 1.0
    return float((sum_ij - expected) / (max_idx - expected))


# ======================================================================================
# Interpretation
# ======================================================================================
def cluster_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """(k, n_features) per-cluster median in physical units (NaN-aware)."""
    out = np.full((k, X.shape[1]), np.nan)
    for c in range(k):                                  # loop over clusters
        sub = X[labels == c]
        for j in range(X.shape[1]):
            col = sub[np.isfinite(sub[:, j]), j]
            if col.size:
                out[c, j] = np.median(col)
    return out


def name_clusters(Z: np.ndarray, labels: np.ndarray, groups, k: int, name_z: float = 0.5
                  ) -> list[str]:
    """Readable names of joint k-means clusters from their mean standardised features.

    ``Z`` (unweighted z, columns in the order of ``flatten_groups(groups)``). For every
    property group (group, features, high name, low name) the mean z of its features in
    the cluster is compared with +-``name_z``; the properties that stand out form the name,
    e.g. "thick, constant speed" ("average" if none does).
    """
    names = []
    for c in range(k):
        zc = Z[labels == c].mean(axis=0) if np.any(labels == c) else np.zeros(Z.shape[1])
        parts, j0 = [], 0
        for _, feats, hi, lo in groups:
            z = float(np.mean(zc[j0:j0 + len(feats)]))
            j0 += len(feats)
            if z > name_z and hi:
                parts.append(hi)
            elif z < -name_z and lo:
                parts.append(lo)
        names.append(", ".join(parts) if parts else "average")
    return names


def level_names(k: int, hi: str, lo: str, group: str = "") -> list[str]:
    """Names of k ordered levels of one property (low -> high), e.g. thin / medium
    thickness / thick."""
    if k == 1:
        return [""]
    mid = [f"medium {group}" if k == 3 else f"{group} level {i}" for i in range(1, k - 1)]
    return [lo] + mid + [hi]


def factorized_kmeans(Z: np.ndarray, groups, k_group: int = 0, k_min: int = 2,
                      k_max: int = 3, attempts: int = 10, seed: int = 0, max_iter: int = 100
                      ) -> dict:
    """k-means of every property group separately; clusters = combinations of the levels.

    For each group (group, features, high name, low name) the objects are clustered on
    the group's standardised features only (k = ``k_group``, or chosen by the silhouette in
    [k_min, k_max]); the group clusters are ordered by their mean z (low -> high) and named
    (e.g. thin / thick). The final cluster of an object is the combination of its group
    levels, e.g. "thick, varying speed" - directly the categories of the brief. Returns
    labels (combined, numbered by the level tuples), names, the per-group levels, ks and
    silhouette scores.
    """
    n = Z.shape[0]
    levels, info, j0 = [], {}, 0
    for gname, feats, hi, lo in groups:
        Zg = Z[:, j0:j0 + len(feats)]
        j0 += len(feats)
        if k_group:
            kg, scores = k_group, {}
        else:
            kg, scores = choose_k(Zg, range(k_min, k_max + 1), attempts, seed)
        lab, cen, _ = kmeans(Zg, kg, attempts, seed, max_iter)
        order = np.argsort(cen.mean(axis=1))              # low -> high mean z
        rank = np.empty(kg, int)
        rank[order] = np.arange(kg)
        lv = rank[lab]
        levels.append(lv)
        info[gname] = {"k": kg, "scores": scores, "names": level_names(kg, hi, lo, gname),
                       "silhouette": silhouette_score(Zg, lv),
                       "sizes": np.bincount(lv, minlength=kg).tolist()}
    combo = np.column_stack(levels) if levels else np.zeros((n, 0), int)
    uniq, labels = np.unique(combo, axis=0, return_inverse=True)
    labels = labels.reshape(-1)
    gnames = [g[0] for g in groups]
    names = [", ".join(p for p in (info[g]["names"][lv] for g, lv in zip(gnames, row)) if p)
             or "all" for row in uniq]
    return {"labels": labels, "names": names, "k": len(uniq), "levels": combo,
            "groups": info}


def rule_labels(thickness_score: np.ndarray, varying: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Simple baseline: thick/thin by an Otsu threshold of a thickness score x the
    Stage-9 speed class -> labels 0..3 and their names.

    ``thickness_score`` e.g. log area under the cross-profile; ``varying`` 1 / 0 / NaN.
    The threshold is computed from the data (no hard-coded value).
    """
    s = np.asarray(thickness_score, float)
    fin = np.isfinite(s)
    thr = threshold_otsu(s[fin]) if fin.sum() > 1 and np.ptp(s[fin]) > 0 else np.inf
    thick = fin & (s > thr)
    var = np.nan_to_num(np.asarray(varying, float), nan=0.0) > 0.5
    names = ["thin, constant speed", "thin, varying speed", "thick, constant speed",
             "thick, varying speed"]
    return 2 * thick.astype(int) + var.astype(int), names


def cluster_objects(features: list[dict], params) -> dict:
    """Stage 10 for a list of object feature dicts (``features.object_features``).

    Standardises the features of ``params.feature_groups`` (``config.ClusteringParams``)
    and runs the factorized (default) or joint k-means. Returns {"labels", "names" (per
    cluster), "object_names" (per object), "k", "silhouette"}; with fewer objects than
    2 * k_min no clustering is done (everything in cluster 0, name "").
    """
    n = len(features)
    if n < 2 * params.k_min:
        return {"labels": np.zeros(n, int), "names": [""], "object_names": [""] * n,
                "k": 1, "silhouette": float("nan")}
    groups = params.feature_groups
    X = np.array([[f.get(k, np.nan) for k in flatten_groups(groups)] for f in features],
                 dtype=np.float64)
    Zu, _, _ = standardize(X, params.robust_scale, params.clip_z)
    if params.mode == "factorized":
        fk = factorized_kmeans(Zu, groups, params.k, params.k_min, params.group_k_max,
                               params.attempts, params.seed, params.max_iter)
        labels, names, k = fk["labels"], fk["names"], fk["k"]
        Z = Zu
    elif params.mode == "joint":
        Z = Zu * group_weights(groups, params.group_weights)
        k = params.k or choose_k(Z, range(params.k_min, min(params.k_max, n - 1) + 1),
                                 params.attempts, params.seed)[0]
        labels, _, _ = kmeans(Z, k, params.attempts, params.seed, params.max_iter)
        names = name_clusters(Zu, labels, groups, k, params.name_z)
    else:
        raise ValueError(f"Unknown clustering mode {params.mode!r}")
    return {"labels": labels, "names": names, "object_names": [names[c] for c in labels],
            "k": k, "silhouette": silhouette_score(Z, labels) if k > 1 else float("nan")}
