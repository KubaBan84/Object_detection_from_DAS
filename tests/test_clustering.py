"""Tests of das.clustering (Stage 10): standardisation, cv2.kmeans wrapper, silhouette,
choice of k, PCA, validation metrics, factorized clustering and naming."""
from __future__ import annotations

import numpy as np
import pytest

from das.clustering import (adjusted_rand_index, choose_k, cluster_centroids, cluster_purity,
                            factorized_kmeans, flatten_groups, group_weights, kmeans,
                            level_names, name_clusters, pca_2d, rule_labels,
                            silhouette_samples, silhouette_score, standardize)

GROUPS = (("thickness", ("a", "b"), "thick", "thin"),
          ("speed change", ("c",), "varying speed", "constant speed"))


def blobs(n=(30, 20, 10), seed=0, spread=0.3):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    X = np.concatenate([c + spread * rng.normal(size=(k, 2)) for c, k in zip(centres, n)])
    y = np.repeat(np.arange(len(n)), n)
    return X, y


def silhouette_loop(X, labels):
    """Reference implementation with explicit loops (tests only)."""
    s = []
    for i in range(len(X)):
        d = np.linalg.norm(X - X[i], axis=1)
        own = labels == labels[i]
        if own.sum() == 1:
            s.append(0.0)
            continue
        a = d[own].sum() / (own.sum() - 1)
        b = min(d[labels == c].mean() for c in np.unique(labels) if c != labels[i])
        s.append((b - a) / max(a, b))
    return np.array(s)


# ---------------------------------------------------------------- standardisation
def test_standardize_nan_constant_and_clip():
    X = np.array([[1.0, 5.0, 0.0], [2.0, 5.0, np.nan], [3.0, 5.0, 100.0], [4.0, 5.0, 1.0]])
    Z, c, s = standardize(X, robust=False)
    assert np.allclose(Z[:, 0].mean(), 0) and np.allclose(Z[:, 0].std(), 1)
    assert np.all(Z[:, 1] == 0) and s[1] == 1.0                 # constant feature
    assert Z[1, 2] == 0.0                                       # NaN -> centre
    Zr, cr, _ = standardize(X, robust=True, clip=2.0)
    assert cr[0] == np.median(X[:, 0]) and np.abs(Zr).max() <= 2.0
    assert standardize(np.zeros((0, 3)))[0].shape == (0, 3)


def test_group_weights_and_flatten():
    assert flatten_groups(GROUPS) == ["a", "b", "c"]
    assert np.allclose(group_weights(GROUPS), [1 / np.sqrt(2), 1 / np.sqrt(2), 1.0])
    assert np.allclose(group_weights(GROUPS, equal=False), 1.0)


# ---------------------------------------------------------------- k-means, silhouette, k
def test_kmeans_recovers_blobs_sorted_by_size():
    X, y = blobs()
    labels, centers, comp = kmeans(X, 3, seed=1)
    assert adjusted_rand_index(labels, y) == pytest.approx(1.0)
    assert np.bincount(labels).tolist() == [30, 20, 10]        # 0 = largest cluster
    assert centers.shape == (3, 2) and comp > 0
    assert np.array_equal(labels, kmeans(X, 3, seed=1)[0])     # reproducible


def test_silhouette_matches_loop_and_edge_cases():
    X, y = blobs((8, 6, 5), seed=3, spread=1.5)
    assert np.allclose(silhouette_samples(X, y), silhouette_loop(X, y))
    Xs, ys = blobs(seed=2)
    assert silhouette_score(Xs, ys) > 0.85                      # well separated
    rng = np.random.default_rng(0)
    assert abs(silhouette_score(Xs, rng.integers(0, 3, len(ys)))) < 0.15   # random labels
    lab = np.zeros(len(X), int)
    lab[0] = 1                                                  # singleton cluster
    assert silhouette_samples(X, lab)[0] == 0.0
    assert np.all(silhouette_samples(X, np.zeros(len(X), int)) == 0)       # one cluster


def test_choose_k_finds_three_blobs():
    X, _ = blobs(seed=4)
    k, scores = choose_k(X, range(2, 6))
    assert k == 3 and set(scores) == {2, 3, 4, 5}
    assert scores[3]["silhouette"] > scores[2]["silhouette"]
    assert scores[2]["compactness"] > scores[3]["compactness"]   # elbow curve decreases


def test_pca_2d_direction_and_variance():
    rng = np.random.default_rng(5)
    t = rng.normal(size=200)
    X = np.column_stack([t, 2 * t, 0.05 * rng.normal(size=200)])
    proj, ratio, comps = pca_2d(X)
    assert proj.shape == (200, 2) and ratio[0] > 0.99 and ratio.sum() <= 1.0 + 1e-12
    assert np.allclose(np.abs(comps[0]), [1, 2, 0] / np.sqrt(5), atol=0.02)
    assert comps[0, np.argmax(np.abs(comps[0]))] > 0            # fixed sign
    assert pca_2d(t[:, None])[0].shape == (200, 2)              # a single feature


# ---------------------------------------------------------------- validation metrics
def test_purity_and_adjusted_rand_index():
    a = np.array([0, 0, 0, 1, 1, 1])
    assert adjusted_rand_index(a, [5, 5, 5, 7, 7, 7]) == pytest.approx(1.0)   # relabelled
    assert adjusted_rand_index(a, [0, 0, 1, 1, 2, 2]) == pytest.approx(0.24242424, abs=1e-6)
    assert cluster_purity(np.array([0, 0, 0, 1, 1, 1]), ["car", "car", "tram", "tram", "tram", "tram"]) \
        == pytest.approx(5 / 6)
    rng = np.random.default_rng(1)
    r = [adjusted_rand_index(rng.integers(0, 3, 300), rng.integers(0, 3, 300)) for _ in range(5)]
    assert abs(np.mean(r)) < 0.02                               # chance level ~0


def test_centroids_and_rule_baseline():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [10.0, 4.0]])
    cen = cluster_centroids(X, np.array([0, 0, 1]), 2)
    assert np.allclose(cen, [[2.0, 2.0], [10.0, 4.0]])
    rng = np.random.default_rng(0)
    score = np.concatenate([rng.normal(0.2, 0.05, 20), rng.normal(2.0, 0.05, 10), [np.nan]])
    varying = np.concatenate([np.tile([0, 1], 15), [np.nan]])
    lab, names = rule_labels(score, varying)
    thick = np.r_[np.zeros(20), np.ones(10), 0]
    assert np.array_equal(lab, 2 * thick + np.nan_to_num(varying))   # NaN -> thin, constant
    assert names[3] == "thick, varying speed"


# ---------------------------------------------------------------- factorized clustering, names
def two_property_data(seed=0):
    """Objects with 2 thickness levels x 2 speed-change levels (z units)."""
    rng = np.random.default_rng(seed)
    thick = np.repeat([0, 1, 0, 1], [20, 15, 10, 12])
    vary = np.repeat([0, 0, 1, 1], [20, 15, 10, 12])
    Z = np.column_stack([3 * thick, 3 * thick, 3 * vary]) + 0.3 * rng.normal(size=(57, 3))
    return Z - Z.mean(axis=0), thick, vary


def test_factorized_kmeans_names_the_four_combinations():
    Z, thick, vary = two_property_data()
    fk = factorized_kmeans(Z, GROUPS, k_group=0, k_max=3)
    assert fk["k"] == 4 and fk["groups"]["thickness"]["k"] == 2
    truth = [f"{'thick' if t else 'thin'}, {'varying speed' if v else 'constant speed'}"
             for t, v in zip(thick, vary)]
    assert [fk["names"][c] for c in fk["labels"]] == truth
    assert fk["levels"].shape == (57, 2)


def test_joint_naming_and_level_names():
    Z, thick, vary = two_property_data(1)
    labels = 2 * thick + vary
    names = name_clusters(Z, labels, GROUPS, 4)
    assert names[3] == "thick, varying speed" and names[0] == "thin, constant speed"
    assert level_names(2, "thick", "thin") == ["thin", "thick"]
    assert level_names(3, "thick", "thin", "thickness") == ["thin", "medium thickness", "thick"]


def test_cluster_objects_from_feature_dicts():
    from das.clustering import cluster_objects
    from das.config import ClusteringParams

    Z, thick, vary = two_property_data(2)
    groups = (("thickness", ("a", "b"), "thick", "thin"),
              ("speed change", ("c",), "varying speed", "constant speed"))
    feats = [{"a": z[0], "b": z[1], "c": z[2]} for z in Z]
    cl = cluster_objects(feats, ClusteringParams(feature_groups=groups))
    truth = [f"{'thick' if t else 'thin'}, {'varying speed' if v else 'constant speed'}"
             for t, v in zip(thick, vary)]
    assert cl["k"] == 4 and cl["object_names"] == truth and cl["silhouette"] > 0.5
    joint = cluster_objects(feats, ClusteringParams(feature_groups=groups, mode="joint", k=4))
    assert adjusted_rand_index(joint["labels"], 2 * thick + vary) == pytest.approx(1.0)
    few = cluster_objects(feats[:3], ClusteringParams(feature_groups=groups))
    assert few["k"] == 1 and few["object_names"] == ["", "", ""]
