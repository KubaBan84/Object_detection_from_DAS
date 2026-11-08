"""Clustering of detected objects (plan: Etap 10, extra 2). cv2.kmeans + numpy metrics."""
from __future__ import annotations

import numpy as np


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError


def kmeans(X: np.ndarray, k: int, attempts: int = 10, seed: int = 0
           ) -> tuple[np.ndarray, np.ndarray, float]:
    """cv2.kmeans wrapper -> (labels, centers, compactness)."""
    raise NotImplementedError


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Vectorised silhouette coefficient (pairwise distance matrix)."""
    raise NotImplementedError


def choose_k(X: np.ndarray, k_range: range = range(2, 7)) -> tuple[int, dict]:
    raise NotImplementedError


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Projection on the first 2 principal components (np.linalg.svd)."""
    raise NotImplementedError


def cluster_purity(labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Purity vs. synthetic vehicle types (validation of clustering)."""
    raise NotImplementedError
