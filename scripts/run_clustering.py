"""Collect features of all detected objects, choose k (silhouette), cluster with cv2.kmeans, plot PCA + cluster galleries.

Plan: Stage 10.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=None, help="force number of clusters")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
