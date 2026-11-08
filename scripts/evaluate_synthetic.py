"""Evaluate detections against synthetic ground truth (precision/recall/F1 per vehicle type, velocity errors).

Plan: Stage 11.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
