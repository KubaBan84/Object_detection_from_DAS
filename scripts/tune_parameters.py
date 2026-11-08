"""Grid search / ablation of key parameters on synthetic + annotated real segments; writes outputs/results/tuning.csv.

Plan: Stage 13.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
