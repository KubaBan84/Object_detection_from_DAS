"""Regenerate every figure used in report/ (scaled down) into report/figures/.

Plan: Stage 14.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dpi", type=int, default=110)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
