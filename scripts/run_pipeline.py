"""Run the full detection pipeline on configured 2-minute segments (or a synthetic sample) and save figures + CSV/JSON.

Plan: Stage 12.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment", default="all", help="segment name or 'all'")
    p.add_argument("--start", default=None, help="HHMMSS of an ad-hoc segment")
    p.add_argument("--n-files", type=int, default=12)
    p.add_argument("--synthetic", default=None, help="synthetic sample id")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
