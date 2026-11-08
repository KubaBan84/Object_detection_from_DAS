"""Exploratory data analysis: global streaming statistics over all 259 files + per-segment statistics and figures.

Plan: Etap 2.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment", default="all", help="segment name from config.SEGMENTS or 'all'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
