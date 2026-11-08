"""Interactive matplotlib tool (ginput) to click start/end points of visible lines in a real segment -> data/annotations/manual_labels.json.

Plan: Stage 11.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
