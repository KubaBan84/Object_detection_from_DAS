"""Extract data/20240507.rar -> data/real and Synthetic Dataset.zip -> data/synthetic; verify file counts and shapes.

Plan: Stage 0.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-extract even if files exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
