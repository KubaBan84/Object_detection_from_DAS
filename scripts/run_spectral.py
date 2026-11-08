"""Frequency analysis: per-channel FFT, Welch PSD, STFT of chosen channels, f-k spectrum, band energies.

Plan: Stage 3.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment", default="all")
    p.add_argument("--channels", type=int, nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
