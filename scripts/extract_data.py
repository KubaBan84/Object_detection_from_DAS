"""Extract the raw archives and verify the datasets.

- ``data/20240507.rar``        -> ``data/real/``      (259 x ``HHMMSS.npy``, shape (6250, 52))
- ``data/Synthetic Dataset.zip`` -> ``data/synthetic/`` (6 x signal / ground truth / metadata)

The RAR archive is extracted with an external tool: the Windows built-in ``tar.exe``
(libarchive / bsdtar, reads RAR), ``bsdtar``, ``unrar`` or ``7z`` - whichever is found first.

Plan: Stage 0.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from das import config
from das import io as dio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="re-extract even if files exist")
    p.add_argument("--verify-only", action="store_true", help="skip extraction, only verify")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------
def rar_extract_command(archive: Path, out_dir: Path) -> list[str]:
    """Command line of the first available tool able to extract RAR archives."""
    if os.name == "nt":
        win_tar = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"
        if win_tar.is_file():  # bsdtar; GNU tar from Git Bash cannot read RAR
            return [str(win_tar), "-xf", str(archive), "-C", str(out_dir)]
    if shutil.which("bsdtar"):
        return ["bsdtar", "-xf", str(archive), "-C", str(out_dir)]
    if shutil.which("unrar"):
        return ["unrar", "x", "-o+", str(archive), str(out_dir) + os.sep]
    if shutil.which("7z"):
        return ["7z", "x", "-y", f"-o{out_dir}", str(archive)]
    raise RuntimeError("No RAR-capable tool found (tar.exe on Windows, bsdtar, unrar or 7z).")


def extract_real(force: bool) -> None:
    out_dir = config.REAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = dio.list_real_files(out_dir)
    if len(existing) >= config.EXPECTED_REAL_FILES and not force:
        print(f"[real] {len(existing)} files already in {out_dir} - skipping (use --force)")
        return
    if not config.REAL_ARCHIVE.is_file():
        raise FileNotFoundError(f"Archive not found: {config.REAL_ARCHIVE}")
    cmd = rar_extract_command(config.REAL_ARCHIVE, out_dir)
    print(f"[real] extracting {config.REAL_ARCHIVE.name} -> {out_dir} with {Path(cmd[0]).name} ...")
    subprocess.run(cmd, check=True)


def extract_synthetic(force: bool) -> None:
    out_dir = config.SYNTH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = dio.list_synthetic_ids(out_dir)
    if len(existing) >= config.EXPECTED_SYNTH_SAMPLES and not force:
        print(f"[synthetic] {len(existing)} samples already in {out_dir} - skipping (use --force)")
        return
    if not config.SYNTH_ARCHIVE.is_file():
        raise FileNotFoundError(f"Archive not found: {config.SYNTH_ARCHIVE}")
    print(f"[synthetic] extracting {config.SYNTH_ARCHIVE.name} -> {out_dir} ...")
    with zipfile.ZipFile(config.SYNTH_ARCHIVE) as zf:
        for info in zf.infolist():
            # Flatten: keep only the base name (also prevents path traversal).
            name = Path(info.filename).name
            if info.is_dir() or not name or name.startswith("._"):
                continue
            with zf.open(info) as src, open(out_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------
def verify_real() -> list[str]:
    errors: list[str] = []
    files = dio.list_real_files()
    if len(files) != config.EXPECTED_REAL_FILES:
        errors.append(f"[real] expected {config.EXPECTED_REAL_FILES} files, found {len(files)}")

    channels: set[int] = set()
    n_nonfinite = 0
    for f in files:
        try:
            arr = dio.load_real_file(f)
        except ValueError as exc:
            errors.append(f"[real] {exc}")
            continue
        channels.add(arr.shape[1])
        n_nonfinite += int(arr.size - np.count_nonzero(np.isfinite(arr)))
    if len(channels) != 1:
        errors.append(f"[real] inconsistent channel counts: {sorted(channels)}")
    if n_nonfinite:
        errors.append(f"[real] {n_nonfinite} non-finite values (NaN/inf)")
    print(f"[real] {len(files)} files, shape ({config.SAMPLES_PER_FILE}, {sorted(channels)}), "
          f"non-finite values: {n_nonfinite}")

    runs = dio.find_continuous_runs(files)
    print(f"[real] continuous runs: {len(runs)}")
    for run in runs:
        print(f"         {run[0].stem} - {run[-1].stem}: {len(run):3d} files "
              f"({len(run) * config.FILE_DURATION_S} s)")

    for spec in config.SEGMENTS:
        try:
            _, core, _ = dio.resolve_segment_files(spec["start"], spec["n_files"])
            print(f"[segments] {spec['name']:<12} {core[0].stem} - {core[-1].stem} "
                  f"({len(core)} files, {spec['kind']}, {spec.get('owner', '-')}) OK")
        except (ValueError, FileNotFoundError) as exc:
            errors.append(f"[segments] {spec['name']}: {exc}")
    return errors


def verify_synthetic() -> list[str]:
    errors: list[str] = []
    ids = dio.list_synthetic_ids()
    if len(ids) != config.EXPECTED_SYNTH_SAMPLES:
        errors.append(f"[synthetic] expected {config.EXPECTED_SYNTH_SAMPLES} samples, "
                      f"found {len(ids)}")
    for sid in ids:
        try:
            signal, gt, meta = dio.load_synthetic(sid)
        except (ValueError, KeyError, OSError) as exc:
            errors.append(f"[synthetic] {sid}: {exc}")
            continue
        dt, dx = dio.synthetic_axes(meta)
        vehicles = meta["ground_truth"]["vehicles"]
        markers = {int(v["marker_value"]) for v in vehicles}
        present = set(np.unique(gt).tolist()) - {int(meta["ground_truth"]["background_value"])}
        if signal.shape != gt.shape:
            errors.append(f"[synthetic] {sid}: signal {signal.shape} != ground truth {gt.shape}")
        if not present <= markers:
            errors.append(f"[synthetic] {sid}: markers {sorted(present - markers)} not in metadata")
        if not np.isfinite(signal).all():
            errors.append(f"[synthetic] {sid}: non-finite values in signal")
        types: dict[str, int] = {}
        for v in vehicles:
            types[v["type"]] = types.get(v["type"], 0) + 1
        print(f"[synthetic] {sid}: signal {signal.shape}, dt={dt}, dx={dx:.4f}, "
              f"vehicles={len(vehicles)} {types}, markers present={len(present)}")
    return errors


def main() -> None:
    args = parse_args()
    if not args.verify_only:
        extract_real(args.force)
        extract_synthetic(args.force)
    errors = verify_real() + verify_synthetic()
    if errors:
        print("\nVERIFICATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("\nOK - data extracted and verified.")


if __name__ == "__main__":
    main()
