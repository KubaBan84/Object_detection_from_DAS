"""Interactive tool to mark the visible lines of a real segment by hand (reference for Stage 9).

This is only the labelling tool that produced the reference of the real segments; it is
not part of the detection algorithm. It shows the raw energy waterfall of the segment
(log10 RMS per 0.04 s, as in the brief; "b" switches to the Stage-4 image - never the
detector output) and
records polylines with a thick / thin label in physical units. Labels are saved to
``data/annotations/manual_labels.json``; existing labels of the segment are loaded, so
the work can be continued or corrected later.

Usage::

    uv run python scripts/annotate_real.py --segment seg_kuba --annotator Kuba
    uv run python scripts/annotate_real.py --segment seg_kuba --list    # print labels only

Keys: see ``das.annotation.HELP`` (printed at start). Plan: Stage 9.
"""
from __future__ import annotations

import argparse

import numpy as np

from das import config
from das import io as dio
from das.annotation import HELP, AnnotationSession, Annotator, load_labels, save_labels
from das.config import PreprocessingParams
from das.preprocessing import preprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", required=True, help="segment name from config.SEGMENTS")
    p.add_argument("--annotator", default="", help="who annotates (stored with the labels)")
    p.add_argument("--window", type=float, default=30.0, help="visible time window [s]")
    p.add_argument("--labels", default=str(config.ANNOT_FILE), help="labels JSON file")
    p.add_argument("--list", action="store_true", help="print the stored labels and exit")
    return p.parse_args()


def backgrounds(seg) -> tuple[list[tuple[str, np.ndarray]], float, float]:
    """Raw log-RMS waterfall and the Stage-4 image, both at the image resolution."""
    pre = preprocess(seg.data, PreprocessingParams(), 1.0 / seg.dt, seg.dx,
                     crop=(seg.pad_before, seg.pad_after))
    imgs = [("raw log10 RMS", np.log10(pre.stages["raw"] + 1e-12)),
            ("Stage-4 image", pre.image)]
    return imgs, pre.row_dt, pre.col_dx


def main() -> None:
    args = parse_args()
    data = load_labels(args.labels)
    stored = data["segments"].get(args.segment, {})
    if args.list:
        for ln in stored.get("lines", []):
            pts = np.asarray(ln["points"])
            print(f"  #{ln['id']:<3} {ln['kind']:<5} x {pts[:, 0].min():6.1f}-{pts[:, 0].max():6.1f} m"
                  f"  t {pts[:, 1].min():6.1f}-{pts[:, 1].max():6.1f} s  ({len(pts)} points)")
        print(f"{args.segment}: {len(stored.get('lines', []))} lines")
        return
    import matplotlib.pyplot as plt

    spec = dio.get_segment_spec(args.segment)
    seg = dio.load_named_segment(args.segment, pad_files=1)
    imgs, row_dt, dx = backgrounds(seg)
    session = AnnotationSession(stored.get("lines", []))
    annotator = args.annotator or stored.get("annotator", "")

    def on_save(s: AnnotationSession):
        data["segments"][args.segment] = s.to_dict(spec["start"], annotator,
                                                   imgs[0][0])
        return save_labels(data, args.labels)

    print(HELP)
    print(f"{args.segment}: {len(session.lines)} stored lines loaded")
    title = (f"{args.segment} {seg.start:%H:%M:%S}-{seg.end:%H:%M:%S}: mark every visible "
             f"line (t = thick, n = thin)")
    tool = Annotator(imgs, row_dt, dx, seg.start, session, on_save, title, args.window)
    plt.show()
    del tool
    print(f"{args.segment}: {len(session.lines)} lines saved to {args.labels}")


if __name__ == "__main__":
    main()
