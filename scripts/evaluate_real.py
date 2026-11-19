"""Stage 9: evaluation on the real segments against the manual labels.

The real data have no ground truth. The reference are the lines marked by hand with
``scripts/annotate_real.py`` (``data/annotations/manual_labels.json``: polylines with a
thick / thin label, drawn on the raw waterfall, *before* looking at the detections).
For every annotated segment the pipeline runs and the grade-4 metrics are computed
(``evaluation.evaluate_objects``):

* **thick recall** - labelled thick lines (trams / heavy trucks) that were detected;
* **precision** - detections that match *some* labelled vehicle (a loud car matched to a
  thin label is a correct detection, not a false positive);
* thin labels found as a bonus, |v| difference to the slope of the label, time offset.

The reference is subjective and approximate (rough clicks on a noisy waterfall), so:

* labels outside the velocity fan of the detector (near-horizontal strokes = horizontal
  noise stripes or imprecise clicks) or shorter than 30 m are listed but not scored
  (``split_references``); the scores with *all* labels are reported as well;
* false positives are split into "near a label" (fragment / duplicate of a labelled
  vehicle, or a pair rejected by the gates) and "no label" (noise or an unmarked vehicle);
* the sensitivity to the matching gates is reported.

The detector is *not* tuned on these labels.

Outputs: ``outputs/results/eval_real.csv`` / ``.json``, ``eval_real_pairs.csv``,
``outputs/figures/evaluation/real_<segment>.png`` and ``real_summary.png``.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das.annotation import load_labels, segment_lines  # noqa: E402
from das.config import PipelineParams  # noqa: E402
from das.evaluation import (evaluate_objects, references_from_annotations,  # noqa: E402
                            split_references, summarize_evaluations, vehicle_class)
from das.pipeline import run_segment, to_jsonable  # noqa: E402
from das.visualization import plot_detections, save_figure  # noqa: E402

FIG_DIR = config.FIG_DIR / "evaluation"
FIELDS = ["segment", "annotator", "n_det", "n_ref", "n_ref_thick", "tp_thick", "fn_thick",
          "recall_thick", "n_ref_thin", "tp_thin", "recall_thin", "tp", "fp", "precision",
          "f1", "fp_near_label", "fp_no_label", "vel_mae_thick", "vel_mae", "vel_mape",
          "mean_dt_s", "n_ref_excluded"]
GATES = [(1.0, 15.0), (1.5, 20.0), (2.0, 25.0), (3.0, 30.0)]
DROP = ("matches", "pairs", "fp_kinds")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", default="all", help="segment name or 'all' (annotated ones)")
    p.add_argument("--labels", default=str(config.ANNOT_FILE))
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def figure_segment(res, seg, refs, excluded, ev, name, dpi) -> None:
    """Raw waterfall with the detections and the manual labels."""
    raw = np.log10(res.pre.stages["raw"] + 1e-12)
    md = {i for i, _ in ev["matches"]}
    mr = {j for _, j in ev["matches"]}
    col = {"near_label": "orange", "no_label": "magenta"}
    colours = ["red" if i in md else col[ev["fp_kinds"][i]] for i in range(len(res.objects))]
    fig, ax = plt.subplots(figsize=(8, 10.5), layout="constrained")
    plot_detections(raw, res.objects, res.pre.row_dt, res.pre.col_dx, seg.start, ax=ax,
                    vmin=float(np.percentile(raw, 1)), vmax=float(np.percentile(raw, 99.5)),
                    cmap="gray", label_fontsize=6.5, colours=colours)
    for j, r in enumerate(refs):
        ax.plot(r["track"][:, 0], r["track"][:, 1], color="lime" if j in mr else "yellow",
                lw=1.0, ls="-" if vehicle_class(r["type"]) == "thick" else ":")
    for r in excluded:
        ax.plot(r["track"][:, 0], r["track"][:, 1], color="cyan", lw=0.8, ls="--")
    ax.set_title(f"{name}: thick labels found {ev['tp_thick']}/{ev['n_ref_thick']}, precision "
                 f"{ev['precision']:.2f} ({ev['tp']}/{ev['n_det']}; thin labels found "
                 f"{ev['tp_thin']}/{ev['n_ref_thin']})\ndetections: red = matched, orange = "
                 f"FP near a label, magenta = FP without a label | labels: lime found, yellow "
                 f"missed (solid thick, dotted thin), cyan not scored", fontsize=8)
    save_figure(fig, FIG_DIR / f"real_{name}.png", dpi)


def figure_summary(rows: list[dict], gates: list[dict], dpi: int) -> None:
    """Metrics per segment, where the errors come from, sensitivity to the matching gates."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    names = [r["segment"] for r in rows]
    x = np.arange(len(names))
    for k, (key, lab) in enumerate((("recall_thick", "recall thick"), ("precision", "precision"),
                                    ("f1", "F1"), ("recall_thin", "thin found (bonus)"))):
        axes[0].bar(x + (k - 1.5) * 0.2, [r[key] for r in rows], width=0.2, label=lab)
    axes[0].set_xticks(x, names, fontsize=8)
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=8, loc="upper center", ncol=4)
    axes[0].set_title("real segments vs manual labels (indicative reference)")
    axes[0].grid(axis="y", alpha=0.3)
    parts = (("tp_thick", "tab:green", "TP thick"), ("tp_thin", "tab:olive", "TP thin (bonus)"),
             ("fp_near_label", "orange", "FP near a label"), ("fp_no_label", "magenta", "FP no label"))
    bottom = np.zeros(len(rows))
    for key, colour, lab in parts:
        v = np.array([r[key] for r in rows], float)
        axes[1].bar(x, v, bottom=bottom, color=colour, label=lab)
        bottom += v
    axes[1].bar(x + 0.3, [r["fn_thick"] for r in rows], width=0.25, color="gold",
                label="missed thick label")
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_ylabel("objects")
    axes[1].legend(fontsize=8)
    axes[1].set_title("detections by outcome (bars) and missed thick labels")
    axes[1].grid(axis="y", alpha=0.3)
    xg = np.arange(len(gates))
    for key, lab in (("recall_thick", "recall thick"), ("precision", "precision"), ("f1", "F1")):
        axes[2].plot(xg, [g[key] for g in gates], "o-", label=lab)
    axes[2].set_xticks(xg, [f"{g['tol_t_s']:.1f} s / {g['tol_v_kmh']:.0f} km/h" for g in gates],
                       rotation=20, fontsize=8)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=8)
    axes[2].set_title("sensitivity to the matching gates (time / velocity)")
    axes[2].grid(alpha=0.3)
    save_figure(fig, FIG_DIR / "real_summary.png", dpi)


def report(ev: dict) -> str:
    return (f"thick {ev['tp_thick']}/{ev['n_ref_thick']} (R {ev['recall_thick']:.2f}) | precision "
            f"{ev['precision']:.2f} ({ev['tp']}/{ev['n_det']}) | F1 {ev['f1']:.2f} | thin found "
            f"{ev['tp_thin']}/{ev['n_ref_thin']} | FP {ev['fp']} ({ev['fp_near_label']} near a "
            f"label) | thick |v| diff {ev['vel_mae_thick']:.2f} km/h")


def main() -> None:
    args = parse_args()
    pp = PipelineParams()
    labels = load_labels(args.labels)
    names = [s["name"] for s in config.SEGMENTS] if args.segment == "all" else [args.segment]
    names = [n for n in names if segment_lines(labels, n)]
    if not names:
        print(f"No manual labels in {args.labels} for {args.segment!r}.\n"
              f"Annotate first: uv run python scripts/annotate_real.py --segment <name>")
        return
    rows, evals, evals_all, pairs, excluded_rows, cache = [], [], [], [], [], []
    for name in names:
        seg = dio.load_named_segment(name, pad_files=1)
        res = run_segment(seg, pp)
        refs_all = references_from_annotations(segment_lines(labels, name))
        refs, excluded = split_references(refs_all, pp.ev.ref_v_range_kmh, pp.ev.ref_min_extent_m)
        ev = evaluate_objects(res.objects, refs, pp.ev, res.n_channels, seg.dx)
        ev_all = evaluate_objects(res.objects, refs_all, pp.ev, res.n_channels, seg.dx)
        cache.append((res.objects, refs, res.n_channels, seg.dx))
        evals.append(ev)
        evals_all.append(ev_all)
        annot = labels["segments"][name].get("annotator", "")
        rows.append({"segment": name, "annotator": annot, "n_ref_excluded": len(excluded), **ev})
        for r in excluded:
            excluded_rows.append({"segment": name, "label_id": r["marker"], "label": r["type"],
                                  "v_kmh": r["velocity_kmh"], "extent_m": r["x_max"] - r["x_min"],
                                  "reason": r["reason"]})
        for p in ev["pairs"]:
            pairs.append({"segment": name, "det_id": res.objects[p["det"]].obj_id,
                          "label_id": refs[p["ref"]]["marker"], "label": p["class"],
                          "v_label_kmh": round(p["v_ref"], 2), "v_det_kmh": round(p["v_det"], 2),
                          "dt_s": round(p["dt_s"], 3)})
        print(f"  {name} ({annot or 'unknown annotator'}): {len(res.objects)} detections / "
              f"{len(refs)} scored labels (+{len(excluded)} not scored)\n"
              f"    scored labels: {report(ev)}\n    all labels:    {report(ev_all)}")
        figure_segment(res, seg, refs, excluded, ev, name, args.dpi)
    total = {"segment": "ALL", "n_ref_excluded": len(excluded_rows), **summarize_evaluations(evals)}
    total_all = {"segment": "ALL (all labels)", "n_ref_excluded": 0,
                 **summarize_evaluations(evals_all)}
    print("  not scored labels:")
    for r in excluded_rows:
        print(f"    {r['segment']} #{r['label_id']} {r['label']}: v {r['v_kmh']:.0f} km/h, "
              f"{r['extent_m']:.0f} m - {r['reason']}")
    for t in (total, total_all):
        print(f"  {t['segment']}: {report(t)} | |v| MAPE {t['vel_mape']:.1f} %, mean |dt| "
              f"{t['mean_dt_s']:.2f} s")
    gates = []
    print("  matching gates (scored labels):")
    for tol_t, tol_v in GATES:
        ep = replace(pp.ev, tol_t_s=tol_t, tol_v_kmh=tol_v)
        g = summarize_evaluations([evaluate_objects(o, r, ep, n, d) for o, r, n, d in cache])
        gates.append({"tol_t_s": tol_t, "tol_v_kmh": tol_v,
                      **{k: g[k] for k in ("recall_thick", "precision", "f1", "recall_thin",
                                           "vel_mae_thick")}})
        print(f"    {tol_t:.1f} s / {tol_v:4.0f} km/h: thick R {g['recall_thick']:.3f} "
              f"P {g['precision']:.3f} F1 {g['f1']:.3f} thin {g['recall_thin']:.3f}")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "eval_real.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows + [total, total_all]])
    with open(config.RESULTS_DIR / "eval_real_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]) if pairs else ["segment"])
        w.writeheader()
        w.writerows(pairs)
    out = {"evaluation_params": asdict(pp.ev), "total": total, "total_all_labels": total_all,
           "not_scored_labels": excluded_rows, "gates": gates,
           "per_segment": [{k: v for k, v in r.items() if k not in DROP} for r in rows]}
    (config.RESULTS_DIR / "eval_real.json").write_text(json.dumps(to_jsonable(out), indent=1),
                                                       encoding="utf-8")
    figure_summary(rows + [total], gates, args.dpi)
    print(f"\nFigures: {FIG_DIR}\nResults: {config.RESULTS_DIR / 'eval_real.csv'}")


if __name__ == "__main__":
    main()
