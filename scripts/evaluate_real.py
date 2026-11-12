"""Stage 11: evaluation on the real segments against the manual labels.

The real data have no ground truth. The reference are the lines marked by hand with
``scripts/annotate_real.py`` (``data/annotations/manual_labels.json``: polylines with a
thick / thin label, drawn on the raw waterfall). For every annotated segment the pipeline
runs and the same metrics as on synthetic data are computed (``evaluate_objects``):
TP / FP / FN, precision / recall / F1 for thick and thin lines, velocity agreement with
the slope of the annotated line, and the trajectory IoU. This is a *subjective*,
approximate reference (rough clicks on a noisy waterfall), therefore:

* labels outside the velocity fan of the detector (near-horizontal strokes = horizontal
  noise stripes or imprecise clicks) or shorter than 30 m are listed but not scored
  (``split_references``); the scores with *all* labels are reported as well;
* false positives are split into "near a label" (fragment / duplicate of a labelled
  vehicle, or a pair rejected by the gates) and "no label" (noise or an unmarked vehicle);
* the sensitivity to the matching gates is reported.

The detector is *not* tuned on these labels.

Outputs: ``outputs/results/eval_real.csv`` / ``.json``, ``eval_real_pairs.csv``,
``outputs/figures/evaluation/real_<segment>.png`` (detections vs labels) and
``real_summary.png``.

Plan: Stage 11.
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
from das.pipeline import run_segment  # noqa: E402
from das.velocity import reported_velocity_kmh  # noqa: E402
from das.visualization import plot_detections, save_figure  # noqa: E402

FIG_DIR = config.FIG_DIR / "evaluation"
FIELDS = ["segment", "annotator", "n_ref", "n_det", "tp", "fp", "fn", "precision", "recall",
          "f1", "n_ref_heavy", "tp_heavy", "recall_heavy", "n_det_heavy", "precision_heavy",
          "f1_heavy", "n_ref_light", "tp_light", "recall_light", "n_det_light",
          "precision_light", "f1_light", "conf_heavy_as_heavy", "conf_heavy_as_light",
          "conf_light_as_heavy", "conf_light_as_light", "vel_mae", "vel_rmse", "vel_mape",
          "vel_mae_heavy", "vel_mae_light", "traj_iou", "fp_near_label", "fp_no_label",
          "n_ref_excluded"]
GATES = [(1.0, 15.0), (1.5, 20.0), (2.0, 25.0), (3.0, 30.0), (4.0, 40.0)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", default="all", help="segment name or 'all' (annotated ones)")
    p.add_argument("--labels", default=str(config.ANNOT_FILE))
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def figure_segment(res, seg, refs, excluded, ev, name, path, dpi) -> None:
    """Raw waterfall with detections and the manual labels (legend in the title)."""
    raw = np.log10(res.pre.common["raw"] + 1e-12)
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
                lw=1.0, ls="-" if vehicle_class(r["type"]) == "heavy" else ":")
    for r in excluded:
        ax.plot(r["track"][:, 0], r["track"][:, 1], color="cyan", lw=0.8, ls="--")
    ax.set_title(f"{name}: detections vs manual labels - F1 {ev['f1']:.2f} (thick "
                 f"{ev['tp_heavy']}/{ev['n_ref_heavy']}, thin {ev['tp_light']}/{ev['n_ref_light']},"
                 f" FP {ev['fp']} = {ev['fp_near_label']} near a label + {ev['fp_no_label']} "
                 f"unlabelled)\ndetections: red matched, orange near a label, magenta no label | "
                 f"labels: lime found, yellow missed (solid thick, dotted thin), cyan not scored",
                 fontsize=8)
    save_figure(fig, path, dpi)


def figure_summary(rows: list[dict], total: dict, total_all: dict, gates: list[dict],
                   dpi: int) -> None:
    """Summary figure: metrics per segment, sensitivity to the matching gates, error sources."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    names = [r["segment"] for r in rows] + ["ALL", "ALL (all labels)"]
    data = rows + [total, total_all]
    x = np.arange(len(names))
    for k, (key, lab) in enumerate((("recall_heavy", "recall thick"),
                                    ("recall_light", "recall thin"),
                                    ("precision", "precision"), ("f1", "F1"))):
        axes[0].bar(x + (k - 1.5) * 0.2, [d[key] for d in data], width=0.2, label=lab)
    axes[0].set_xticks(x, names, fontsize=8)
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=8, loc="upper center", ncol=4)
    axes[0].set_title("real segments vs manual labels (indicative reference)")
    axes[0].grid(axis="y", alpha=0.3)
    tp = np.array([d["tp"] for d in data])
    near = np.array([d["fp_near_label"] for d in data])
    nol = np.array([d["fp_no_label"] for d in data])
    fn = np.array([d["fn"] for d in data])
    axes[1].bar(x, tp, color="tab:green", label="TP")
    axes[1].bar(x, near, bottom=tp, color="orange", label="FP near a label")
    axes[1].bar(x, nol, bottom=tp + near, color="magenta", label="FP no label")
    axes[1].bar(x, fn, bottom=tp + near + nol, color="gold", label="FN (missed label)")
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_ylabel("objects")
    axes[1].legend(fontsize=8)
    axes[1].set_title("where the errors come from")
    axes[1].grid(axis="y", alpha=0.3)
    lab = [f"{g['tol_t_s']:.1f} s / {g['tol_v_kmh']:.0f} km/h" for g in gates]
    xg = np.arange(len(gates))
    for key, name in (("recall_heavy", "recall thick"), ("recall_light", "recall thin"),
                      ("precision", "precision"), ("f1", "F1")):
        axes[2].plot(xg, [g[key] for g in gates], "o-", label=name)
    axes[2].set_xticks(xg, lab, rotation=30, fontsize=8)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=8)
    axes[2].set_title("sensitivity to the matching gates (time / velocity)")
    axes[2].grid(alpha=0.3)
    save_figure(fig, FIG_DIR / "real_summary.png", dpi)


def report(ev: dict) -> str:
    return (f"P {ev['precision']:.2f} R {ev['recall']:.2f} F1 {ev['f1']:.2f} | thick "
            f"{ev['tp_heavy']}/{ev['n_ref_heavy']} | thin {ev['tp_light']}/{ev['n_ref_light']} | "
            f"FP {ev['fp']} ({ev['fp_near_label']} near a label, {ev['fp_no_label']} no label) | "
            f"|v| diff {ev['vel_mae']:.2f} km/h | IoU {ev['traj_iou']:.2f}")


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
        refs, excluded = split_references(refs_all, pp.ev.ref_v_range_kmh,
                                          pp.ev.ref_min_extent_m)
        shape = res.pre.images["hf"].shape
        ev = evaluate_objects(res.objects, refs, pp.ev, shape, res.row_dt, seg.dx)
        ev_all = evaluate_objects(res.objects, refs_all, pp.ev, shape, res.row_dt, seg.dx)
        cache.append((res.objects, refs, shape, res.row_dt, seg.dx))
        evals.append(ev)
        evals_all.append(ev_all)
        annot = labels["segments"][name].get("annotator", "")
        rows.append({"segment": name, "annotator": annot, "n_ref": len(refs),
                     "n_det": len(res.objects), "n_ref_excluded": len(excluded), **ev})
        for r in excluded:
            excluded_rows.append({"segment": name, "label_id": r["marker"], "label": r["type"],
                                  "v_kmh": r["velocity_kmh"],
                                  "extent_m": r["x_max"] - r["x_min"], "reason": r["reason"]})
        for i, j in ev["matches"]:
            o, r = res.objects[i], refs[j]
            pairs.append({"segment": name, "det_id": o.obj_id, "label_id": r["marker"],
                          "label": r["type"], "kind": o.kind,
                          "v_label_kmh": round(r["velocity_kmh"], 2),
                          "v_det_kmh": round(reported_velocity_kmh(o), 2)})
        print(f"  {name} ({annot or 'unknown annotator'}): {len(res.objects)} det / {len(refs)} "
              f"scored labels (+{len(excluded)} not scored)\n"
              f"    scored labels: {report(ev)}\n    all labels:    {report(ev_all)}")
        figure_segment(res, seg, refs, excluded, ev, name, FIG_DIR / f"real_{name}.png",
                       args.dpi)
    total = summarize_evaluations(evals)
    total.update({"segment": "ALL", "n_ref": sum(r["n_ref"] for r in rows),
                  "n_det": sum(r["n_det"] for r in rows), "n_ref_excluded": len(excluded_rows)})
    total_all = summarize_evaluations(evals_all)
    total_all.update({"segment": "ALL (all labels)",
                      "n_ref": total["n_ref"] + len(excluded_rows),
                      "n_det": total["n_det"], "n_ref_excluded": 0})
    print("  not scored labels:")
    for r in excluded_rows:
        print(f"    {r['segment']} #{r['label_id']} {r['label']}: v {r['v_kmh']:.0f} km/h, "
              f"{r['extent_m']:.0f} m - {r['reason']}")
    for t, nm in ((total, "ALL"), (total_all, "ALL (all labels)")):
        print(f"  {nm}: P {t['precision']:.3f} R {t['recall']:.3f} F1 {t['f1']:.3f}; "
              f"thick R {t['recall_heavy']:.3f} P {t['precision_heavy']:.3f}; thin R "
              f"{t['recall_light']:.3f} P {t['precision_light']:.3f}; FP {t['fp']} "
              f"({t['fp_near_label']} near a label); |v| diff {t['vel_mae']:.2f} km/h "
              f"(MAPE {t['vel_mape']:.1f} %); IoU {t['traj_iou']:.3f}")
    gates = []
    print("  matching gates (scored labels):")
    for tol_t, tol_v in GATES:
        ep = replace(pp.ev, tol_t_s=tol_t, tol_v_kmh=tol_v)
        g = summarize_evaluations([evaluate_objects(o, r, ep, s, rd, d)
                                   for o, r, s, rd, d in cache])
        gates.append({"tol_t_s": tol_t, "tol_v_kmh": tol_v,
                      **{k: g[k] for k in ("precision", "recall", "f1", "recall_heavy",
                                           "recall_light", "vel_mae", "fp_near_label")}})
        print(f"    {tol_t:.1f} s / {tol_v:4.0f} km/h: P {g['precision']:.3f} "
              f"R {g['recall']:.3f} F1 {g['f1']:.3f} thick R {g['recall_heavy']:.3f} "
              f"thin R {g['recall_light']:.3f} |v| {g['vel_mae']:.2f}")
    with open(config.RESULTS_DIR / "eval_real.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows + [total, total_all]])
    with open(config.RESULTS_DIR / "eval_real_pairs.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]) if pairs else ["segment"])
        w.writeheader()
        w.writerows(pairs)
    drop = ("matches", "pairs", "fp_kinds")
    out = {"evaluation_params": asdict(pp.ev), "total": total, "total_all_labels": total_all,
           "not_scored_labels": excluded_rows, "gates": gates,
           "per_segment": [{k: v for k, v in r.items() if k not in drop} for r in rows],
           "per_segment_all_labels": [{"segment": n,
                                       **{k: v for k, v in e.items() if k not in drop}}
                                      for n, e in zip(names, evals_all)]}
    (config.RESULTS_DIR / "eval_real.json").write_text(json.dumps(to_jsonable(out), indent=1),
                                                       encoding="utf-8")
    figure_summary(rows, total, total_all, gates, args.dpi)
    print(f"\nFigures: {FIG_DIR}\nResults: {config.RESULTS_DIR / 'eval_real.csv'}")


if __name__ == "__main__":
    main()
