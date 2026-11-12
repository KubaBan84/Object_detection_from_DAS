"""Stage 11: quantitative evaluation on the synthetic samples (ground truth from the mask).

For every synthetic sample the full pipeline (Stages 4-9) runs with dt / dx from the
metadata (59 channels - nothing assumes 52), the reference objects are built from the
ground-truth mask (``gt_objects_from_mask``) and matched one-to-one with the detections.
Reported per sample and pooled over all samples, **separately for thick (tram / truck)
and thin (car / van) lines**:

- TP / FP / FN, precision, recall, F1 (class precision: detections of the strong path =
  thick, weak path = thin; thick / thin confusion of the matched pairs);
- errors of the reported velocity (Stage-8 track fit): MAE, RMSE, MAPE;
- trajectory IoU: detected vs GT trajectory images (one pixel per channel), dilated by
  +-0.5 s along time;
- sensitivity of the numbers to the matching gates (time / velocity tolerance).

Outputs: ``outputs/results/eval_synthetic.csv`` / ``.json``,
``outputs/figures/evaluation/synthetic_*.png``.

Plan: Stage 11.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das.config import PipelineParams  # noqa: E402
from das.evaluation import (evaluate_objects, gt_objects_from_mask,  # noqa: E402
                            summarize_evaluations, vehicle_class)
from das.pipeline import run_pipeline  # noqa: E402
from das.visualization import plot_detections, save_figure  # noqa: E402

FIG_DIR = config.FIG_DIR / "evaluation"
GATES = [(1.0, 10.0), (1.0, 20.0), (1.5, 10.0), (1.5, 20.0), (1.5, 30.0), (2.0, 20.0),
         (3.0, 30.0)]                                  # (tol_t_s, tol_v_kmh) sensitivity
ROW_FIELDS = ["sample", "n_ref", "n_det", "tp", "fp", "fn", "precision", "recall", "f1",
              "n_ref_heavy", "tp_heavy", "recall_heavy", "n_det_heavy", "precision_heavy",
              "f1_heavy", "n_ref_light", "tp_light", "recall_light", "n_det_light",
              "precision_light", "f1_light", "conf_heavy_as_heavy", "conf_heavy_as_light",
              "conf_light_as_heavy", "conf_light_as_light", "vel_mae", "vel_rmse", "vel_mape",
              "vel_mae_heavy", "vel_mae_light", "traj_iou", "time_s"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", default=None, help="synthetic ids (default: all)")
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


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_table(rows: list[dict], path) -> None:
    """Write the per-sample metric rows to a CSV file."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: fmt(r.get(k, "")) for k in ROW_FIELDS} for r in rows])


def figure_sample(res, refs, ev, sid, path, dpi) -> None:
    """Detections (red = matched, magenta = false) over the enhanced map, GT in lime /
    yellow (missed)."""
    bg = np.maximum.reduce([e.combined for e in res.enhanced.values()])
    matched_d = {i for i, _ in ev["matches"]}
    matched_r = {j for _, j in ev["matches"]}
    fig, ax = plt.subplots(figsize=(7.5, 10), layout="constrained")
    plot_detections(bg, res.objects, res.row_dt, res.col_dx, None, ax=ax, vmin=0, vmax=8,
                    cmap="gray", label=False,
                    colours=["red" if i in matched_d else "magenta"
                             for i in range(len(res.objects))])
    for j, r in enumerate(refs):
        col = "lime" if j in matched_r else "yellow"
        ax.plot(r["track"][:, 0], r["track"][:, 1], color=col, lw=0.8,
                ls="-" if vehicle_class(r["type"]) == "heavy" else ":")
    ax.set_title(f"Synthetic {sid}: F1 {ev['f1']:.2f} (thick {ev['tp_heavy']}/{ev['n_ref_heavy']},"
                 f" thin {ev['tp_light']}/{ev['n_ref_light']}, FP {ev['fp']}), IoU "
                 f"{ev['traj_iou']:.2f}\nred = matched, magenta = false detection; GT lime = "
                 f"found, yellow = missed (solid thick, dotted thin)", fontsize=9)
    save_figure(fig, path, dpi)


def figure_summary(rows: list[dict], total: dict, gates: list[dict], dpi: int) -> None:
    """Summary figure: metrics per sample, sensitivity to the matching gates, velocity error."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    names = [r["sample"] for r in rows] + ["ALL"]
    data = rows + [total]
    x = np.arange(len(names))
    for k, (key, lab) in enumerate((("f1_heavy", "F1 thick"), ("f1_light", "F1 thin"),
                                    ("f1", "F1 all"), ("traj_iou", "trajectory IoU"))):
        axes[0].bar(x + (k - 1.5) * 0.2, [d[key] for d in data], width=0.2, label=lab)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=8, loc="upper center", ncol=4)
    axes[0].set_title("detection quality per synthetic sample")
    axes[0].grid(axis="y", alpha=0.3)
    for k, (key, lab) in enumerate((("vel_mae_heavy", "thick"), ("vel_mae_light", "thin"))):
        axes[1].bar(x + (k - 0.5) * 0.35, [d[key] for d in data], width=0.35, label=lab)
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("|v| MAE [km/h]")
    axes[1].legend(fontsize=8)
    axes[1].set_title("velocity error (reported velocity = track fit)")
    axes[1].grid(axis="y", alpha=0.3)
    lab = [f"{g['tol_t_s']:.1f} s / {g['tol_v_kmh']:.0f} km/h" for g in gates]
    xg = np.arange(len(gates))
    for key, name in (("recall_heavy", "recall thick"), ("recall_light", "recall thin"),
                      ("precision", "precision"), ("vel_mae", "|v| MAE / 10")):
        axes[2].plot(xg, [g[key] / (10 if key == "vel_mae" else 1) for g in gates], "o-",
                     label=name)
    axes[2].set_xticks(xg, lab, rotation=30, fontsize=8)
    axes[2].legend(fontsize=8)
    axes[2].set_title("sensitivity to the matching gates (time / velocity)")
    axes[2].grid(alpha=0.3)
    save_figure(fig, FIG_DIR / "synthetic_summary.png", dpi)


def main() -> None:
    args = parse_args()
    pp = PipelineParams()
    ids = args.ids or dio.list_synthetic_ids()
    rows, evals, cache = [], [], []
    print(f"Synthetic samples {ids}")
    for sid in ids:
        sig, gt, meta = dio.load_synthetic(sid)
        dt, dx = dio.synthetic_axes(meta)
        res = run_pipeline(sig, pp, 1.0 / dt, dx)
        refs = gt_objects_from_mask(gt, meta, dt, dx)
        shape = res.pre.images["hf"].shape
        ev = evaluate_objects(res.objects, refs, pp.ev, shape, res.row_dt, dx)
        cache.append((res.objects, refs, shape, res.row_dt, dx))
        evals.append(ev)
        row = {"sample": sid, "n_ref": len(refs), "n_det": len(res.objects), **ev,
               "time_s": sum(res.timings.values())}
        rows.append(row)
        print(f"  {sid}: {len(res.objects)} det / {len(refs)} GT ({sig.shape[1]} ch) "
              f"P {ev['precision']:.2f} R {ev['recall']:.2f} F1 {ev['f1']:.2f} | thick "
              f"{ev['tp_heavy']}/{ev['n_ref_heavy']} P {ev['precision_heavy']:.2f} | thin "
              f"{ev['tp_light']}/{ev['n_ref_light']} P {ev['precision_light']:.2f} | |v| MAE "
              f"{ev['vel_mae']:.2f} km/h | IoU {ev['traj_iou']:.2f} | {row['time_s']:.1f} s")
        figure_sample(res, refs, ev, sid, FIG_DIR / f"synthetic_{sid}.png", args.dpi)
    total = summarize_evaluations(evals)
    total.update({"sample": "ALL", "n_ref": sum(r["n_ref"] for r in rows),
                  "n_det": sum(r["n_det"] for r in rows), "time_s": sum(r["time_s"] for r in rows)})
    print(f"  ALL: P {total['precision']:.3f} R {total['recall']:.3f} F1 {total['f1']:.3f} "
          f"(TP {total['tp']}, FP {total['fp']}, FN {total['fn']})")
    for c, name in (("heavy", "thick (tram/truck)"), ("light", "thin (car/van)")):
        print(f"    {name:<19} recall {total[f'recall_{c}']:.3f} ({total[f'tp_{c}']}/"
              f"{total[f'n_ref_{c}']})  precision {total[f'precision_{c}']:.3f} "
              f"({total[f'tp_det_{c}']}/{total[f'n_det_{c}']})  F1 {total[f'f1_{c}']:.3f}  "
              f"|v| MAE {total[f'vel_mae_{c}']:.2f} km/h (n {total[f'vel_n_{c}']})")
    print(f"    confusion of matched pairs (GT -> predicted path): "
          f"{ {k: v for k, v in total.items() if k.startswith('conf_')} }")
    print(f"    |v| MAE {total['vel_mae']:.2f} RMSE {total['vel_rmse']:.2f} km/h, MAPE "
          f"{total['vel_mape']:.1f} %, trajectory IoU (+-{pp.ev.iou_tol_s} s) {total['traj_iou']:.3f}")

    print("Sensitivity to the matching gates")
    gates = []
    for tol_t, tol_v in GATES:
        ep = replace(pp.ev, tol_t_s=tol_t, tol_v_kmh=tol_v)
        g = summarize_evaluations([evaluate_objects(o, r, ep, s, rd, d) for o, r, s, rd, d in cache])
        gates.append({"tol_t_s": tol_t, "tol_v_kmh": tol_v,
                      **{k: g[k] for k in ("precision", "recall", "f1", "recall_heavy",
                                           "recall_light", "vel_mae")}})
        print(f"  gate {tol_t:.1f} s / {tol_v:4.0f} km/h: P {g['precision']:.3f} R {g['recall']:.3f}"
              f" (thick {g['recall_heavy']:.3f}, thin {g['recall_light']:.3f}) F1 {g['f1']:.3f} "
              f"|v| MAE {g['vel_mae']:.2f}")
    write_table(rows + [total], config.RESULTS_DIR / "eval_synthetic.csv")
    figure_summary(rows, total, gates, args.dpi)
    out = {"evaluation_params": asdict(pp.ev), "total": total,
           "per_sample": [{k: v for k, v in r.items() if k not in ("matches", "pairs")} for r in rows],
           "gate_sensitivity": gates}
    (config.RESULTS_DIR / "eval_synthetic.json").write_text(json.dumps(to_jsonable(out), indent=1),
                                                            encoding="utf-8")
    print(f"\nFigures: {FIG_DIR}\nResults: {config.RESULTS_DIR / 'eval_synthetic.csv'}")


if __name__ == "__main__":
    main()
