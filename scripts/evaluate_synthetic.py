"""Stage 9: validation on the synthetic samples (ground truth from the mask).

The brief requires the analysis of real data for grade 4; the synthetic samples are
used only as a *validation* with a true ground truth - the only place where the velocity
error can be measured exactly. For every sample the full pipeline runs with dt / dx
from the metadata (59 channels - nothing assumes 52), the reference objects are built
from the ground-truth mask (``gt_objects_from_mask``) and matched one-to-one with the
detections. Reported per sample and pooled (``evaluation.evaluate_objects``):

- recall of the thick vehicles (tram / truck) - the grade-4 target;
- precision (a detection matched to a car / van is a correct vehicle, not an FP);
- cars / vans found as a bonus;
- |v| errors (MAE, RMSE, MAPE) of the matched thick vehicles and of all matched ones;
- sensitivity to the matching gates (time / velocity tolerance).

Outputs: ``outputs/results/eval_synthetic.csv`` / ``.json``,
``outputs/figures/evaluation/synthetic_*.png``.
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
from das.config import PipelineParams  # noqa: E402
from das.evaluation import (evaluate_objects, gt_objects_from_mask,  # noqa: E402
                            summarize_evaluations, vehicle_class)
from das.pipeline import run_pipeline, to_jsonable  # noqa: E402
from das.visualization import plot_detections, save_figure  # noqa: E402

FIG_DIR = config.FIG_DIR / "evaluation"
GATES = [(1.0, 10.0), (1.0, 20.0), (1.5, 10.0), (1.5, 20.0), (2.0, 20.0), (3.0, 30.0)]
FIELDS = ["sample", "n_det", "n_ref", "n_ref_thick", "tp_thick", "fn_thick", "recall_thick",
          "n_ref_thin", "tp_thin", "recall_thin", "tp", "fp", "precision", "f1",
          "vel_mae_thick", "vel_rmse_thick", "vel_mape_thick", "vel_mae", "mean_dt_s"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", default=None, help="synthetic ids (default: all)")
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def figure_sample(res, refs, ev, sid, dpi) -> None:
    """Detections (red = matched, magenta = false) over the response map, GT in lime
    (found) / yellow (missed); solid = thick vehicle, dotted = thin."""
    md = {i for i, _ in ev["matches"]}
    mr = {j for _, j in ev["matches"]}
    fig, ax = plt.subplots(figsize=(7.5, 10), layout="constrained")
    plot_detections(res.enhanced.combined, res.objects, res.row_dt, res.col_dx, None, ax=ax,
                    vmin=0, vmax=8, cmap="gray", label=False,
                    colours=["red" if i in md else "magenta" for i in range(len(res.objects))])
    for j, r in enumerate(refs):
        ax.plot(r["track"][:, 0], r["track"][:, 1], color="lime" if j in mr else "yellow",
                lw=0.9, ls="-" if vehicle_class(r["type"]) == "thick" else ":")
    ax.set_title(f"synthetic {sid}: thick {ev['tp_thick']}/{ev['n_ref_thick']}, precision "
                 f"{ev['precision']:.2f} ({ev['tp']}/{ev['n_det']}), thin found {ev['tp_thin']}/"
                 f"{ev['n_ref_thin']}\nred = matched, magenta = false detection; GT lime = found, "
                 f"yellow = missed (solid thick, dotted thin)", fontsize=9)
    save_figure(fig, FIG_DIR / f"synthetic_{sid}.png", dpi)


def figure_summary(rows: list[dict], gates: list[dict], dpi: int) -> None:
    """Metrics per sample, velocity error, sensitivity to the matching gates."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    names = [r["sample"] for r in rows]
    x = np.arange(len(names))
    for k, (key, lab) in enumerate((("recall_thick", "recall thick"), ("precision", "precision"),
                                    ("f1", "F1"), ("recall_thin", "thin found (bonus)"))):
        axes[0].bar(x + (k - 1.5) * 0.2, [r[key] for r in rows], width=0.2, label=lab)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=8, loc="upper center", ncol=4)
    axes[0].set_title("detection quality per synthetic sample")
    axes[0].grid(axis="y", alpha=0.3)
    for k, (key, lab) in enumerate((("vel_mae_thick", "thick vehicles"), ("vel_mae", "all matched"))):
        axes[1].bar(x + (k - 0.5) * 0.35, [r[key] for r in rows], width=0.35, label=lab)
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("|v| MAE [km/h]")
    axes[1].legend(fontsize=8)
    axes[1].set_title("velocity error of the averaged line")
    axes[1].grid(axis="y", alpha=0.3)
    xg = np.arange(len(gates))
    for key, lab in (("recall_thick", "recall thick"), ("precision", "precision"),
                     ("vel_mae_thick", "thick |v| MAE / 10")):
        axes[2].plot(xg, [g[key] / (10 if "mae" in key else 1) for g in gates], "o-", label=lab)
    axes[2].set_xticks(xg, [f"{g['tol_t_s']:.1f} s / {g['tol_v_kmh']:.0f} km/h" for g in gates],
                       rotation=20, fontsize=8)
    axes[2].legend(fontsize=8)
    axes[2].set_title("sensitivity to the matching gates (time / velocity)")
    axes[2].grid(alpha=0.3)
    save_figure(fig, FIG_DIR / "synthetic_summary.png", dpi)


def report(ev: dict) -> str:
    return (f"thick {ev['tp_thick']}/{ev['n_ref_thick']} | precision {ev['precision']:.2f} "
            f"({ev['tp']}/{ev['n_det']}) | F1 {ev['f1']:.2f} | thin found {ev['tp_thin']}/"
            f"{ev['n_ref_thin']} | thick |v| MAE {ev['vel_mae_thick']:.2f} km/h")


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
        ev = evaluate_objects(res.objects, refs, pp.ev, res.n_channels, dx)
        cache.append((res.objects, refs, res.n_channels, dx))
        evals.append(ev)
        rows.append({"sample": sid, **ev})
        print(f"  {sid} ({sig.shape[1]} ch): {report(ev)}")
        figure_sample(res, refs, ev, sid, args.dpi)
    total = {"sample": "ALL", **summarize_evaluations(evals)}
    print(f"  ALL: {report(total)}\n    thick |v| RMSE {total['vel_rmse_thick']:.2f} km/h, MAPE "
          f"{total['vel_mape_thick']:.1f} %; all matched |v| MAE {total['vel_mae']:.2f} km/h; "
          f"mean |dt| {total['mean_dt_s']:.2f} s")
    gates = []
    print("  sensitivity to the matching gates")
    for tol_t, tol_v in GATES:
        ep = replace(pp.ev, tol_t_s=tol_t, tol_v_kmh=tol_v)
        g = summarize_evaluations([evaluate_objects(o, r, ep, n, d) for o, r, n, d in cache])
        gates.append({"tol_t_s": tol_t, "tol_v_kmh": tol_v,
                      **{k: g[k] for k in ("recall_thick", "precision", "f1", "vel_mae_thick")}})
        print(f"    {tol_t:.1f} s / {tol_v:4.0f} km/h: thick R {g['recall_thick']:.3f} "
              f"P {g['precision']:.3f} F1 {g['f1']:.3f} thick |v| MAE {g['vel_mae_thick']:.2f}")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "eval_synthetic.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows + [total]])
    figure_summary(rows + [total], gates, args.dpi)
    out = {"evaluation_params": asdict(pp.ev), "total": total, "gate_sensitivity": gates,
           "per_sample": [{k: v for k, v in r.items() if k not in ("matches", "pairs", "fp_kinds")}
                          for r in rows]}
    (config.RESULTS_DIR / "eval_synthetic.json").write_text(json.dumps(to_jsonable(out), indent=1),
                                                            encoding="utf-8")
    print(f"\nFigures: {FIG_DIR}\nResults: {config.RESULTS_DIR / 'eval_synthetic.csv'}")


if __name__ == "__main__":
    main()
