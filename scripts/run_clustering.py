"""Stage 10 (extra point 2): features and clustering of the detected objects.

Runs the pipeline (Stages 4-9) on the 3 real segments and the synthetic samples,
describes every object (``das.features``: line energy / thickness, cross-profile shape,
raw spectrum, speed change) and clusters the objects with ``cv2.kmeans``:

- default ("factorized"): one k-means per property of the brief - line thickness
  (thick / thin) and speed change (constant / varying) - with k chosen by the silhouette;
  the clusters are the combinations, e.g. "thick, varying speed";
- comparison ("joint"): one k-means on all features, k by the silhouette.

1. feature separability: AUC of every descriptor for heavy vs light and varying vs
   constant speed (synthetic ground truth) - the basis of the feature choice;
2. real segments (all 3 together): PCA scatter, choice of k, features per cluster,
   gallery of straightened lines ("shape of the lines"), waterfalls and v(t) coloured by
   cluster, object and centroid tables, agreement with the manual thick / thin labels
   of Stage 11 (if present);
3. synthetic samples: the same clustering validated against the vehicle types (purity,
   adjusted Rand index) and the GT speed class, vs a rule baseline (Otsu threshold of
   the line area x Stage-9 speed class);
4. method / feature-set study (``config.CLUSTER_STUDY``).

Outputs: ``outputs/figures/clustering/``, ``outputs/results/clustering_*.csv`` and
``outputs/results/clustering_summary.json``.

Plan: Stage 10.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from das import config  # noqa: E402
from das import io as dio  # noqa: E402
from das.annotation import load_labels, segment_lines  # noqa: E402
from das.clustering import (adjusted_rand_index, choose_k, cluster_centroids,  # noqa: E402
                            cluster_purity, factorized_kmeans, flatten_groups,
                            group_weights, kmeans, name_clusters, pca_2d, rule_labels,
                            silhouette_score, standardize)
from das.config import CLUSTER_STUDY, PipelineParams  # noqa: E402
from das.evaluation import (auc_score, gt_objects_from_mask, match_detections,  # noqa: E402
                            profile_errors, references_from_annotations,
                            split_references, vehicle_class)
from das.features import (feature_context, features_matrix, object_features,  # noqa: E402
                          section_thumbnail)
from das.pipeline import run_pipeline, run_segment  # noqa: E402
from das.visualization import (cluster_colour, plot_cluster_features,  # noqa: E402
                               plot_cluster_gallery, plot_clusters, plot_detections,
                               plot_k_selection, plot_velocity_profiles, save_figure)

FIG_DIR = config.FIG_DIR / "clustering"
REF_WINDOW = 3                     # GT v(t) window as in Stage 9 (speed class of the GT)
AUC_FEATURES = ["thickness_s", "log_peak_z", "log_area_zs", "snr_hf_db", "snr_lf_db",
                "z_hf", "z_lf", "lf_hf_db", "spec_centroid_hz", "n_sublines", "tail_asym",
                "abs_v_kmh", "log_dv_rel", "log_abs_accel", "log_v_std", "r2",
                "duration_s", "extent_m", "score"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=None,
                   help="force k (joint mode) / k per property (factorized mode)")
    p.add_argument("--mode", choices=("factorized", "joint"), default=None)
    p.add_argument("--segment", default="all", help="segment name from config.SEGMENTS or 'all'")
    p.add_argument("--synthetic", default="all",
                   help="comma-separated synthetic ids, 'all' or 'none'")
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


# ======================================================================================
# Features of all objects
# ======================================================================================
def describe(res, raw, fs, offset, pp, source: str) -> list[dict]:
    """Feature dict + straightened thumbnail of every object of one pipeline result."""
    ctx = feature_context(res.enhanced, res.row_dt, pp.feat, raw, fs, offset)
    n_ch = ctx.energy[1].shape[1]
    w = int(round(pp.feat.profile_halfwidth_s / res.row_dt))
    items = []
    for o in res.objects:
        f, cs = object_features(o, ctx, pp.feat, return_section=True)
        items.append({"source": source, "obj": o, "features": f, "row_dt": res.row_dt,
                      "dx": ctx.dx, "thumb": section_thumbnail(cs, n_ch, w)})
    return items


def collect_real(specs, pp: PipelineParams) -> tuple[list[dict], dict]:
    """Run the pipeline on the real segments; return object items and the results."""
    items, results = [], {}
    labels = load_labels()
    for spec in specs:
        seg = dio.load_named_segment(spec["name"], pad_files=1)
        res = run_segment(seg, pp)
        results[spec["name"]] = (res, seg)
        its = describe(res, seg.data, 1.0 / seg.dt, seg.pad_before, pp, spec["name"])
        refs, _ = split_references(references_from_annotations(segment_lines(labels, spec["name"])),
                                   pp.ev.ref_v_range_kmh, pp.ev.ref_min_extent_m)
        m = dict(match_detections(res.objects, refs, pp.ev.tol_t_s, pp.ev.tol_v_kmh)) if refs else {}
        for i, it in enumerate(its):
            it["manual"] = refs[m[i]]["type"] if i in m else ("unmatched" if refs else "")
            it["clock"] = (seg.start + timedelta(seconds=float(it["obj"].t_ref_s))
                           ).strftime("%H:%M:%S")
        items += its
        print(f"  {spec['name']}: {len(its)} objects"
              + (f", {len(m)} matched to {len(refs)} manual labels" if refs else
                 " (no manual labels yet)"))
    return items, results


def collect_synthetic(ids, pp: PipelineParams) -> list[dict]:
    """Run the pipeline on the synthetic samples, objects matched to the GT type."""
    items = []
    for sid in ids:
        sig, gt, meta = dio.load_synthetic(sid)
        dt, dx = dio.synthetic_axes(meta)
        res = run_pipeline(sig, pp, 1.0 / dt, dx)
        its = describe(res, sig, 1.0 / dt, 0, pp, sid)
        refs = gt_objects_from_mask(gt, meta, dt, dx)
        m = dict(match_detections(res.objects, refs, pp.ev.tol_t_s, pp.ev.tol_v_kmh))
        for i, it in enumerate(its):
            if i in m:
                r = refs[m[i]]
                it["type"] = r["type"]
                it["gt_class"] = vehicle_class(r["type"])
                it["gt_speed"] = profile_errors(it["obj"], r, pp.trk, REF_WINDOW)["class_gt"]
            else:
                it["type"], it["gt_class"], it["gt_speed"] = "false", "false", "false"
            it["gt4"] = f"{it['gt_class']}/{it['gt_speed']}"
        items += its
        print(f"  synthetic {sid}: {len(its)} objects ({len(m)} matched to {len(refs)} vehicles)")
    return items


# ======================================================================================
# Clustering
# ======================================================================================
def cluster(items: list[dict], groups, cp, mode: str | None = None,
            weighted: bool | None = None, k: int | None = None) -> dict:
    """Standardise the features of ``groups`` and cluster (factorized or joint k-means)."""
    mode = mode or cp.mode
    k = cp.k if k is None else k
    keys = flatten_groups(groups)
    X = features_matrix([it["features"] for it in items], keys)
    Zu, _, _ = standardize(X, cp.robust_scale, cp.clip_z)
    Z = Zu * group_weights(groups, cp.group_weights if weighted is None else weighted)
    if mode == "factorized":
        fk = factorized_kmeans(Zu, groups, k, cp.k_min, cp.group_k_max, cp.attempts, cp.seed,
                               cp.max_iter)
        labels, names, kk = fk["labels"], fk["names"], fk["k"]
        scores = {g: v["scores"] for g, v in fk["groups"].items()}
        chosen = {g: v["k"] for g, v in fk["groups"].items()}
        groups_info = fk["groups"]
    elif mode == "joint":
        best, sc = choose_k(Z, range(cp.k_min, cp.k_max + 1), cp.attempts, cp.seed)
        kk = k or best
        labels, _, _ = kmeans(Z, kk, cp.attempts, cp.seed, cp.max_iter)
        names = name_clusters(Zu, labels, groups, kk, cp.name_z)
        scores, chosen, groups_info = {"joint": sc}, {"joint": kk}, {}
    else:
        raise ValueError(f"Unknown clustering mode {mode!r}")
    proj, ratio, comps = pca_2d(Z)
    return {"mode": mode, "keys": keys, "X": X, "Zu": Zu, "Z": Z, "k": kk, "labels": labels,
            "names": names, "scores": scores, "chosen": chosen, "groups": groups_info,
            "silhouette": silhouette_score(Z, labels) if kk > 1 else float("nan"),
            "centroids": cluster_centroids(X, labels, kk), "pca": proj, "pca_ratio": ratio,
            "pca_components": comps}


def baseline(items: list[dict]) -> tuple[np.ndarray, list[str]]:
    f = [it["features"] for it in items]
    return rule_labels(np.array([d["log_area_zs"] for d in f]),
                       np.array([d["varying"] for d in f]))


def group_level(cl: dict, items: list[dict], group: str) -> np.ndarray | None:
    """Level of one property (factorized mode) for every object, else None."""
    if group not in cl["groups"]:
        return None
    names = cl["groups"][group]["names"]
    return np.array([next((names.index(p) for p in nm.split(", ") if p in names), -1)
                     for nm in (cl["names"][c] for c in cl["labels"])])


def validation(items: list[dict], cl: dict, synthetic: bool) -> dict:
    """Purity / adjusted Rand index of the clusters against the reference partitions."""
    lab = cl["labels"]
    rl, _ = baseline(items)
    out = {"k": cl["k"], "silhouette": cl["silhouette"],
           "ari_vs_rule_baseline": adjusted_rand_index(lab, rl)}
    thick = group_level(cl, items, "thickness")
    speed = group_level(cl, items, "speed change")
    if synthetic:
        sel = np.array([it["type"] != "false" for it in items])
        ref = {"type": [it["type"] for it in items], "heavy_light": [it["gt_class"] for it in items],
               "gt_speed": [it["gt_speed"] for it in items], "gt4": [it["gt4"] for it in items]}
        for name, r in ref.items():
            r = [x for x, s in zip(r, sel) if s]
            out[f"purity_{name}"] = cluster_purity(lab[sel], r)
            out[f"ari_{name}"] = adjusted_rand_index(lab[sel], r)
        out["baseline_ari_gt4"] = adjusted_rand_index(rl[sel], [x for x, s in zip(ref["gt4"], sel) if s])
        out["baseline_ari_heavy_light"] = adjusted_rand_index(
            rl[sel] // 2, [x for x, s in zip(ref["heavy_light"], sel) if s])
        if thick is not None:
            out["thickness_level_ari_heavy_light"] = adjusted_rand_index(
                thick[sel], [x for x, s in zip(ref["heavy_light"], sel) if s])
        if speed is not None:
            gs = [x for x, s in zip(ref["gt_speed"], sel) if s]
            out["speed_level_ari_gt_speed"] = adjusted_rand_index(speed[sel], gs)
            out["speed_level_purity_gt_speed"] = cluster_purity(speed[sel], gs)
    else:
        kind = [it["obj"].kind for it in items]
        out.update({"purity_kind": cluster_purity(lab, kind), "ari_kind": adjusted_rand_index(lab, kind)})
        if speed is not None:
            has = np.array([bool(it["obj"].speed_class) for it in items])
            st9 = [it["obj"].speed_class for it in items if it["obj"].speed_class]
            out["speed_level_ari_stage9"] = adjusted_rand_index(speed[has], st9)
            out["speed_level_agree_stage9"] = float(np.mean(
                (speed[has] == speed[has].max()) == np.array([s == "varying" for s in st9])))
        man = np.array([it.get("manual", "") in ("thick", "thin") for it in items])
        if man.any():
            mv = [items[i]["manual"] for i in np.flatnonzero(man)]
            out.update({"n_manual": int(man.sum()),
                        "purity_manual_thick_thin": cluster_purity(lab[man], mv),
                        "ari_manual_thick_thin": adjusted_rand_index(lab[man], mv),
                        "baseline_ari_manual": adjusted_rand_index(rl[man] // 2, mv),
                        "kind_ari_manual": adjusted_rand_index(np.array(kind)[man], mv)})
            if thick is not None:
                out["thickness_level_ari_manual"] = adjusted_rand_index(thick[man], mv)
                out["thickness_level_agree_manual"] = float(np.mean(
                    (thick[man] == thick[man].max()) == np.array([m == "thick" for m in mv])))
    return out


def composition(items: list[dict], cl: dict, field: str) -> dict:
    """{cluster: Counter of a reference field} (e.g. synthetic type per cluster)."""
    def value(it):
        return it[field] if field in it else getattr(it["obj"], field)
    return {f"{c}: {cl['names'][c]}": dict(Counter(value(items[i]) for i in
                                                    np.flatnonzero(cl["labels"] == c)))
            for c in range(cl["k"])}


def feature_auc(syn_items: list[dict], real_items: list[dict]) -> list[dict]:
    """AUC of every descriptor for the reference partitions (separability study)."""
    def auc(items, key, pos_fn, use_fn):
        """AUC of one feature for the objects selected by ``use_fn`` (positives: ``pos_fn``)."""
        it = [x for x in items if use_fn(x)]
        v = np.array([x["features"][key] for x in it], float)
        p = np.array([pos_fn(x) for x in it])
        ok = np.isfinite(v)
        return auc_score(v[ok], p[ok]) if ok.any() else float("nan")
    rows = []
    for key in AUC_FEATURES:
        row = {"feature": key}
        if syn_items:
            row["syn_heavy_vs_light"] = auc(syn_items, key, lambda x: x["gt_class"] == "heavy",
                                            lambda x: x["type"] != "false")
            row["syn_varying_vs_constant"] = auc(syn_items, key,
                                                 lambda x: x["gt_speed"] == "varying",
                                                 lambda x: x["gt_speed"] in ("varying", "constant"))
        row["real_strong_vs_weak"] = auc(real_items, key, lambda x: x["obj"].kind == "strong",
                                         lambda x: True)
        row["real_varying_vs_constant_stage9"] = auc(
            real_items, key, lambda x: x["obj"].speed_class == "varying",
            lambda x: bool(x["obj"].speed_class))
        if any(x.get("manual") in ("thick", "thin") for x in real_items):
            row["real_manual_thick_vs_thin"] = auc(real_items, key,
                                                   lambda x: x["manual"] == "thick",
                                                   lambda x: x.get("manual") in ("thick", "thin"))
        rows.append(row)
    return rows


# ======================================================================================
# Figures and tables
# ======================================================================================
def caption(it: dict) -> str:
    f = it["features"]
    src = it["source"].replace("seg_", "")
    return f"{src}#{it['obj'].obj_id} {f['abs_v_kmh']:.0f} km/h"


def figures(items: list[dict], cl: dict, tag: str, title: str, dpi: int, groups) -> None:
    """Figures of one run: PCA scatter, loadings, k selection, feature box plots, gallery."""
    d = FIG_DIR / tag
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), layout="constrained",
                             gridspec_kw={"width_ratios": [1.6, 1]})
    plot_clusters(cl["pca"], cl["labels"], cl["names"], ax=axes[0], groups=groups,
                  explained=cl["pca_ratio"],
                  title=f"{title}: {cl['mode']} k-means, PCA of the standardised features")
    load = cl["pca_components"]
    y = np.arange(len(cl["keys"]))
    axes[1].barh(y - 0.2, load[0], height=0.4, label="PC1")
    if load.shape[0] > 1:
        axes[1].barh(y + 0.2, load[1], height=0.4, label="PC2")
    axes[1].set_yticks(y, cl["keys"], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].legend(fontsize=8)
    axes[1].set_title("loadings of the principal components")
    save_figure(fig, d / "pca.png", dpi)
    sc = {g: s for g, s in cl["scores"].items() if s}
    if sc:
        fig, axes = plt.subplots(1, len(sc), figsize=(6 * len(sc), 3.6), layout="constrained",
                                 squeeze=False)
        for ax, (g, s) in zip(axes.flat, sc.items()):
            plot_k_selection(s, cl["chosen"][g], f"{title}: {g} (chosen k = {cl['chosen'][g]})",
                             ax=ax)
        save_figure(fig, d / "k_selection.png", dpi)
    save_figure(plot_cluster_features(cl["X"], cl["labels"], cl["keys"], cl["names"]),
                d / "features.png", dpi)
    save_figure(plot_cluster_gallery([it["thumb"] for it in items], cl["labels"], cl["names"],
                                     [caption(it) for it in items], items[0]["row_dt"],
                                     items[0]["dx"],
                                     title=f"{title}: straightened lines per cluster (vertical: "
                                           f"+-2 s around the ridge, horizontal: cable position)"),
                d / "gallery.png", dpi)


def figure_auc(rows: list[dict], dpi: int) -> None:
    """Horizontal bars of the single-feature AUC (thick vs thin, constant vs varying)."""
    cols = [c for c in rows[0] if c != "feature"]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(rows) + 2), layout="constrained")
    h = 0.8 / len(cols)
    for j, c in enumerate(cols):
        ax.barh(y + (j - (len(cols) - 1) / 2) * h, [abs(r[c] - 0.5) * 2 for r in rows],
                height=h, label=c)
    ax.set_yticks(y, [r["feature"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("separability |2 AUC - 1| (0 = none, 1 = perfect)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Which descriptor separates which classes (basis of the feature choice)")
    save_figure(fig, FIG_DIR / "feature_separability.png", dpi)


def segment_figures(results: dict, items: list[dict], cl: dict, dpi: int) -> None:
    """Waterfall and v(t) of every real segment, objects coloured by cluster."""
    for name, (res, seg) in results.items():
        idx = [i for i, it in enumerate(items) if it["source"] == name]
        objs = [items[i]["obj"] for i in idx]
        cols = [cluster_colour(cl["labels"][i]) for i in idx]
        raw = np.log10(res.pre.common["raw"] + 1e-12)
        fig, axes = plt.subplots(1, 2, figsize=(17, 10), layout="constrained",
                                 gridspec_kw={"width_ratios": [1, 1.4]})
        plot_detections(raw, objs, res.pre.row_dt, res.pre.col_dx, seg.start, ax=axes[0],
                        vmin=float(np.percentile(raw, 1)), vmax=float(np.percentile(raw, 99.5)),
                        cmap="gray", colours=cols, label_fontsize=7,
                        title=f"{name}: objects coloured by cluster")
        plot_velocity_profiles(objs, seg.start, ax=axes[1], colours=cols,
                               title=f"{name}: v(t) coloured by cluster")
        handles = [plt.Line2D([], [], color=cluster_colour(c), lw=2, label=f"{c}: {n}")
                   for c, n in enumerate(cl["names"])]
        axes[1].legend(handles=handles, fontsize=8, loc="upper left")
        save_figure(fig, FIG_DIR / "real" / f"{name}_clusters.png", dpi)


def write_csv(rows: list[dict], name: str) -> None:
    """Write rows with the union of their keys as the CSV header."""
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(config.RESULTS_DIR / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_tables(items: list[dict], cl: dict, prefix: str, extra_fields) -> None:
    """Write the per-object cluster assignment table of one run."""
    rows = []
    for i, it in enumerate(items):
        o = it["obj"]
        rows.append({"source": it["source"], "id": o.obj_id, "cluster": int(cl["labels"][i]),
                     "cluster_name": cl["names"][cl["labels"][i]], "kind": o.kind,
                     "speed_class": o.speed_class, **{f: it.get(f, "") for f in extra_fields},
                     "pc1": cl["pca"][i, 0], "pc2": cl["pca"][i, 1], **it["features"]})
    write_csv(rows, f"{prefix}_objects.csv")
    cen = cl["centroids"]
    write_csv([{"cluster": c, "name": cl["names"][c], "n": int(np.sum(cl["labels"] == c)),
                **{k: round(float(cen[c, j]), 3) for j, k in enumerate(cl["keys"])},
                **{f"median_{k}": round(float(np.nanmedian(
                    [items[i]["features"][k] for i in np.flatnonzero(cl["labels"] == c)])), 3)
                   for k in ("thickness_s", "abs_v_kmh", "accel_ms2", "dv_rel")}}
               for c in range(cl["k"])], f"{prefix}_centroids.csv")


def print_clusters(cl: dict, items: list[dict]) -> None:
    """Print the mean feature values of every cluster."""
    keys = cl["keys"] + ["thickness_s", "abs_v_kmh", "dv_rel"]
    X = features_matrix([it["features"] for it in items], keys)
    print("    " + " " * 38 + " ".join(f"{k[:11]:>11}" for k in keys))
    for c in range(cl["k"]):
        sel = cl["labels"] == c
        med = [np.nanmedian(X[sel, j]) if np.isfinite(X[sel, j]).any() else np.nan
               for j in range(len(keys))]
        print(f"    {c} {cl['names'][c][:30]:<30} n={int(sel.sum()):<3}"
              + " ".join(f"{v:11.2f}" for v in med))


def summary_block(items, cl, val, comps) -> dict:
    """JSON summary of one clustering run (k, scores, cluster sizes, validation)."""
    return {"n_objects": len(items), "mode": cl["mode"], "k": cl["k"], "chosen": cl["chosen"],
            "scores": cl["scores"], "names": cl["names"],
            "sizes": np.bincount(cl["labels"], minlength=cl["k"]).tolist(),
            "groups": {g: {k: v for k, v in info.items() if k != "scores"}
                       for g, info in cl["groups"].items()},
            "validation": val, "pca_explained": cl["pca_ratio"],
            **{f"composition_{f}": composition(items, cl, f) for f in comps}}


# ======================================================================================
def main() -> None:
    args = parse_args()
    pp = PipelineParams()
    cp = replace(pp.clu, **({"mode": args.mode} if args.mode else {}),
                 **({"k": args.k} if args.k else {}))
    specs = config.SEGMENTS if args.segment == "all" else [dio.get_segment_spec(args.segment)]
    ids = ([] if args.synthetic == "none" else dio.list_synthetic_ids()
           if args.synthetic == "all" else args.synthetic.split(","))
    summary = {"feature_params": asdict(pp.feat), "clustering_params": asdict(cp)}

    print(f"[1/5] real segments {[s['name'] for s in specs]}: pipeline + features")
    real_items, results = collect_real(specs, pp)
    syn_items = []
    if ids:
        print(f"[2/5] synthetic samples {ids}: pipeline + features")
        syn_items = collect_synthetic(ids, pp)

    print("[3/5] feature separability (AUC)")
    auc_rows = feature_auc(syn_items, real_items)
    for r in auc_rows:
        print(f"  {r['feature']:<17}" + "  ".join(f"{k} {v:.2f}" for k, v in r.items()
                                                  if k != "feature"))
    write_csv(auc_rows, "clustering_feature_auc.csv")
    figure_auc(auc_rows, args.dpi)
    summary["feature_auc"] = auc_rows

    print(f"[4/5] clustering ({cp.mode}, features {flatten_groups(cp.feature_groups)})")
    real = cluster(real_items, cp.feature_groups, cp)
    real_val = validation(real_items, real, synthetic=False)
    print(f"  real: k = {real['k']} {real['chosen']}, silhouette {real['silhouette']:.3f}")
    print_clusters(real, real_items)
    print(f"  validation: {json.dumps(to_jsonable(real_val))}")
    figures(real_items, real, "real", "Real segments (3 x 2 min)", args.dpi,
            [it["source"] for it in real_items])
    segment_figures(results, real_items, real, args.dpi)
    write_tables(real_items, real, "clustering_real", ["clock", "manual"])
    summary["real"] = summary_block(real_items, real, real_val, ["kind", "speed_class", "source"]
                                    + (["manual"] if "n_manual" in real_val else []))
    if syn_items:
        syn = cluster(syn_items, cp.feature_groups, cp)
        syn_val = validation(syn_items, syn, synthetic=True)
        print(f"  synthetic: k = {syn['k']} {syn['chosen']}, silhouette {syn['silhouette']:.3f}")
        print_clusters(syn, syn_items)
        print(f"  validation: {json.dumps(to_jsonable(syn_val))}")
        print(f"  composition (type): {composition(syn_items, syn, 'type')}")
        print(f"  composition (GT speed): {composition(syn_items, syn, 'gt_speed')}")
        figures(syn_items, syn, "synthetic", "Synthetic samples", args.dpi,
                [it["type"] for it in syn_items])
        write_tables(syn_items, syn, "clustering_synthetic", ["type", "gt_class", "gt_speed"])
        summary["synthetic"] = summary_block(syn_items, syn, syn_val, ["type", "gt_speed", "gt4"])

    print(f"[5/5] method / feature-set study ({len(CLUSTER_STUDY)} variants)")
    study = []
    for name, (groups, mode, weighted, k) in CLUSTER_STUDY.items():
        for tag, items in (("synthetic", syn_items), ("real", real_items)):
            if not items:
                continue
            cl = cluster(items, groups, replace(cp, k=0), mode, weighted, k)
            val = validation(items, cl, synthetic=tag == "synthetic")
            study.append({"data": tag, "variant": name, "names": " | ".join(cl["names"]), **val})
            keys = (("ari_gt4", "purity_gt4", "ari_heavy_light", "ari_gt_speed")
                    if tag == "synthetic" else ("ari_vs_rule_baseline", "ari_kind",
                                                "ari_manual_thick_thin"))
            print(f"  {tag:<9} {name:<45} k={cl['k']:<2} sil {val['silhouette']:.3f}  "
                  + "  ".join(f"{m} {val[m]:.2f}" for m in keys if m in val))
    write_csv(study, "clustering_study.csv")
    summary["study"] = study
    (config.RESULTS_DIR / "clustering_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=1), encoding="utf-8")
    print(f"\nFigures: {FIG_DIR}\nResults: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
