#!/usr/bin/env python3
"""Simulation engine for the MSB-GP genetic-architecture benchmark.

Generates nine genetic architectures spanning pure additive, additive ->
dominance transitions, and epistatic (AA / AD / DD / mixed) effects, and
provides shared helpers for the simulation benchmarks and figure scripts.

Architectures (CATTLE-style):
  Pheno:0 = 100% additive
  Pheno:1 = 75% A + 25% D
  Pheno:2 = 50% A + 50% D
  Pheno:3 = 25% A + 75% D
  Pheno:4 = 100% dominance
  Pheno:5 = 33% A + 33% D + 34% AA epistasis
  Pheno:6 = 33% A + 33% D + 34% AD epistasis
  Pheno:7 = 33% A + 33% D + 34% DD epistasis
  Pheno:8 = mixed A + D + all epistasis

Datasets are written under ``<results>/simulation_benchmark`` as one ``.npz``
per architecture (sample matrix, phenotype and true genetic value).
"""

import os
import sys
import time
import json
import argparse

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from msbgp.bases import GBLUP, RRBLUP, BayesA, BayesC
from msbgp.msbgp import MSBGP
from msbgp.calibration import MSBCal
from msbgp.ml_baselines import RandomForest
from msbgp.gmkls import MeanGMKL, FHGMKL, CKAGMKL
from msbgp.cv import random_kfold
from msbgp.metrics import pearson_corr

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

# --------------------------------------------------------------------------- #
# Output directory (all results live under the shared results tree)
# --------------------------------------------------------------------------- #
_H_CANDIDATE = ("H:\\\u73b0\u4ee3\u519c\u4e1a\u7814\u7a76\u9662\u674e\u535a\u751f\u8001\u5e08\u8bfe\u9898\u7ec4"
                "\\AI\u80b2\u79cd\u5927\u8d5b\\results")

ARCHITECTURES = ["Pheno:0", "Pheno:1", "Pheno:2", "Pheno:3", "Pheno:4",
                 "Pheno:5", "Pheno:6", "Pheno:7", "Pheno:8"]

ARCH_LABELS = {
    "Pheno:0": "100% A",
    "Pheno:1": "75% A + 25% D",
    "Pheno:2": "50% A + 50% D",
    "Pheno:3": "25% A + 75% D",
    "Pheno:4": "100% D",
    "Pheno:5": "A+D+AA-epi",
    "Pheno:6": "A+D+AD-epi",
    "Pheno:7": "A+D+DD-epi",
    "Pheno:8": "Mixed (A+D+all epi)",
}

METHOD_COLORS = {
    "GBLUP": "#9E9E9E", "RR-BLUP": "#C8C8C8",
    "BayesA": "#E15759", "BayesCpi": "#FF9D9A",
    "MSB-GP": "#B07AA1", "MSB-GP-Add": "#D4A6C8",
    "RandomForest": "#F28E2B", "DNNGP": "#4E79A7",
    "MeanGMKL": "#76B7B2", "FHGMKL": "#59A14F", "CKAGMKL": "#1F77B4",
}


def resolve_out_dir():
    parent = os.path.dirname(_H_CANDIDATE)
    if os.path.isdir(parent):
        out = os.path.join(_H_CANDIDATE, "simulation_benchmark")
    else:
        out = os.path.join(ROOT, "..", "simulation_benchmark")
    os.makedirs(out, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Simulation (n is configurable)
# --------------------------------------------------------------------------- #
def simulate_architecture(arch_name, n=800, m=1000, h2=0.6, seed=2026):
    rng = np.random.default_rng(seed)
    p_freq = rng.uniform(0.1, 0.5, size=m)
    a1 = (rng.random((n, m)) < p_freq).astype(float)
    a2 = (rng.random((n, m)) < p_freq).astype(float)
    X = a1 + a2
    Xc = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    n_qtl = max(50, m // 10)
    qtl_idx = rng.choice(m, size=n_qtl, replace=False)
    Xq = Xc[:, qtl_idx]

    alpha = rng.normal(0, 1, size=n_qtl)
    g_add = Xq @ alpha

    X_dom = (X[:, qtl_idx] == 1).astype(float)
    X_dom = (X_dom - X_dom.mean(axis=0)) / (X_dom.std(axis=0) + 1e-12)
    delta = rng.normal(0, 1, size=n_qtl)
    g_dom = X_dom @ delta

    n_epi = min(30, n_qtl)
    g_epi_aa = np.zeros(n)
    g_epi_ad = np.zeros(n)
    g_epi_dd = np.zeros(n)
    for i in range(n_epi - 1):
        for j in range(i + 1, n_epi):
            g_epi_aa += rng.normal(0, 1) * Xq[:, i] * Xq[:, j]
            g_epi_ad += rng.normal(0, 1) * Xq[:, i] * X_dom[:, j]
            g_epi_dd += rng.normal(0, 1) * X_dom[:, i] * X_dom[:, j]

    def _norm(v):
        return (v - v.mean()) / (v.std() + 1e-12)

    g_add, g_dom = _norm(g_add), _norm(g_dom)
    g_epi_aa, g_epi_ad, g_epi_dd = _norm(g_epi_aa), _norm(g_epi_ad), _norm(g_epi_dd)

    if arch_name == "Pheno:0":
        g = 1.0 * g_add
    elif arch_name == "Pheno:1":
        g = 0.75 * g_add + 0.25 * g_dom
    elif arch_name == "Pheno:2":
        g = 0.50 * g_add + 0.50 * g_dom
    elif arch_name == "Pheno:3":
        g = 0.25 * g_add + 0.75 * g_dom
    elif arch_name == "Pheno:4":
        g = 1.0 * g_dom
    elif arch_name == "Pheno:5":
        g = 0.33 * g_add + 0.33 * g_dom + 0.34 * g_epi_aa
    elif arch_name == "Pheno:6":
        g = 0.33 * g_add + 0.33 * g_dom + 0.34 * g_epi_ad
    elif arch_name == "Pheno:7":
        g = 0.33 * g_add + 0.33 * g_dom + 0.34 * g_epi_dd
    elif arch_name == "Pheno:8":
        g = (0.20 * g_add + 0.20 * g_dom + 0.20 * g_epi_aa
             + 0.20 * g_epi_ad + 0.20 * g_epi_dd)
    else:
        raise ValueError(f"Unknown architecture {arch_name}")

    g = _norm(g)
    var_e = (1.0 - h2) / h2
    y = g + rng.normal(0.0, np.sqrt(var_e), size=n)
    return X.astype(np.float32), y.astype(np.float64), g.astype(np.float64)


def build_methods(with_dnngp=False, with_gmkl=True):
    methods = {
        "GBLUP":        lambda: GBLUP(),
        "RR-BLUP":      lambda: RRBLUP(),
        "BayesA":       lambda: BayesA(n_iter=300, burnin=100),
        "BayesCpi":     lambda: BayesC(n_iter=300, burnin=100),
        "MSB-GP":       lambda: MSBCal(k_inner=5, seed=0),
        "MSB-GP-Add":   lambda: MSBCal(kernels=["additive"], k_inner=5, seed=0),
        "RandomForest": lambda: RandomForest(n_estimators=100, max_depth=8,
                                             max_features="sqrt", seed=0),
    }
    if with_gmkl:
        methods["MeanGMKL"] = lambda: MeanGMKL()
        methods["FHGMKL"] = lambda: FHGMKL()
        methods["CKAGMKL"] = lambda: CKAGMKL()
    if with_dnngp:
        from msbgp.dnngp import DNNGP
        methods["DNNGP"] = lambda: DNNGP(epochs=150, patience=15, seed=0)
    return methods


def run_arch(arch, X, y, g_true, methods, K, n_repeats, seeds, log):
    rows = []
    for name, ctor in methods.items():
        accs, pooled_rs, pooled_sps, mses, true_rs = [], [], [], [], []
        t0 = time.time()
        for rep in range(n_repeats):
            seed = seeds[rep]
            model = ctor()
            r = random_kfold(X.astype(float), y, model, k=K, seed=seed)
            accs.extend(list(r["accuracy"]))
            pred, true = r["pooled_pred"], r["pooled_true"]
            pooled_rs.append(pearson_corr(true, pred))
            mses.append(float(np.mean((pred - true) ** 2)))
            true_rs.append(pearson_corr(g_true, pred))
            if spearmanr is not None:
                pooled_sps.append(float(spearmanr(true, pred).correlation))
            else:
                pooled_sps.append(np.nan)
        rt = time.time() - t0
        row = {
            "architecture": arch,
            "arch_label": ARCH_LABELS[arch],
            "method": name,
            "perfold_pearson_mean": round(float(np.mean(accs)), 4),
            "perfold_pearson_std": round(float(np.std(accs)), 4),
            "pooled_pearson_mean": round(float(np.mean(pooled_rs)), 4),
            "pooled_pearson_std": round(float(np.std(pooled_rs)), 4),
            "pooled_spearman_mean": round(float(np.nanmean(pooled_sps)), 4),
            "true_genetic_accuracy": round(float(np.mean(true_rs)), 4),
            "mse_mean": round(float(np.mean(mses)), 4),
            "mse_std": round(float(np.std(mses)), 4),
            "runtime_s": round(rt, 1),
        }
        rows.append(row)
        log(f"  {name:>14s}: perfold={row['perfold_pearson_mean']:.3f}"
            f"+-{row['perfold_pearson_std']:.3f}  pooled={row['pooled_pearson_mean']:.3f}"
            f"  trueG={row['true_genetic_accuracy']:.3f}  ({rt:.1f}s)")
    return rows


def make_figures(df, method_order, out_dir, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    def grouped(metric_mean, metric_std, fname, ylabel, title):
        fig, axes = plt.subplots(3, 3, figsize=(16, 13.5))
        x = np.arange(len(method_order))
        colors = [METHOD_COLORS.get(m, "#999999") for m in method_order]
        for ai, arch in enumerate(ARCHITECTURES):
            ax = axes[ai // 3][ai % 3]
            sub = df[df["architecture"] == arch]
            means = [sub.loc[sub["method"] == m, metric_mean].mean()
                     if (sub["method"] == m).any() else 0.0 for m in method_order]
            stds = [sub.loc[sub["method"] == m, metric_std].mean()
                    if (sub["method"] == m).any() else 0.0 for m in method_order]
            ax.bar(x, means, yerr=stds, capsize=2, color=colors,
                   edgecolor="white", linewidth=0.4)
            for xi, (mv, sv) in enumerate(zip(means, stds)):
                ax.text(xi, (mv or 0) + (sv or 0) + 0.008, f"{mv:.2f}",
                        ha="center", va="bottom", fontsize=4.5, rotation=90)
            ax.set_title(f"{arch}\n{ARCH_LABELS[arch]}", fontsize=8)
            ax.set_ylim(0, max(0.6, max([m for m in means if np.isfinite(m)] or [0])
                               + max(stds or [0]) + 0.1))
            ax.axhline(0, color="black", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(method_order, rotation=45, ha="right", fontsize=6)
            ax.grid(axis="y", alpha=0.2, linewidth=0.3)
            if ai % 3 == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if ai // 3 == 2:
                ax.set_xlabel("Method", fontsize=8)
        fig.suptitle(title, fontsize=13, y=1.0)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{fname}.{ext}"), dpi=600,
                        bbox_inches="tight")
        plt.close(fig)
        log(f"Figure saved: {fname}.png/.pdf")

    grouped("perfold_pearson_mean", "perfold_pearson_std",
            "fig_simulation_pearson", "Pearson r (per-fold mean)",
            "Per-fold prediction accuracy across architectures")
    grouped("pooled_pearson_mean", "pooled_pearson_std",
            "fig_simulation_pooled", "Pooled Pearson r",
            "Pooled prediction accuracy across architectures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--h2", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--with-dnngp", action="store_true")
    ap.add_argument("--no-gmkl", action="store_true")
    ap.add_argument("--save-data", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = resolve_out_dir()
    logf = open(os.path.join(out_dir, "simulation.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    log("=" * 68)
    log("Simulation benchmark: genetic architectures")
    log(f"N={args.n} M={args.m} h2={args.h2} K={args.k} repeats={args.repeats}")
    log(f"Output: {out_dir}")
    log("=" * 68)

    data_dir = os.path.join(out_dir, "simulation_datasets")
    os.makedirs(data_dir, exist_ok=True)

    seeds = [args.seed + rep * 100 for rep in range(args.repeats)]
    methods = build_methods(with_dnngp=args.with_dnngp,
                            with_gmkl=not args.no_gmkl)
    method_order = list(methods.keys())

    all_rows = []
    manifest = []
    for arch in ARCHITECTURES:
        log(f"\n{'='*60}\n  {arch} ({ARCH_LABELS[arch]})\n{'='*60}")
        X, y, g = simulate_architecture(arch, n=args.n, m=args.m,
                                        h2=args.h2, seed=args.seed)
        log(f"  X:{X.shape} var(y)={np.var(y):.3f} var(g)={np.var(g):.3f}")
        if args.save_data:
            np.savez_compressed(os.path.join(data_dir, f"{arch.replace(':','')}.npz"),
                                X=X, y=y, g=g)
            manifest.append({"architecture": arch, "n": args.n, "m": args.m,
                             "h2": args.h2, "seed": args.seed,
                             "var_y": float(np.var(y)), "var_g": float(np.var(g))})
        rows = run_arch(arch, X, y, g, methods, args.k, args.repeats, seeds, log)
        all_rows.extend(rows)
        # incremental checkpoint
        pd.DataFrame(all_rows).to_csv(
            os.path.join(out_dir, "simulation_results.csv"), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, "simulation_results.csv"), index=False)
    with open(os.path.join(out_dir, "simulation_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"params": vars(args), "datasets": manifest}, fh, indent=2)
    log(f"\nSaved results -> {os.path.join(out_dir, 'simulation_results.csv')}")

    # Console summary (pooled Pearson)
    piv = df.pivot(index="architecture", columns="method",
                   values="pooled_pearson_mean").reindex(ARCHITECTURES)
    log("\nPooled Pearson summary:\n" + piv.to_string())

    try:
        make_figures(df, method_order, out_dir, log)
    except Exception as e:  # pragma: no cover
        import traceback
        log("Figure error: " + repr(e))
        log(traceback.format_exc())

    log("\nALL_DONE")
    logf.close()


if __name__ == "__main__":
    main()




