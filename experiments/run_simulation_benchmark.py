#!/usr/bin/env python3
"""run_simulation_benchmark_stable.py — Systematic simulation benchmark across genetic architectures.

Simulates 9 phenotype types spanning from pure additive to complex epistasis,
then benchmarks all methods (GBLUP, RR-BLUP, BayesA, BayesC, MSB-GP,
RandomForest, DNNGP) on each.

Architecture design (following the CATTLE benchmark style):
  Pheno:0 = 100% additive
  Pheno:1 = 75% additive + 25% dominance
  Pheno:2 = 50% additive + 50% dominance
  Pheno:3 = 25% additive + 75% dominance
  Pheno:4 = 100% dominance
  Pheno:5 = 33% A + 33% D + 34% additive-by-additive epistasis
  Pheno:6 = 33% A + 33% D + 34% additive-by-dominance epistasis
  Pheno:7 = 33% A + 33% D + 34% dominance-by-dominance epistasis
  Pheno:8 = 33% A + 33% D + 34% mixed epistasis (all types combined)

Outputs:
  simulation_results.csv      species=pheno_type, method, pearson, spearman, MSE
  fig_simulation.png/pdf      grouped bar chart (Pearson + Spearman), 600 dpi
"""

import os, sys, time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from msbgp.bases import GBLUP, RRBLUP, BayesA, BayesC
from msbgp.msbgp import MSBGP
from msbgp.calibration import MSBCal
from msbgp.ml_baselines import RandomForest
from msbgp.dnngp import DNNGP
from msbgp.cv import random_kfold
from msbgp.gmkls import MeanGMKL, FHGMKL, CKAGMKL
from msbgp.metrics import pearson_corr

_H_ROOT = r"H:\现代农业研究院李博生老师课题组\AI育种大赛\results"
if os.path.isdir(os.path.dirname(_H_ROOT)):
    OUT_DIR = os.path.join(_H_ROOT, "simulation_benchmark_stable")
else:
    OUT_DIR = os.path.join(ROOT, "..", "simulation_benchmark_stable")
os.makedirs(OUT_DIR, exist_ok=True)

N = 300       # samples
M = 1000     # markers
K = 5         # CV folds
N_REPEATS = 2
SEED = 2026
H2 = 0.6     # broad-sense heritability


# ------------------------------------------------------------------ #
# Simulation
# ------------------------------------------------------------------ #
def simulate_architecture(arch_name, n=N, m=M, h2=H2, seed=0):
    """Simulate genotype + phenotype with specified genetic architecture."""
    rng = np.random.default_rng(seed)
    # biallelic genotypes {0,1,2}
    p_freq = rng.uniform(0.1, 0.5, size=m)
    # Simulate as sum of two independent Bernoulli draws per allele
    a1 = (rng.random((n, m)) < p_freq).astype(float)
    a2 = (rng.random((n, m)) < p_freq).astype(float)
    X = a1 + a2  # {0, 1, 2} allele counts
    # centre & scale
    Xc = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # select causal subset
    n_qtl = max(50, m // 10)
    qtl_idx = rng.choice(m, size=n_qtl, replace=False)
    Xq = Xc[:, qtl_idx]

    # additive effects
    alpha = rng.normal(0, 1, size=n_qtl)
    g_add = Xq @ alpha

    # dominance effects (dominance coding: 1 for heterozygotes, 0 otherwise)
    X_dom = (X[:, qtl_idx] == 1).astype(float)
    X_dom = (X_dom - X_dom.mean(axis=0)) / (X_dom.std(axis=0) + 1e-12)
    delta = rng.normal(0, 1, size=n_qtl)
    g_dom = X_dom @ delta

    # epistatic effects (pairwise among first 30 QTL)
    n_epi = min(30, n_qtl)
    g_epi_aa = np.zeros(n); g_epi_ad = np.zeros(n); g_epi_dd = np.zeros(n)
    for i in range(n_epi - 1):
        for j in range(i + 1, n_epi):
            c = rng.normal(0, 1)
            g_epi_aa += c * Xq[:, i] * Xq[:, j]
            c2 = rng.normal(0, 1)
            g_epi_ad += c2 * Xq[:, i] * X_dom[:, j]
            c3 = rng.normal(0, 1)
            g_epi_dd += c3 * X_dom[:, i] * X_dom[:, j]

    # normalise each component
    def _norm(v):
        return (v - v.mean()) / (v.std() + 1e-12)
    g_add, g_dom = _norm(g_add), _norm(g_dom)
    g_epi_aa, g_epi_ad, g_epi_dd = _norm(g_epi_aa), _norm(g_epi_ad), _norm(g_epi_dd)

    # mix according to architecture
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
        g = 0.20 * g_add + 0.20 * g_dom + 0.20 * g_epi_aa + \
            0.20 * g_epi_ad + 0.20 * g_epi_dd
    else:
        raise ValueError(f"Unknown architecture {arch_name}")

    g = _norm(g)
    var_g = np.var(g)
    var_e = var_g * (1.0 - h2) / h2
    e = rng.normal(0, np.sqrt(var_e), size=n)
    y = g + e
    return X, y, g


ARCHITECTURES = ["Pheno:0", "Pheno:1", "Pheno:2", "Pheno:3", "Pheno:4",
                 "Pheno:5", "Pheno:6", "Pheno:7", "Pheno:8"]

ARCH_LABELS = {
    "Pheno:0": "100% A",
    "Pheno:1": "75% A + 25% D",
    "Pheno:2": "50% A + 50% D",
    "Pheno:3": "25% A + 75% D",
    "Pheno:4": "100% D",
    "Pheno:5": "33A+33D+34 AA-epi",
    "Pheno:6": "33A+33D+34 AD-epi",
    "Pheno:7": "33A+33D+34 DD-epi",
    "Pheno:8": "Mixed (A+D+all epi)",
}


def build_methods():
    return {
        "GBLUP":        GBLUP(),
        "RR-BLUP":      RRBLUP(),
        "BayesA":       BayesA(n_iter=500, burnin=150),
        "BayesCpi":     BayesC(n_iter=500, burnin=150),
        "MSB-GP":       MSBCal(k_inner=5, seed=0),
        "MSB-GP-Add":   MSBCal(kernels=["additive"], k_inner=5, seed=0),
        "RandomForest": RandomForest(n_estimators=100, max_depth=8,
                                     max_features="sqrt", seed=0),
        "DNNGP":        DNNGP(epochs=100, lr=0.001, batch_size=32, seed=0),
        "MeanGMKL":     MeanGMKL(),
        "FHGMKL":       FHGMKL(),
        "CKAGMKL":      CKAGMKL(),
    }


def main():
    logf = open(os.path.join(OUT_DIR, "simulation.log"), "w", encoding="utf-8")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log(f"Simulation benchmark: N={N}, M={M}, h2={H2}, {K}-fold x {N_REPEATS} repeats")
    log(f"Output: {OUT_DIR}\n")

    methods = build_methods()
    method_order = list(methods.keys())
    rows = []

    for arch in ARCHITECTURES:
        t0 = time.time()
        log(f"\n{'='*60}")
        log(f"  {arch} ({ARCH_LABELS[arch]})")
        log(f"{'='*60}")

        X, y, g_true = simulate_architecture(arch, seed=SEED)
        log(f"  X: {X.shape}, var(y)={np.var(y):.3f}")

        for name, method in methods.items():
            accs, spearmans, mses = [], [], []
            t1 = time.time()
            for rep in range(N_REPEATS):
                seed_rep = SEED + rep * 100
                r = random_kfold(X, y, method, k=K, seed=seed_rep)
                for a in r["accuracy"]:
                    accs.append(a)
                pred, true = r["pooled_pred"], r["pooled_true"]
                sp = spearmanr(true, pred).correlation
                spearmans.append(sp)
                mses.append(float(np.mean((pred - true) ** 2)))
            rt = time.time() - t1
            row = {
                "architecture": arch,
                "arch_label": ARCH_LABELS[arch],
                "method": name,
                "pearson_mean": round(float(np.mean(accs)), 4),
                "pearson_std": round(float(np.std(accs)), 4),
                "spearman_mean": round(float(np.mean(spearmans)), 4),
                "spearman_std": round(float(np.std(spearmans)), 4),
                "mse_mean": round(float(np.mean(mses)), 4),
                "mse_std": round(float(np.std(mses)), 4),
                "runtime_s": round(rt, 1),
            }
            rows.append(row)
            log(f"  {name:>14s}: r={row['pearson_mean']:.4f}±{row['pearson_std']:.4f}  "
                f"sp={row['spearman_mean']:.4f}  ({rt:.1f}s)")

        log(f"  {arch} done in {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "simulation_results.csv")
    df.to_csv(csv_path, index=False)
    log(f"\nSaved -> {csv_path}")

    # ---- figure ----
    log("Generating figure ...")
    try:
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

        METHOD_COLORS = {
            "GBLUP": "#AAAAAA", "RR-BLUP": "#CCCCCC",
            "BayesA": "#E15759", "BayesCpi": "#FF9D9A",
            "MSB-GP": "#B07AA1", "MSB-GP-Add": "#D4A6C8",
            "RandomForest": "#F28E2B", "DNNGP": "#4E79A7",
            "MeanGMKL": "#76B7B2", "FHGMKL": "#59A14F", "CKAGMKL": "#1F77B4",
        }

        # 9 architectures in 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        x = np.arange(len(method_order))
        for ai, arch in enumerate(ARCHITECTURES):
            ax = axes[ai // 3][ai % 3]
            sub = df[df["architecture"] == arch]
            means, stds = [], []
            for m in method_order:
                row = sub[sub["method"] == m]
                if len(row):
                    means.append(row["pearson_mean"].values[0])
                    stds.append(row["pearson_std"].values[0])
                else:
                    means.append(0); stds.append(0)
            colors = [METHOD_COLORS.get(m, "#999999") for m in method_order]
            bars = ax.bar(x, means, yerr=stds, capsize=2, color=colors,
                          edgecolor="white", linewidth=0.4)
            for xi, (mv, sv) in enumerate(zip(means, stds)):
                ax.text(xi, mv + sv + 0.008, f"{mv:.3f}",
                        ha="center", va="bottom", fontsize=5, rotation=90,
                        fontweight="bold")
            ax.set_title(f"{arch}\n{ARCH_LABELS[arch]}", fontsize=8)
            ax.set_ylim(0, max(max(means) + max(stds) + 0.1, 0.6))
            ax.set_xticks(x)
            ax.set_xticklabels(method_order, rotation=45, ha="right", fontsize=6)
            ax.grid(axis="y", alpha=0.2, linewidth=0.3)
            if ai % 3 == 0:
                ax.set_ylabel("Pearson r", fontsize=8)
            if ai // 3 == 2:
                ax.set_xlabel("Method", fontsize=8)

        fig.suptitle("Simulation benchmark: MSB-GP vs baselines across genetic architectures",
                     fontsize=13, y=1.0)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT_DIR, f"fig_simulation.{ext}"), dpi=600,
                        bbox_inches="tight")
        plt.close(fig)
        log(f"Figure saved (600 dpi PNG + PDF) to {OUT_DIR}")
    except Exception as e:
        log(f"Figure error: {e}")

    log("\nALL_DONE")
    logf.close()


if __name__ == "__main__":
    main()






