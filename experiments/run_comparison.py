"""Main benchmarking experiment: MSB-GP vs classical GS baselines."""
import os, time, sys

# Confine BLAS/LAPACK/OpenMP to a single thread BEFORE numpy/scipy import.
# The simulated marker matrices are small (n~200 x m~500); multi-threaded
# BLAS oversubscribes the cores and makes otherwise-fast small-matrix solves
# pathologically slow (minutes vs <1s).  Setting these before import makes
# the benchmark deterministic and fast regardless of how it is launched.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msbgp.data import simulate_gs_data
from msbgp.bases import GBLUP, RRBLUP, BayesA, BayesC
from msbgp.calibration import MSBCal
from msbgp.cv import random_kfold

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    architectures = ["polygenic", "oligogenic", "epistatic"]
    n_seeds = 3

    methods = {
        "GBLUP": GBLUP(),
        "RR-BLUP": RRBLUP(),
        "BayesA": BayesA(n_iter=400, burnin=120),
        "BayesCpi": BayesC(n_iter=400, burnin=120),
        "MSB-GP": MSBCal(k_inner=5, seed=0),
    }

    logf = open(os.path.join(OUT_DIR, "benchmark_progress.log"), "w", encoding="utf-8")
    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    rows = []
    for arch in architectures:
        log("=== Architecture: %s ===" % arch)
        per_method = {name: [] for name in methods}
        for seed in range(n_seeds):
            t0 = time.time()
            d = simulate_gs_data(n=220, m=500, architecture=arch, h2=0.5, seed=seed)
            for name, method in methods.items():
                mt0 = time.time()
                r = random_kfold(d["X"], d["y"], method, k=5, seed=(seed * 7 + 1) % 19)
                per_method[name].append(r["mean"])
                rows.append({"architecture": arch, "method": name, "seed": seed,
                             "accuracy": r["mean"], "std": r["std"]})
                log("  [%s] seed %d %-8s acc=%.4f (%.0fs)" % (arch, seed, name, r["mean"], time.time() - mt0))
            log("  seed %d done (%.0fs)" % (seed, time.time() - t0))
        log("  aggregate:")
        for name in methods:
            a = per_method[name]
            log("    %-8s: %.4f +- %.4f" % (name, np.mean(a), np.std(a)))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    log("\nSaved results -> %s" % csv_path)
    log("ALL_DONE")
    logf.close()
    _make_plot(df, os.path.join(OUT_DIR, "benchmark_barplot.png"))


def _make_plot(df, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arch_order = ["polygenic", "oligogenic", "epistatic"]
    method_order = ["GBLUP", "RR-BLUP", "BayesA", "BayesCpi", "MSB-GP"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, arch in zip(axes, arch_order):
        sub = df[df["architecture"] == arch]
        means, stds, labels = [], [], []
        for m in method_order:
            vals = sub[sub["method"] == m]["accuracy"]
            if len(vals):
                means.append(vals.mean()); stds.append(vals.std()); labels.append(m)
        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, capsize=4,
               color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(arch)
        if arch == "polygenic": ax.set_ylabel("Prediction accuracy (Pearson r)")
        ax.set_ylim(0, 1.0)
    fig.suptitle("Genomic selection: MSB-GP vs classical baselines (5-fold CV)")
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print("Saved figure -> %s" % path)


if __name__ == "__main__":
    main()

