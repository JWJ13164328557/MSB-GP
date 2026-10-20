# -*- coding: utf-8 -*-
"""Plot the simulation benchmark as two metric panels with independent y-axis limits."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = Path(r"H:\现代农业研究院李博生老师课题组\AI育种大赛\results\simulation_benchmark_stable")
df = pd.read_csv(p / "simulation_results.csv")
archs = df[["architecture", "arch_label"]].drop_duplicates().sort_values("architecture", ascending=False)
methods = [
    "GBLUP", "RR-BLUP", "BayesA", "BayesCpi", "MSB-GP", "MSB-GP-Add",
    "RandomForest", "DNNGP", "MeanGMKL", "FHGMKL", "CKAGMKL"
]
colors = {
    "GBLUP": "#AAAAAA",
    "RR-BLUP": "#CCCCCC",
    "BayesA": "#E15759",
    "BayesCpi": "#FF9D9A",
    "MSB-GP": "#B07AA1",
    "MSB-GP-Add": "#D4A6C8",
    "RandomForest": "#F28E2B",
    "DNNGP": "#4E79A7",
    "MeanGMKL": "#76B7B2",
    "FHGMKL": "#59A14F",
    "CKAGMKL": "#1F77B4",
}
metrics = [("pearson_mean", "pearson_std", "Pearson r"),
           ("spearman_mean", "spearman_std", "Spearman ρ")]

fig, axes = plt.subplots(9, 2, figsize=(10, 18), sharey=False, constrained_layout=True)
for col, (mean_col, std_col, metric_label) in enumerate(metrics):
    axes[0, col].set_title(metric_label, fontsize=11, fontweight="bold", pad=8)
    metric_max = (df[mean_col] + df[std_col]).max()
    for row, (_, arow) in enumerate(archs.iterrows()):
        ax = axes[row, col]
        arch = arow["architecture"]
        sub = df[df["architecture"] == arch].set_index("method")
        vals = sub.loc[methods, mean_col].values
        errs = sub.loc[methods, std_col].values
        x = np.arange(len(methods))
        ax.bar(x, vals, color=[colors[m] for m in methods], edgecolor="white",
               linewidth=.4, yerr=errs, error_kw=dict(elinewidth=.6, capsize=1.4))
        if col == 0:
            ax.set_ylabel(f"{arch}\n{arow['arch_label']}", rotation=0,
                          ha="right", va="center", labelpad=45, fontsize=7.5)
        ax.set_ylim(0, metric_max * 1.08)
        ax.grid(axis="y", color="#E6E6E6", linewidth=.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if row == len(archs) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=6)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])

fig.supxlabel("Method", fontsize=10)
fig.supylabel("Accuracy", fontsize=10)
fig.savefig(p / "fig_simulation_pearson_spearman_split.png", dpi=600, bbox_inches="tight")
fig.savefig(p / "fig_simulation_pearson_spearman_split.pdf", bbox_inches="tight")
print(f"Saved: {p / 'fig_simulation_pearson_spearman_split.png'}")
print(f"Saved: {p / 'fig_simulation_pearson_spearman_split.pdf'}")
