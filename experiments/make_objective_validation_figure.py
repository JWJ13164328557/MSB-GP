# -*- coding: utf-8 -*-
"""Publication figure for the four-objective MSB-GP validation."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "..", "..", "simulation_objective_validation")
OUT = os.path.abspath(OUT)

summ = pd.read_csv(os.path.join(OUT, "objective_validation_summary.csv"))
wts = pd.read_csv(os.path.join(OUT, "objective_validation_weights.csv"))
ARCHS = [f"Pheno:{i}" for i in range(9)]
LAB = {a: l for a, l in summ[["architecture", "arch_label"]].drop_duplicates().values.tolist()}
METHODS = ["GBLUP", "MSB-GP-Core", "MSB-GP-Full", "MSB-GP"]
COL = {"GBLUP": "#4C72B0", "MSB-GP-Core": "#55A868",
       "MSB-GP-Full": "#C44E52", "MSB-GP": "#B07AA1"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120})

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# --- Panel A: Pearson r per architecture, grouped bars with SD -------------
ax = axes[0]
x = np.arange(len(ARCHS)); w = 0.2
for i, meth in enumerate(METHODS):
    sub = summ[summ.method == meth].set_index("architecture").reindex(ARCHS)
    ax.bar(x + (i - 1.5) * w, sub["pearson_mean"], w,
           yerr=sub["pearson_sd"], capsize=2, color=COL[meth], label=meth,
           error_kw=dict(lw=0.7))
ax.set_xticks(x)
ax.set_xticklabels([LAB[a].replace("/", "/\n") for a in ARCHS], rotation=45,
                   ha="right", fontsize=7.5)
ax.set_ylabel("Pearson r (5-fold)")
ax.set_title("A  Prediction accuracy (higher = better)", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.axvspan(4.5, 8.5, color="0.9", zorder=0)

# --- Panel B: higher-order score vs true epistasis -------------------------
ax = axes[1]
ho = summ[summ.method == "MSB-GP"].set_index("architecture").reindex(ARCHS)
true_epi = ho["true_epi_share"].values
score = ho["ho_score_mean"].values
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.plot(true_epi * 100, score, "o-", color="#B07AA1", ms=6, lw=1.2,
        label="MSB-GP higher-order score")
for a, xe, ys in zip(ARCHS, true_epi * 100, score):
    ax.annotate(a.replace("Pheno:", "P"), (xe, ys), fontsize=7,
                xytext=(3, 4), textcoords="offset points")
ax.set_xlabel("True epistatic variance share (%)")
ax.set_ylabel("r(full) - r(core)  (out-of-fold)")
ax.set_title("B  Higher-order signal detection", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8)

# --- Panel C: learned kernel weights (MSB-GP core vs epi) ------------------
ax = axes[2]
wk = wts[wts.method == "MSB-GP"]
core = ("additive", "dominance")
def group(name):
    if str(name).lower() in core: return "A + D (core)"
    if str(name).upper() in ("AA", "AD", "DD"): return "AA/AD/DD (epi)"
    return "Gaussian/Poly/RSI"
wk = wk.assign(grp=wk["kernel"].map(group))
gp = wk.groupby(["architecture", "grp"])["weight"].mean().unstack(fill_value=0.0).reindex(ARCHS)
bottom = np.zeros(len(ARCHS))
for grp, c in (("A + D (core)", "#55A868"), ("AA/AD/DD (epi)", "#C44E52"),
               ("Gaussian/Poly/RSI", "#9E9E9E")):
    if grp in gp:
        ax.bar(range(len(ARCHS)), gp[grp].values, bottom=bottom, color=c, label=grp)
        bottom += gp[grp].values
ax.set_xticks(range(len(ARCHS)))
ax.set_xticklabels([LAB[a].replace("/", "/\n") for a in ARCHS], rotation=45,
                   ha="right", fontsize=7.5)
ax.set_ylabel("Learned kernel share")
ax.set_title("C  Kernel decomposition (MSB-GP)", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right")

fig.tight_layout()
for ext, dpi in (("png", 600), ("pdf", 600)):
    fig.savefig(os.path.join(OUT, f"fig_objective_validation.{ext}"),
                dpi=dpi, bbox_inches="tight")
print("saved figures to", OUT)
