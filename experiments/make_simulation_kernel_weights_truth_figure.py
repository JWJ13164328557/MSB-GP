# -*- coding: utf-8 -*-
"""Does MSB-GP recover the *true* kernel weights of each simulated architecture?

For every simulation architecture we know the ground-truth genetic composition
(how the phenotype was built from normalised A / D / AA / AD / DD components).
The true *variance share* of a component is its squared coefficient divided by
the sum of squared coefficients (the components are standardised to unit
variance before they are mixed, so the variance share is exactly c^2 / sum c^2).

MSB-GP's kernel library is {additive, dominance, gaussian(RBF), poly-2}.  It has
NO dedicated AA/AD/DD epistasis kernel, so any epistasis signal must either be
absorbed by the nonlinear kernels or remain unexplained.  This figure therefore
places, for each architecture, the *simulated truth* next to the *MSB-GP
recovered* kernel shares, and reports the additive/dominance agreement.

Inputs : simulation_kernel_weights.csv   (recovered, 3 rep x 5 fold)
Outputs: fig_simulation_kernel_weights_truth_vs_recovered.{png,pdf}
         simulation_true_kernel_weights.csv
         simulation_kernel_weight_recovery.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RES = (r"H:\现代农业研究院李博生老师课题组\AI育种大赛\results"
       r"\simulation_benchmark")
CSV = os.path.join(RES, "simulation_kernel_weights.csv")

# ---------------------------------------------------------------------------
# Ground truth: coefficients used in simulation_engine.simulate_architecture
# ---------------------------------------------------------------------------
# Raw coefficients on each standardised component, exactly as used in
# simulation_engine.simulate_architecture.  The variance share of a
# component is its squared coefficient divided by the sum of squared
# coefficients (all components are standardised to unit variance, so they are
# combined as c1*z1 + c2*z2 + ... with Var(z_i)=1).
TRUE_COEF = {
    "Pheno:0": {"A": 1.00},
    "Pheno:1": {"A": 0.75, "D": 0.25},
    "Pheno:2": {"A": 0.50, "D": 0.50},
    "Pheno:3": {"A": 0.25, "D": 0.75},
    "Pheno:4": {"D": 1.00},
    "Pheno:5": {"A": 0.33, "D": 0.33, "AA": 0.34},
    "Pheno:6": {"A": 0.33, "D": 0.33, "AD": 0.34},
    "Pheno:7": {"A": 0.33, "D": 0.33, "DD": 0.34},
    "Pheno:8": {"A": 0.20, "D": 0.20, "AA": 0.20, "AD": 0.20, "DD": 0.20},
}
ARCH_ORDER = [f"Pheno:{i}" for i in range(9)]
ARCH_TICK = {
    "Pheno:0": "P0\n100% A", "Pheno:1": "P1\n75A/25D",
    "Pheno:2": "P2\n50A/50D", "Pheno:3": "P3\n25A/75D",
    "Pheno:4": "P4\n100% D", "Pheno:5": "P5\nA+D+AA",
    "Pheno:6": "P6\nA+D+AD", "Pheno:7": "P7\nA+D+DD",
    "Pheno:8": "P8\nMixed",
}

# Recovered kernel palette (same as published real-data fig5)
K_COL = {"additive": "#2166AC", "dominance": "#1B7837",
         "gaussian": "#F1A340", "poly": "#762A83"}
# Truth palette (additive/dominance share colours with recovered panels)
T_ADD, T_DOM, T_EPI = "#2166AC", "#1B7837", "#BDBDBD"
RECOVERED = ["additive", "dominance", "gaussian", "poly"]


def true_shares(coef):
    sq = {k: v * v for k, v in coef.items()}
    s = sum(sq.values())
    sh = {k: v / s for k, v in sq.items()}
    return {
        "A": sh.get("A", 0.0),
        "D": sh.get("D", 0.0),
        "Epi": sh.get("AA", 0.0) + sh.get("AD", 0.0) + sh.get("DD", 0.0),
    }


def main():
    df = pd.read_csv(CSV)
    rec = (df.groupby(["architecture", "kernel"])["weight"].mean()
             .unstack(fill_value=0.0)
             .reindex(ARCH_ORDER)
             .reindex(columns=RECOVERED, fill_value=0.0))
    rsd = (df.groupby(["architecture", "kernel"])["weight"].std()
             .unstack(fill_value=0.0).reindex(ARCH_ORDER)
             .reindex(columns=RECOVERED, fill_value=0.0))

    truth = pd.DataFrame({a: true_shares(TRUE_COEF[a]) for a in ARCH_ORDER}).T
    truth = truth.reindex(columns=["A", "D", "Epi"], fill_value=0.0)

    # persisted tables -------------------------------------------------------
    truth.to_csv(os.path.join(RES, "simulation_true_kernel_weights.csv"))
    out = rec.copy()
    for k in RECOVERED:
        out[k + "_sd"] = rsd[k]
    out.to_csv(os.path.join(RES, "simulation_kernel_weight_recovery.csv"))

    # agreement stats on the A/D axis (P0-P4 pure A/D; also report all) ------
    rows = []
    for a in ARCH_ORDER:
        rows.append({
            "architecture": a,
            "true_A": truth.loc[a, "A"], "true_D": truth.loc[a, "D"],
            "true_Epi": truth.loc[a, "Epi"],
            "rec_additive": rec.loc[a, "additive"],
            "rec_dominance": rec.loc[a, "dominance"],
            "rec_gaussian": rec.loc[a, "gaussian"],
            "rec_poly": rec.loc[a, "poly"],
        })
    agree = pd.DataFrame(rows)
    ad_pure = agree[agree["true_Epi"] == 0]
    rA = np.corrcoef(ad_pure["true_A"], ad_pure["rec_additive"])[0, 1]
    rD = np.corrcoef(ad_pure["true_D"], ad_pure["rec_dominance"])[0, 1]
    maeA = np.mean(np.abs(ad_pure["true_A"] - ad_pure["rec_additive"]))
    maeD = np.mean(np.abs(ad_pure["true_D"] - ad_pure["rec_dominance"]))

    # figure -----------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(ARCH_ORDER))
    bw = 0.34

    for i, a in enumerate(ARCH_ORDER):
        # truth bar (left)
        base = 0.0
        for key, col, hatch in (("A", T_ADD, None), ("D", T_DOM, None),
                                ("Epi", T_EPI, "//")):
            v = float(truth.loc[a, key])
            if v > 0:
                ax.bar(x[i] - bw / 2 - 0.01, v, bw, bottom=base, color=col,
                       edgecolor="white", linewidth=0.5, hatch=hatch)
                if v >= 0.08:
                    ax.text(x[i] - bw / 2 - 0.01, base + v / 2, f"{v:.2f}",
                            ha="center", va="center", fontsize=6.3,
                            color="white" if key != "Epi" else "#404040",
                            fontweight="bold")
                base += v
        # recovered bar (right)
        base = 0.0
        for key in RECOVERED:
            v = float(rec.loc[a, key])
            if v > 0:
                ax.bar(x[i] + bw / 2 + 0.01, v, bw, bottom=base,
                       color=K_COL[key], edgecolor="white", linewidth=0.5)
                if v >= 0.08:
                    ax.text(x[i] + bw / 2 + 0.01, base + v / 2, f"{v:.2f}",
                            ha="center", va="center", fontsize=6.3,
                            color="white", fontweight="bold")
                base += v

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Weight share", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_TICK[a] for a in ARCH_ORDER], fontsize=7.4)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # barrier between each truth/recovered pair already implied by gaps

    handles = [
        Patch(facecolor=T_ADD, label="Simulated: additive (A)"),
        Patch(facecolor=T_DOM, label="Simulated: dominance (D)"),
        Patch(facecolor=T_EPI, hatch="//", edgecolor="white",
              label="Simulated: epistasis (AA/AD/DD)"),
        Patch(facecolor="#FFFFFF", edgecolor="none", label=" "),
        Patch(facecolor=K_COL["additive"], label="MSB-GP: additive"),
        Patch(facecolor=K_COL["dominance"], label="MSB-GP: dominance"),
        Patch(facecolor=K_COL["gaussian"], label="MSB-GP: Gaussian (RBF)"),
        Patch(facecolor=K_COL["poly"], label="MSB-GP: Poly-2"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=7.6, handlelength=1.3, labelspacing=0.6)

    # left bar = truth, right bar = MSB-GP, annotate once
    # per-pair orientation hint (left = truth, right = MSB-GP)
    ax.text(-0.62, 1.055, "left bar = Truth   |   right bar = MSB-GP",
            ha="left", va="center", fontsize=7.6, color="#B2182B",
            fontweight="bold")

    fig.text(0.012, 0.955,
             "Pure A/D architectures (P0-P4):  r(additive)=%.2f, "
             "r(dominance)=%.2f,  MAE=%.2f / %.2f" % (rA, rD, maeA, maeD),
             fontsize=7.6, color="#333333")

    out_png = os.path.join(RES, "fig_simulation_kernel_weights_truth_vs_recovered.png")
    out_pdf = os.path.join(RES, "fig_simulation_kernel_weights_truth_vs_recovered.pdf")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print("Saved:", out_png)
    print("Saved:", out_pdf)
    print("\nTrue shares:\n", truth.round(3).to_string())
    print("\nRecovered (mean):\n", rec.round(3).to_string())
    print("\nP0-P4 agreement: r_add=%.3f r_dom=%.3f MAE_add=%.3f MAE_dom=%.3f"
          % (rA, rD, maeA, maeD))


if __name__ == "__main__":
    main()



