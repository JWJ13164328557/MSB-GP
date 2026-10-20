# -*- coding: utf-8 -*-
"""Real-data MSB-GP recovered kernel weights (gated model).

Parses the gated full-data run log and builds, for each species, the mean+-sd
softmax kernel shares of the final MSB-GP model (last repeat, captured per
fold).  Writes a tidy CSV and, once every species is present, a stacked-bar
figure in the published fig5 style (unified palette).

Input : real_data_benchmark_singleMSB_gated/real_data_full_progress.log
Outputs: real_data_benchmark_singleMSB_gated/real_data_kernel_weights.csv
         real_data_benchmark_singleMSB_gated/fig_real_data_kernel_weights.{png,pdf}
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = (r"H:\现代农业研究院李博生老师课题组\AI育种大赛\results")
GDIR = os.path.join(RES, "real_data_benchmark_singleMSB_gated")
LOG = os.path.join(GDIR, "real_data_full_progress.log")

SPECIES_ORDER = ["Maize", "Rice", "Wheat", "Soybean"]
KERNELS = ["additive", "dominance", "gaussian", "poly"]
K_COL = {"additive": "#2166AC", "dominance": "#1B7837",
         "gaussian": "#F1A340", "poly": "#762A83"}
K_LABEL = {"additive": "Additive", "dominance": "Dominance",
           "gaussian": "Gaussian (RBF)", "poly": "Poly-2"}

RE_SPECIES = re.compile(r"^(?:===|---)\s*(\w+)[:\s]")
RE_SHARES = re.compile(r"MSB-GP:\s*acc=.*?kernel shares:\s*(.*)$")


def parse():
    rows = []
    cur = None
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_SPECIES.match(line.strip())
            if m:
                cur = m.group(1)
            m2 = RE_SHARES.search(line)
            if m2 and cur:
                body = m2.group(1)
                for item in body.split(","):
                    item = item.strip()
                    mm = re.match(r"(\w+)=([\d.]+)pm([\d.]+)", item)
                    if mm:
                        rows.append({
                            "species": cur,
                            "kernel": mm.group(1),
                            "weight_mean": float(mm.group(2)),
                            "weight_std": float(mm.group(3)),
                        })
    return pd.DataFrame(rows)


def main():
    df = parse()
    if df.empty:
        print("No kernel-share lines parsed yet.")
        return
    df = df[df["kernel"].isin(KERNELS)]
    out_csv = os.path.join(GDIR, "real_data_kernel_weights.csv")
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)
    piv = (df.pivot_table(index="species", columns="kernel",
                          values="weight_mean")
             .reindex(SPECIES_ORDER).reindex(columns=KERNELS))
    print(piv.round(3).to_string())

    present = [s for s in SPECIES_ORDER if s in piv.index and
               piv.loc[s].notna().any()]
    if len(present) < len(SPECIES_ORDER):
        print(f"\nOnly {len(present)}/{len(SPECIES_ORDER)} species present "
              f"({present}); skipping figure until run completes.")
        return

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    x = np.arange(len(SPECIES_ORDER))
    bottom = np.zeros(len(SPECIES_ORDER))
    for k in KERNELS:
        vals = piv[k].values.astype(float)
        ax.bar(x, vals, 0.62, bottom=bottom, color=K_COL[k],
               edgecolor="white", linewidth=0.6, label=K_LABEL[k])
        for xi, v in enumerate(vals):
            if v >= 0.06:
                ax.text(xi, bottom[xi] + v / 2, f"{v:.2f}", ha="center",
                        va="center", fontsize=9, color="white",
                        fontweight="bold")
        bottom += vals
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Kernel weight (softmax share)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(SPECIES_ORDER, fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=10)
    out_png = os.path.join(GDIR, "fig_real_data_kernel_weights.png")
    out_pdf = os.path.join(GDIR, "fig_real_data_kernel_weights.pdf")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()

