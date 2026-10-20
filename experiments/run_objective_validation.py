# -*- coding: utf-8 -*-
"""Validate the four MSB-GP optimisation objectives on simulation P0-P8.

Objectives (current iteration):
  1. Pure-additive architectures must not be damaged by complex kernels.  MUST
  2. The additive -> dominance transition must be tracked.               MUST
  3. Epistasis must produce a detectable increase in higher-order signal. MUST
  4. Prediction accuracy takes priority over fine-grained decomposition.

The harness compares, on the same folds:
  * GBLUP                (linear additive baseline)
  * MSB-GP-Core          (additive+dominance only, gate/calibration off kernelwise)
  * MSB-GP-Full          (full kernel library, higher-order gate disabled)
  * MSB-GP               (full library + prediction-first higher-order gate, NEW default)

For every MSB-GP variant we also store the learned kernel shares and the
out-of-fold higher-order gate score, so objective 3 can be read directly as
``higher_order_score`` growing with the true epistatic fraction.
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from msbgp.bases import GBLUP
from msbgp.cv import _make_folds
from msbgp.metrics import pearson_corr
from msbgp.calibration import MSBCal

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

ARCHS = [f"Pheno:{i}" for i in range(9)]
ARCH_LABELS = {
    "Pheno:0": "100% A", "Pheno:1": "75A/25D", "Pheno:2": "50A/50D",
    "Pheno:3": "25A/75D", "Pheno:4": "100% D",
    "Pheno:5": "A+D+AA-epi", "Pheno:6": "A+D+AD-epi",
    "Pheno:7": "A+D+DD-epi", "Pheno:8": "Mixed +epi",
}
# True variance (squared-coefficient) shares of A / D / epistasis.
TRUE_EPI = {"Pheno:0": 0.0, "Pheno:1": 0.0, "Pheno:2": 0.0,
            "Pheno:3": 0.0, "Pheno:4": 0.0,
            "Pheno:5": 0.3467, "Pheno:6": 0.3467, "Pheno:7": 0.3467,
            "Pheno:8": 0.6}
TRUE_A = {"Pheno:0": 1.0, "Pheno:1": 0.9, "Pheno:2": 0.5,
          "Pheno:3": 0.1, "Pheno:4": 0.0,
          "Pheno:5": 0.3266, "Pheno:6": 0.3266, "Pheno:7": 0.3266,
          "Pheno:8": 0.2}
TRUE_D = {"Pheno:0": 0.0, "Pheno:1": 0.1, "Pheno:2": 0.5,
          "Pheno:3": 0.9, "Pheno:4": 1.0,
          "Pheno:5": 0.3266, "Pheno:6": 0.3266, "Pheno:7": 0.3266,
          "Pheno:8": 0.2}

METHODS = ["GBLUP", "MSB-GP-Core", "MSB-GP-Full", "MSB-GP"]


def _make_model(method, k_inner, seed):
    if method == "GBLUP":
        return GBLUP()
    if method == "MSB-GP-Core":
        return MSBCal(k_inner=k_inner, seed=seed, adaptive_gate=False,
                      kernels=["additive", "dominance"],
                      epi_kernels=False, rsi=False)
    if method == "MSB-GP-Full":
        return MSBCal(k_inner=k_inner, seed=seed, adaptive_gate=False,
                      epi_kernels=True, epi_top_markers=0,
                      kernel_prior="nonlinear", kernel_prior_strength=0.02)
    if method == "MSB-GP":
        return MSBCal(k_inner=k_inner, seed=seed, adaptive_gate=True,
                      epi_kernels=True, epi_top_markers=0,
                      kernel_prior="nonlinear", kernel_prior_strength=0.02)
    raise ValueError(method)


def _volatile_weights(method):
    if method == "MSB-GP-Core":
        return {"additive", "dominance"}
    return None


def _run_arch(arch, data_dir, out_dir, seed, k, k_inner):
    z = np.load(os.path.join(data_dir, f"{arch.replace(':','')}.npz"))
    X = z["X"].astype(float)
    y = z["y"].astype(float).ravel()
    metric_rows, weight_rows = [], []
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    folds = _make_folds(len(y), k, seed)
    for fi, order in enumerate(folds):
        test = np.sort(order)
        train = np.setdiff1d(np.arange(len(y)), test)
        for method in METHODS:
            t0 = time.time()
            mod = _make_model(method, k_inner, seed + fi)
            mod.fit(X[train], y[train])
            pred = np.asarray(mod.predict(X[test])).ravel()
            r = pearson_corr(y[test], pred)
            rho = (float(spearmanr(y[test], pred).correlation)
                   if spearmanr is not None else float("nan"))
            mse = float(np.mean((pred - y[test]) ** 2))
            score = None
            passed = None
            if method in ("MSB-GP", "MSB-GP-Full"):
                score = getattr(mod, "higher_order_score_", None)
                passed = getattr(mod, "gate_passed_", None)
            metric_rows.append({
                "architecture": arch, "arch_label": ARCH_LABELS[arch],
                "method": method, "fold": fi,
                "pearson": float(r), "spearman": float(rho), "mse": mse,
                "higher_order_score": score, "gate_passed": passed,
                "runtime_s": float(time.time() - t0),
            })
            if method != "GBLUP":
                for kname, kval in mod.get_kernel_weights().items():
                    weight_rows.append({
                        "architecture": arch, "method": method, "fold": fi,
                        "kernel": kname, "weight": float(kval),
                    })
    pd.DataFrame(metric_rows).to_csv(
        os.path.join(out_dir, "_shards", f"{arch.replace(':','')}_metrics.csv"), index=False)
    pd.DataFrame(weight_rows).to_csv(
        os.path.join(out_dir, "_shards", f"{arch.replace(':','')}_weights.csv"), index=False)
    return arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "..", "..",
                                                     "simulation_objective_validation"))
    ap.add_argument("--data-dir", default=os.path.join(
        ROOT, "..", "..", "simulation_epi_validation", "simulation_datasets"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "_shards"), exist_ok=True)
    print("=" * 72, flush=True)
    print("MSB-GP four-objective validation (P0-P8)", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"protocol: {args.k}-fold, k_inner={args.k_inner}", flush=True)
    print("=" * 72, flush=True)
    t0 = time.time()
    workers = max(1, min(int(args.workers), len(ARCHS)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_arch, a, args.data_dir, out_dir, args.seed,
                          args.k, args.k_inner): a for a in ARCHS}
        for fut in as_completed(futs):
            try:
                print(f"DONE {fut.result()} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"FAIL {futs[fut]}: {e!r}", flush=True)
                raise
    # combine
    mdfs, wdfs = [], []
    for a in ARCHS:
        mp = os.path.join(out_dir, "_shards", f"{a.replace(':','')}_metrics.csv")
        wp = os.path.join(out_dir, "_shards", f"{a.replace(':','')}_weights.csv")
        if os.path.exists(mp):
            mdfs.append(pd.read_csv(mp))
        if os.path.exists(wp):
            wdfs.append(pd.read_csv(wp))
    metrics = pd.concat(mdfs, ignore_index=True)
    weights = pd.concat(wdfs, ignore_index=True)
    metrics.to_csv(os.path.join(out_dir, "objective_validation_folds.csv"), index=False)
    weights.to_csv(os.path.join(out_dir, "objective_validation_weights.csv"), index=False)

    summary = (metrics.groupby(["architecture", "arch_label", "method"], as_index=False)
               .agg(pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
                    spearman_mean=("spearman", "mean"),
                    mse_mean=("mse", "mean"),
                    ho_score_mean=("higher_order_score", "mean"),
                    gate_pass_rate=("gate_passed", "mean"),
                    runtime_s=("runtime_s", "sum")))
    summary["true_epi_share"] = summary["architecture"].map(TRUE_EPI)
    summary.to_csv(os.path.join(out_dir, "objective_validation_summary.csv"), index=False)

    wide = summary.pivot_table(index=["architecture", "arch_label"],
                               columns="method", values="pearson_mean").reset_index()
    wide.to_csv(os.path.join(out_dir, "objective_validation_pearson_wide.csv"), index=False)
    print(summary.to_string(index=False), flush=True)

    # objective 3: monotonicity of higher-order score vs true epistasis
    ho = summary[summary.method == "MSB-GP"][["architecture", "true_epi_share",
                                              "ho_score_mean"]].dropna()
    rho3 = float(spearmanr(ho["true_epi_share"], ho["ho_score_mean"]).correlation) \
        if (spearmanr is not None and len(ho) > 2) else float("nan")
    checks = {
        "obj1_pureA_gate_on_ge_core": bool(
            wide.loc[wide.architecture == "Pheno:0", "MSB-GP"].values[0] >=
            wide.loc[wide.architecture == "Pheno:0", "MSB-GP-Core"].values[0] - 0.02),
        "obj3_ho_score_spearman_vs_true_epi": rho3,
    }
    with open(os.path.join(out_dir, "objective_checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)
    print(json.dumps(checks, indent=2), flush=True)
    print(f"\nALL_DONE in {time.time()-t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
