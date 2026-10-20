# -*- coding: utf-8 -*-
"""Validate MSB-GP with in-model AA/AD/DD kernels on simulation P0-P8.

This script keeps the public method name as MSB-GP.  It uses the MSBGP core
(adaptive multi-kernel REML) with the AA/AD/DD product kernels enabled by
default, records every fold's learned kernel shares, and compares them with the
simulated architecture's own coefficient/variance composition.

Outputs are intentionally self-contained so the raw simulation data, truth
tables, recovered weights, fold-level metrics and summary tables can be copied
together for reproducibility.
"""
import argparse
import hashlib
import json
import os
import shutil
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
sys.path.insert(0, HERE)

from msbgp.bases import GBLUP
from msbgp.cv import _make_folds
from msbgp.metrics import pearson_corr
from msbgp.msbgp import MSBGP
from msbgp.calibration import MSBCal

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

ARCHS = [f"Pheno:{i}" for i in range(9)]
ARCH_LABELS = {
    "Pheno:0": "100% A",
    "Pheno:1": "75A/25D coefficients",
    "Pheno:2": "50A/50D coefficients",
    "Pheno:3": "25A/75D coefficients",
    "Pheno:4": "100% D",
    "Pheno:5": "A+D+AA-epi",
    "Pheno:6": "A+D+AD-epi",
    "Pheno:7": "A+D+DD-epi",
    "Pheno:8": "Mixed A+D+all epi",
}

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
TRUE_COMPONENTS = ["A", "D", "AA", "AD", "DD"]
RECOVERED_COMPONENTS = ["additive", "dominance", "gaussian", "poly", "AA", "AD", "DD"]


def _md5(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _arch_from_file(path):
    base = os.path.basename(path)
    return base.replace(".npz", "").replace("Pheno", "Pheno:")


def _true_tables():
    rows_coef = []
    rows_var = []
    for arch in ARCHS:
        coef = TRUE_COEF[arch]
        # Coefficient share is the literal mixing coefficient share.  The
        # simulation standardises each component before mixing, so the
        # statistically comparable target is the squared-coefficient share.
        abs_total = sum(abs(v) for v in coef.values())
        sq_total = sum(v * v for v in coef.values())
        for comp in TRUE_COMPONENTS:
            c = float(coef.get(comp, 0.0))
            rows_coef.append({
                "architecture": arch,
                "arch_label": ARCH_LABELS[arch],
                "component": comp,
                "coefficient": c,
                "coefficient_share": c / abs_total if abs_total else 0.0,
            })
            rows_var.append({
                "architecture": arch,
                "arch_label": ARCH_LABELS[arch],
                "component": comp,
                "variance_share": (c * c) / sq_total if sq_total else 0.0,
            })
    return pd.DataFrame(rows_coef), pd.DataFrame(rows_var)


def _load_or_make_dataset(arch, data_dir, n, m, h2, seed):
    path = os.path.join(data_dir, f"{arch.replace(':','')}.npz")
    if os.path.isfile(path):
        z = np.load(path)
        return z["X"].astype(float), z["y"].astype(float).ravel(), z["g"].astype(float).ravel(), path
    import simulation_engine as eng
    X, y, g = eng.simulate_architecture(arch, n=n, m=m, h2=h2, seed=seed)
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, g=g)
    return X.astype(float), y.astype(float).ravel(), g.astype(float).ravel(), path


def _spearman(a, b):
    if spearmanr is None:
        return float("nan")
    v = float(spearmanr(a, b).correlation)
    return v if np.isfinite(v) else float("nan")


def _fit_predict(method, X_tr, y_tr, X_te):
    if method == "GBLUP":
        mod = GBLUP()
    elif method == "MSB-GP":
        mod = MSBCal(epi_kernels=True, epi_top_markers=0,
                     kernel_prior="nonlinear", kernel_prior_strength=0.02)
    else:
        raise ValueError(method)
    mod.fit(X_tr, y_tr)
    pred = np.asarray(mod.predict(X_te)).ravel()
    return mod, pred


def _run_arch(arch, data_dir, out_dir, n, m, h2, seed, k, repeats,
              max_workers_arch=None):
    X, y, g, data_path = _load_or_make_dataset(arch, data_dir, n, m, h2, seed)
    shard_dir = os.path.join(out_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    metric_rows = []
    weight_rows = []
    t_arch = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for rep in range(int(repeats)):
        cv_seed = int(seed) + int(rep) * 100
        folds = _make_folds(len(y), int(k), cv_seed)
        for fi, order in enumerate(folds):
            test = np.sort(order)
            train = np.setdiff1d(np.arange(len(y)), test)
            for method in ("GBLUP", "MSB-GP"):
                t0 = time.time()
                mod, pred = _fit_predict(method, X[train], y[train], X[test])
                dt = time.time() - t0
                r = pearson_corr(y[test], pred)
                rho = _spearman(y[test], pred)
                rg = pearson_corr(g[test], pred)
                mse = float(np.mean((pred - y[test]) ** 2))
                metric_rows.append({
                    "architecture": arch,
                    "arch_label": ARCH_LABELS[arch],
                    "method": method,
                    "repeat": rep,
                    "fold": fi,
                    "cv_seed": cv_seed,
                    "n_train": int(train.size),
                    "n_test": int(test.size),
                    "pearson_phenotype": float(r),
                    "spearman_phenotype": float(rho),
                    "pearson_true_genetic": float(rg),
                    "mse_phenotype": mse,
                    "runtime_s": float(dt),
                })
                if method == "MSB-GP":
                    kw = mod.get_kernel_weights()
                    for kname, kval in kw.items():
                        weight_rows.append({
                            "architecture": arch,
                            "arch_label": ARCH_LABELS[arch],
                            "repeat": rep,
                            "fold": fi,
                            "cv_seed": cv_seed,
                            "kernel": kname,
                            "weight": float(kval),
                        })
    metrics = pd.DataFrame(metric_rows)
    weights = pd.DataFrame(weight_rows)
    metrics.to_csv(os.path.join(shard_dir, f"{arch.replace(':','')}_metrics.csv"), index=False)
    weights.to_csv(os.path.join(shard_dir, f"{arch.replace(':','')}_weights.csv"), index=False)
    return {
        "architecture": arch,
        "data_path": data_path,
        "n": int(X.shape[0]),
        "m": int(X.shape[1]),
        "seconds": float(time.time() - t_arch),
        "metric_rows": int(len(metrics)),
        "weight_rows": int(len(weights)),
    }


def _combine(out_dir, summaries, data_manifest_rows):
    shard_dir = os.path.join(out_dir, "_shards")
    mdfs, wdfs = [], []
    for s in summaries:
        arch = s["architecture"].replace(":", "")
        mp = os.path.join(shard_dir, f"{arch}_metrics.csv")
        wp = os.path.join(shard_dir, f"{arch}_weights.csv")
        if os.path.exists(mp):
            mdfs.append(pd.read_csv(mp))
        if os.path.exists(wp):
            wdfs.append(pd.read_csv(wp))
    metrics = pd.concat(mdfs, ignore_index=True) if mdfs else pd.DataFrame()
    weights = pd.concat(wdfs, ignore_index=True) if wdfs else pd.DataFrame()
    metrics.to_csv(os.path.join(out_dir, "simulation_epi_validation_folds.csv"), index=False)
    weights.to_csv(os.path.join(out_dir, "simulation_msbgp_kernel_weights_folds.csv"), index=False)

    if not metrics.empty:
        summary = (metrics.groupby(["architecture", "arch_label", "method"], as_index=False)
                   .agg(pearson_mean=("pearson_phenotype", "mean"),
                        pearson_sd=("pearson_phenotype", "std"),
                        spearman_mean=("spearman_phenotype", "mean"),
                        spearman_sd=("spearman_phenotype", "std"),
                        true_genetic_pearson_mean=("pearson_true_genetic", "mean"),
                        true_genetic_pearson_sd=("pearson_true_genetic", "std"),
                        mse_mean=("mse_phenotype", "mean"),
                        mse_sd=("mse_phenotype", "std"),
                        runtime_s_sum=("runtime_s", "sum")))
        summary.to_csv(os.path.join(out_dir, "simulation_epi_validation_summary.csv"), index=False)
    else:
        summary = pd.DataFrame()

    coef, var = _true_tables()
    coef.to_csv(os.path.join(out_dir, "simulation_true_kernel_coefficients_long.csv"), index=False)
    var.to_csv(os.path.join(out_dir, "simulation_true_kernel_variance_share_long.csv"), index=False)
    true_wide = (var.pivot_table(index=["architecture", "arch_label"],
                                 columns="component", values="variance_share",
                                 fill_value=0.0).reset_index())
    true_wide.to_csv(os.path.join(out_dir, "simulation_true_kernel_weights.csv"), index=False)

    if not weights.empty:
        rec = (weights.groupby(["architecture", "kernel"])["weight"]
               .agg(["mean", "std", "count"]).reset_index())
        rec_wide = rec.pivot_table(index="architecture", columns="kernel",
                                   values="mean", fill_value=0.0)
        rec_sd = rec.pivot_table(index="architecture", columns="kernel",
                                 values="std", fill_value=0.0)
        rec_wide = rec_wide.reindex(ARCHS).reindex(columns=RECOVERED_COMPONENTS, fill_value=0.0)
        rec_sd = rec_sd.reindex(ARCHS).reindex(columns=RECOVERED_COMPONENTS, fill_value=0.0)
        comp = true_wide.set_index("architecture")
        for col in TRUE_COMPONENTS:
            comp[f"true_{col}"] = comp.get(col, 0.0)
        for col in RECOVERED_COMPONENTS:
            comp[f"recovered_{col}_mean"] = rec_wide[col]
            comp[f"recovered_{col}_sd"] = rec_sd[col]
        comp["recovered_epi_mass"] = (rec_wide.get("AA", 0.0) + rec_wide.get("AD", 0.0)
                                      + rec_wide.get("DD", 0.0))
        comp["recovered_generic_nonlinear_mass"] = (rec_wide.get("gaussian", 0.0)
                                                    + rec_wide.get("poly", 0.0))
        # A compact match score over the five simulated components.  The
        # generic RBF/poly kernels are not simulated components, so they are not
        # assigned to A/D/AA/AD/DD here; they remain visible as "other" mass.
        comp["recovered_simulated_component_mass"] = (
            rec_wide.get("additive", 0.0) + rec_wide.get("dominance", 0.0)
            + rec_wide.get("AA", 0.0) + rec_wide.get("AD", 0.0)
            + rec_wide.get("DD", 0.0))
        comp["truth_recovered_l1_over_simulated_mass"] = (
            (comp["true_A"] - rec_wide.get("additive", 0.0)).abs()
            + (comp["true_D"] - rec_wide.get("dominance", 0.0)).abs()
            + (comp["true_AA"] - rec_wide.get("AA", 0.0)).abs()
            + (comp["true_AD"] - rec_wide.get("AD", 0.0)).abs()
            + (comp["true_DD"] - rec_wide.get("DD", 0.0)).abs())
        comp = comp.reset_index()
        comp.to_csv(os.path.join(out_dir, "simulation_kernel_weight_recovery.csv"), index=False)

    pd.DataFrame(data_manifest_rows).to_csv(os.path.join(out_dir, "simulation_data_manifest.csv"), index=False)
    with open(os.path.join(out_dir, "simulation_epi_validation_run.json"), "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "output_dir": os.path.abspath(out_dir)}, f,
                  ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "..", "..", "simulation_epi_validation"))
    ap.add_argument("--data-dir", default=(r"H:\现代农业研究院李博生老师课题组\AI育种大赛\results"
                                           r"\simulation_benchmark\simulation_datasets"))
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--h2", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    data_copy = os.path.join(out_dir, "simulation_datasets")
    os.makedirs(data_copy, exist_ok=True)
    print("=" * 72, flush=True)
    print("MSB-GP simulation validation with in-model AA/AD/DD kernels", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"data_dir={args.data_dir}", flush=True)
    print(f"protocol: {args.repeats} repeats x {args.k}-fold; MSB-GP (MSBCal, calibrated)", flush=True)
    print("=" * 72, flush=True)

    # Copy the raw simulation datasets before any result tables, so the exact
    # input data lives next to the weight-recovery and metric outputs.
    manifest_rows = []
    for arch in ARCHS:
        src = os.path.join(args.data_dir, f"{arch.replace(':','')}.npz")
        if os.path.isfile(src):
            dst = os.path.join(data_copy, os.path.basename(src))
            shutil.copy2(src, dst)
            z = np.load(src)
            manifest_rows.append({
                "architecture": arch,
                "arch_label": ARCH_LABELS[arch],
                "file": os.path.relpath(dst, out_dir).replace("\\", "/"),
                "source_file": src,
                "n": int(z["X"].shape[0]),
                "m": int(z["X"].shape[1]),
                "seed": int(args.seed),
                "h2": float(args.h2),
                "md5": _md5(dst),
            })
        else:
            X, y, g, made = _load_or_make_dataset(arch, data_copy, args.n, args.m, args.h2, args.seed)
            manifest_rows.append({
                "architecture": arch,
                "arch_label": ARCH_LABELS[arch],
                "file": os.path.relpath(made, out_dir).replace("\\", "/"),
                "source_file": "generated_in_place",
                "n": int(X.shape[0]),
                "m": int(X.shape[1]),
                "seed": int(args.seed),
                "h2": float(args.h2),
                "md5": _md5(made),
            })

    workers = max(1, min(int(args.workers), len(ARCHS)))
    summaries = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_arch, arch, data_copy, out_dir, args.n, args.m, args.h2,
                          args.seed, args.k, args.repeats): arch for arch in ARCHS}
        for fut in as_completed(futs):
            arch = futs[fut]
            try:
                s = fut.result()
                summaries.append(s)
                print(f"DONE {arch}: metrics={s['metric_rows']} weights={s['weight_rows']} "
                      f"in {s['seconds']:.1f}s", flush=True)
            except Exception as e:
                print(f"FAIL {arch}: {e!r}", flush=True)
                raise
    summaries = sorted(summaries, key=lambda x: ARCHS.index(x["architecture"]))
    _combine(out_dir, summaries, manifest_rows)
    print(f"\nALL_DONE in {time.time()-t0:.1f}s", flush=True)
    print(f"Outputs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
