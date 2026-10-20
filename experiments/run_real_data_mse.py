# -*- coding: utf-8 -*-
"""Recompute genomic-prediction comparison with MSE / RMSE / MAE in addition
to Pearson r, using IDENTICAL data preparation, markers and fold seeds as
run_real_data_full.py so the existing Pearson r results are reproduced exactly.

Since the original benchmark only saved per-fold Pearson r (not the pooled
predictions), MSE cannot be derived from those CSVs -- it requires refitting.
This script refits the same models and records per-fold and pooled predictions.

Outputs (written next to the existing benchmark results):
    real_data_mse_points.csv  per (species, method, rep, fold) point metrics
    real_data_mse_results.csv per (species, method) aggregate (mean, sd)
"""
import os, sys, time, importlib.util

# Confine BLAS/LAPACK/OpenMP BEFORE numpy/scipy import
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from msbgp.cv import random_kfold
from msbgp.metrics import pearson_corr

# --- reuse identical prep / methods from the full benchmark module ---
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "run_real_data_full", os.path.join(HERE, "run_real_data_full.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SPECIES_ORDER = mod.SPECIES_ORDER
SPECIES = mod.SPECIES
BASE_SEED = mod.BASE_SEED
K = mod.K
N_REPEATS = mod.N_REPEATS
OUT_DIR = mod.OUT_DIR

def compute_point_metrics(yt, yp):
    # Pearson uses the identical definition as the benchmark (msbgp.metrics)
    from msbgp.metrics import pearson_corr
    r = pearson_corr(yt, yp)
    mse = float(np.mean((np.asarray(yt) - np.asarray(yp)) ** 2))
    return r, mse, float(np.sqrt(mse)), float(np.mean(np.abs(yt - yp)))

def main(only=None, resume=True):
    import importlib
    os.makedirs(OUT_DIR, exist_ok=True)
    points_f = os.path.join(OUT_DIR, "real_data_mse_points.csv")
    # load any already-computed points so we can resume after an interrupted run
    points = []
    if resume and os.path.isfile(points_f):
        try:
            old = pd.read_csv(points_f)
            points = old.to_dict("records")
        except Exception:
            points = []
    logf = open(os.path.join(HERE, "..", "results", "real_data_mse_progress.log"), "a", encoding="utf-8")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    species_sel = SPECIES_ORDER if only is None else [s for s in SPECIES_ORDER if s in only]
    for sp in species_sel:
        cfg = SPECIES[sp]
        t0 = time.time()
        X, y = mod.prepare_species(sp, cfg)
        methods = mod.build_methods()
        log(f"--- {sp}: {X.shape[0]} samples x {X.shape[1]} markers ---")
        for name, method in methods.items():
            if any(pt["species"] == sp and pt["method"] == name for pt in points):
                log(f"  {name:>13s}: SKIP (done) ---")
                continue
            mt0 = time.time()
            all_true, all_pred = [], []
            for rep in range(N_REPEATS):
                seed = BASE_SEED + rep
                r = random_kfold(X, y, method, k=K, seed=seed)
                all_true.append(np.asarray(r["pooled_true"]))
                all_pred.append(np.asarray(r["pooled_pred"]))
                # reconstruct per-fold test indices with the SAME fold split
                from msbgp.cv import _make_folds
                folds = _make_folds(len(y), K, seed)
                for fi, order in enumerate(folds):
                    test = np.sort(order)
                    yt = np.asarray(y)[test]
                    yp = all_pred[rep][test]
                    r_, mse_, rmse_, mae_ = compute_point_metrics(yt, yp)
                    points.append({"species": sp, "method": name, "rep": rep + 1,
                                   "fold": fi + 1, "pearson": r_,
                                   "mse": mse_, "rmse": rmse_, "mae": mae_})
            # pooled across all folds (full predictions)
            yt_cat = np.concatenate(all_true)
            yp_cat = np.concatenate(all_pred)
            pr_, pmse_, prmse_, pmae_ = compute_point_metrics(yt_cat, yp_cat)
            log(f"  {name:>13s}: pooled pearson={pr_:.4f} mse={pmse_:.5f} "
                f"rmse={prmse_:.4f} mae={pmae_:.4f}  ({time.time()-mt0:.0f}s)")
            # incremental save after each method so a crash loses little work
            pd.DataFrame(points).to_csv(points_f, index=False)
        log(f"--- {sp} done in {time.time()-t0:.0f}s\n")

        # aggregate results after each species
        pdf = pd.DataFrame(points)
        results = []
        for (sp2, name), g in pdf.groupby(["species", "method"]):
            results.append({"species": sp2, "method": name,
                            "pearson": round(float(g["pearson"].mean()), 4),
                            "pearson_std": round(float(g["pearson"].std()), 4),
                            "mse": round(float(g["mse"].mean()), 5),
                            "mse_std": round(float(g["mse"].std()), 5),
                            "rmse": round(float(g["rmse"].mean()), 4),
                            "rmse_std": round(float(g["rmse"].std()), 4),
                            "mae": round(float(g["mae"].mean()), 4),
                            "mae_std": round(float(g["mae"].std()), 4),
                            "n": int(len(g))})
        res = pd.DataFrame(results)
        res.to_csv(os.path.join(OUT_DIR, "real_data_mse_results.csv"), index=False)
        log(f"Saved incremental aggregates ({len(points)} points, {len(results)} method-species)")

    log("ALL_DONE")
    logf.close()
    print("\nDONE. points ->", points_f)

if __name__ == "__main__":
    argv = sys.argv[1:]
    only = None
    if argv and argv[0] == "--only":
        only = argv[1].split(",")
    main(only=only)
