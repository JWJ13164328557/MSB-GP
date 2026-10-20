#!/usr/bin/env python3
"""run_new_species.py — Apply MSB-GP (calibrated) to a NEW species dataset.

Pure application tool: runs ONLY the final MSB-GP model with leakage-free
calibration on your data, reports results in the same column format as the
existing benchmark (`summary_all_methods.csv`), and produces a ranked
selection-candidate list.

Usage
-----
    python experiments/run_new_species.py --data-dir "H:/.../NewCrop"

Full control:

    python experiments/run_new_species.py ^
        --data-dir "H:/.../NewCrop" ^
        --species-name "MyCrop" ^
        --id-col "Sample" --coding "nucleotide" --trait-col "Trait_1" ^
        --max-markers 3000 --max-samples 500 ^
        --k 5 --repeats 1 --seed 1234 --top-percent 10 ^
        --output-dir "H:/.../results/msbgp_NewCrop"

Expected data folder:
    Genotypes.csv   rows = markers, cols = samples
    Phenotypes.csv  rows = records, cols = <ID-col>, Trait_1

Outputs (same format as existing benchmark):
    results_summary.csv        species,method,pearson_mean,pearson_std,mse_mean,...
    msbgp_predictions.csv      per-sample OOF true vs predicted
    selection_candidates.csv   ranked GEBV list with top-N% flagged
    msbgp_kernel_weights.csv   learned kernel shares
    fig_scatter.png/pdf        predicted vs observed
    fig_kernel_weights.png/pdf kernel decomposition
    fig_calibration.png/pdf    calibration diagnostics
    fig_selection.png/pdf      ranked selection candidates
    selection_report.txt       one-paragraph summary for breeders
"""

import os, sys, time, argparse

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from msbgp.calibration import MSBCal
from msbgp.cv import random_kfold
from msbgp.metrics import pearson_corr
from msbgp.data import (load_genotype_memmap, save_genotype_memmap,
                        validate_genotype_memmap)


def _is_numeric(v):
    try:
        float(v); return True
    except (ValueError, TypeError):
        return False


def load_genotypes(species_dir, coding, max_markers, max_samples, seed):
    gpath = os.path.join(species_dir, "Genotypes.csv")
    if not os.path.isfile(gpath):
        raise FileNotFoundError(f"Genotypes.csv not found in {species_dir}")
    print(f"  reading {gpath}", flush=True)
    raw = pd.read_csv(gpath, index_col=0, dtype=str)
    samples = list(raw.columns)
    if raw.shape[0] > 25000:
        rng = np.random.default_rng(seed)
        mk = rng.choice(raw.shape[0], size=25000, replace=False)
        raw = raw.iloc[mk]
        print(f"  pre-sampled to {raw.shape[0]} markers for load", flush=True)
    if coding == "numeric":
        X = raw.values.astype(np.float32).T
    else:
        vals = raw.values
        n_m, n_s = vals.shape
        X = np.zeros((n_m, n_s), dtype=np.float32)
        for i in range(n_m):
            row = vals[i]
            chset = set()
            for c in np.unique(row):
                chset.update([x for x in str(c)[:2] if x not in ("N", "")])
            chset = sorted(chset)
            if len(chset) < 2:
                if not chset: continue
                chset = [chset[0], chset[0]]
            b = chset[1]
            d = np.char.count(row.astype(str), b).astype(np.float32)
            bad = np.char.find(row.astype(str), "N") >= 0
            bad |= np.char.str_len(row.astype(str)) != 2
            d[bad] = np.nan
            X[i] = d
        X = X.T
        for mk in range(X.shape[1]):
            col = X[:, mk]
            valid = col[~np.isnan(col)]
            if valid.size:
                v, cnt = np.unique(valid, return_counts=True)
                mode = v[np.argmax(cnt)]
                mask = np.isnan(col)
                if mask.any():
                    col = col.copy(); col[mask] = mode; X[:, mk] = col
    n = X.shape[0]
    p = 0.5 * np.nansum(X, axis=0) / n
    maf = np.minimum(p, 1.0 - p)
    X = X[:, (maf >= 0.05) & (maf <= 0.5)]
    print(f"  after MAF>=0.05: {X.shape[1]} markers", flush=True)
    if max_markers and X.shape[1] > max_markers:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[1], size=max_markers, replace=False)
        X = X[:, idx]
        print(f"  subsampled to {X.shape[1]} markers", flush=True)
    if max_samples and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        choice = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[choice]
        samples = [samples[i] for i in choice]
        print(f"  subsampled to {X.shape[0]} samples", flush=True)
    return X.astype(np.float32), samples



def load_genotypes_memmap_matrix(memmap_path, sample_ids_path, phenotype_index,
                                 max_markers, max_samples, seed,
                                 save_memmap_path=None, dtype="int8"):
    """Load sample x marker genotypes from an exact int8/0-1-2 memmap."""
    X = load_genotype_memmap(memmap_path, mode="r")
    validate_genotype_memmap(X, require_integer=True, allow_missing=False,
                             name="genotype_memmap", verbose=True)
    if sample_ids_path:
        ids = pd.read_csv(sample_ids_path, header=None).iloc[:, 0].astype(str).tolist()
        if len(ids) != X.shape[0]:
            raise ValueError(
                f"--sample-ids has {len(ids)} rows but genotype memmap has "
                f"{X.shape[0]} samples")
        samples = ids
    elif phenotype_index is not None and len(phenotype_index) == X.shape[0]:
        samples = [str(v) for v in phenotype_index]
        print("  using Phenotypes.csv ID order as memmap sample IDs", flush=True)
    else:
        raise ValueError(
            "A genotype memmap needs sample IDs. Supply --sample-ids with one "
            "ID per row, or provide a Phenotypes.csv whose ID count matches "
            "the genotype row count.")

    n = int(X.shape[0])
    sums = np.zeros(int(X.shape[1]), dtype=np.float64)
    step = max(1, int(8_000_000 // max(1, X.shape[1])))
    for start in range(0, n, step):
        stop = min(n, start + step)
        sums += np.asarray(X[start:stop], dtype=np.float64).sum(axis=0)
    p = 0.5 * sums / max(1, n)
    maf = np.minimum(p, 1.0 - p)
    X = X[:, (maf >= 0.05) & (maf <= 0.5)]
    print(f"  after MAF>=0.05: {X.shape[1]} markers", flush=True)
    if max_markers and X.shape[1] > max_markers:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[1], size=max_markers, replace=False)
        X = X[:, idx]
        print(f"  subsampled to {X.shape[1]} markers", flush=True)
    if max_samples and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        choice = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[choice]
        samples = [samples[i] for i in choice]
        print(f"  subsampled to {X.shape[0]} samples", flush=True)
    if save_memmap_path:
        validate_genotype_memmap(X, require_integer=True, allow_missing=False,
                                 name="filtered_genotypes")
        save_genotype_memmap(X, save_memmap_path, dtype=np.dtype(dtype),
                             overwrite=True, validate=True, verbose=True)
        print(f"  saved filtered genotype memmap: {save_memmap_path}", flush=True)
    return X, samples


def load_phenotypes(species_dir, id_col, trait_col):
    ppath = os.path.join(species_dir, "Phenotypes.csv")
    if not os.path.isfile(ppath):
        raise FileNotFoundError(f"Phenotypes.csv not found in {species_dir}")
    ph = pd.read_csv(ppath)
    if id_col not in ph.columns:
        raise KeyError(f"ID column '{id_col}' not in Phenotypes.csv: {list(ph.columns)}")
    if trait_col not in ph.columns:
        raise KeyError(f"Trait column '{trait_col}' not in Phenotypes.csv: {list(ph.columns)}")
    return ph.groupby(id_col)[trait_col].mean().dropna()


def main():
    ap = argparse.ArgumentParser(
        description="Apply MSB-GP (calibrated) to a new species dataset.")
    ap.add_argument("--data-dir", required=True,
                    help="Folder containing Genotypes.csv + Phenotypes.csv")
    ap.add_argument("--species-name", default=None,
                    help="Display name (default = folder name)")
    ap.add_argument("--id-col", default=None,
                    help="Sample-ID column (default: auto-detect)")
    ap.add_argument("--coding", default=None,
                    choices=["numeric", "nucleotide"],
                    help="Genotype encoding (default: auto-detect)")
    ap.add_argument("--trait-col", default="Trait_1",
                    help="Phenotype trait column (default: Trait_1)")
    ap.add_argument("--max-markers", type=int, default=3000)
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--k", type=int, default=5, help="CV folds")
    ap.add_argument("--repeats", type=int, default=1, help="CV repeats")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--k-inner", type=int, default=5,
                    help="Inner CV folds for MSB-GP calibration")
    ap.add_argument("--top-percent", type=float, default=10,
                    help="Percentage of top individuals to flag as selection candidates")
    ap.add_argument("--genotype-memmap", default=None,
                    help="Optional .npy or .dat genotype memmap (samples x markers, 0/1/2 int8) instead of Genotypes.csv")
    ap.add_argument("--sample-ids", default=None,
                    help="CSV/text file with one sample ID per row for --genotype-memmap")
    ap.add_argument("--save-genotype-memmap", default=None,
                    help="Save the filtered genotype matrix as an int8 memmap")
    ap.add_argument("--genotype-dtype", default="int8",
                    help="dtype used by --save-genotype-memmap (default: int8)")
    ap.add_argument("--reml-init", default="single",
                    choices=["single", "heuristic", "multi_start", "multistart", "auto"],
                    help="REML starting-point strategy; default preserves the reproducible single-start path")
    ap.add_argument("--reml-multistart", type=int, default=1,
                    help="Number of deterministic REML starting points when multistart is requested")
    ap.add_argument("--reml-max-eval", type=int, default=None,
                    help="Optional max function evaluations per REML optimizer run")
    ap.add_argument("--reml-gtol", type=float, default=None,
                    help="Optional projected-gradient tolerance for L-BFGS-B")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    species = args.species_name or os.path.basename(os.path.normpath(args.data_dir))
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.data_dir)),
        "results", f"msbgp_{species}")
    os.makedirs(out_dir, exist_ok=True)

    logf = open(os.path.join(out_dir, "analysis.log"), "w", encoding="utf-8")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    # ---- auto-detect ----
    coding = args.coding
    if coding is None and args.genotype_memmap:
        coding = "numeric"
        log("  genotype memmap supplied; coding = numeric 0/1/2")
    elif coding is None:
        head = pd.read_csv(os.path.join(args.data_dir, "Genotypes.csv"),
                           nrows=5, index_col=0, dtype=str)
        coding = "numeric" if _is_numeric(head.values.flat[0]) else "nucleotide"
        log(f"  auto-detected coding = {coding}")
    id_col = args.id_col
    if id_col is None:
        ph_cols = pd.read_csv(os.path.join(args.data_dir, "Phenotypes.csv"),
                              nrows=0).columns.tolist()
        for cand in ("Sample", "Hybrid", "sample_id", "ID", "id", "GID"):
            if cand in ph_cols:
                id_col = cand; break
        if id_col is None:
            id_col = ph_cols[0]
        log(f"  auto-detected ID column = '{id_col}'")

    # ---- load ----
    log(f"\n=== MSB-GP analysis: {species} ===")
    pheno = load_phenotypes(args.data_dir, id_col, args.trait_col)
    if args.genotype_memmap:
        log(f"  reading genotype memmap {args.genotype_memmap}")
        X, samples = load_genotypes_memmap_matrix(
            args.genotype_memmap, args.sample_ids, pheno.index,
            args.max_markers, args.max_samples, args.seed,
            save_memmap_path=args.save_genotype_memmap,
            dtype=args.genotype_dtype)
    else:
        X, samples = load_genotypes(args.data_dir, coding,
                                    args.max_markers, args.max_samples,
                                    args.seed)
        if args.save_genotype_memmap:
            validate_genotype_memmap(X, require_integer=True, allow_missing=False,
                                     name="loaded_genotypes")
            save_genotype_memmap(X, args.save_genotype_memmap,
                                 dtype=np.dtype(args.genotype_dtype),
                                 overwrite=True, validate=True, verbose=True)
            log(f"  saved filtered genotype memmap: {args.save_genotype_memmap}")
    keep = [i for i, s in enumerate(samples) if s in pheno.index]
    if len(keep) < 20:
        raise RuntimeError(f"Only {len(keep)} samples with phenotype; need >=20.")
    X = X[keep]
    y = np.array([float(pheno[samples[i]]) for i in keep], dtype=float)
    kept_ids = [samples[i] for i in keep]
    log(f"  final dataset: {X.shape[0]} samples x {X.shape[1]} markers")

    # ---- run MSB-GP with calibration ----
    log(f"\n  Running MSB-GP (k_inner={args.k_inner}) ...")
    t0 = time.time()
    kern_acc = {}
    accs_all, oof_pool, true_pool, id_pool = [], [], [], []

    reml_kwargs = dict(reml_init=args.reml_init,
                      reml_multistart=args.reml_multistart,
                      reml_max_eval=args.reml_max_eval,
                      reml_gtol=args.reml_gtol)
    model = MSBCal(k_inner=args.k_inner, seed=0, **reml_kwargs)
    for rep in range(args.repeats):
        seed_rep = args.seed + rep * 100
        r = random_kfold(X, y, model, k=args.k, seed=seed_rep,
                         on_fold=lambda fi, m: kern_acc.update(
                             {fi: m.get_kernel_weights()})
                         if hasattr(m, "get_kernel_weights") else None)
        accs_all.extend(r["accuracy"])
        oof_pool = np.concatenate([oof_pool, r["pooled_pred"]])
        true_pool = np.concatenate([true_pool, r["pooled_true"]])
        id_pool.extend(kept_ids)

    runtime = time.time() - t0
    mean_acc = float(np.mean(accs_all))
    std_acc = float(np.std(accs_all))
    mse = float(np.mean((oof_pool - true_pool) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(oof_pool - true_pool)))
    r_p = pearson_corr(true_pool, oof_pool)

    log(f"  Pearson r  = {r_p:.4f} ± {std_acc:.4f}")
    log(f"  MSE        = {mse:.4f}")
    log(f"  RMSE       = {rmse:.4f}")
    log(f"  MAE        = {mae:.4f}")
    log(f"  Cal slope  = {model.a_:.4f}, intercept = {model.b_:.4f}")

    # ---- kernel weights ----
    if kern_acc:
        kw0 = next(v for v in kern_acc.values() if v)
        names_ = list(kw0.keys())
        wmat = np.array([[kern_acc[f][k] for k in names_]
                         for f in sorted(kern_acc) if kern_acc.get(f)],
                        dtype=float)
        log(f"\n  Kernel shares (mean ± std):")
        for k, m, s in zip(names_, wmat.mean(axis=0), wmat.std(axis=0)):
            log(f"    {k:>12s}: {m:.4f} ± {s:.4f}")

    # ================================================================ #
    # Save results in SAME column format as existing benchmark
    # ================================================================ #
    # summary (same columns as summary_all_methods.csv)
    summary_df = pd.DataFrame([{
        "species": species,
        "method": "MSB-GP",
        "pearson_mean": round(r_p, 4),
        "pearson_std": round(std_acc, 4),
        "mse_mean": round(mse, 4),
        "mse_std": round(float(np.std((oof_pool - true_pool) ** 2)), 4),
        "rmse_mean": round(rmse, 4),
        "rmse_std": round(float(np.std(np.sqrt((oof_pool - true_pool) ** 2))), 4),
        "mae_mean": round(mae, 4),
        "mae_std": round(float(np.std(np.abs(oof_pool - true_pool))), 4),
        "calibration_slope": round(model.a_, 4),
        "calibration_intercept": round(model.b_, 4),
        "runtime_s": round(runtime, 1),
    }])
    summary_df.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
    log(f"\n  results_summary.csv saved (same format as summary_all_methods.csv)")

    # predictions
    pred_df = pd.DataFrame({
        "sample_id": id_pool,
        "true": true_pool,
        "predicted_GEBV": oof_pool,
    })
    pred_df.to_csv(os.path.join(out_dir, "msbgp_predictions.csv"), index=False)

    # kernel weights
    kern_rows = []
    for fi, kw in kern_acc.items():
        if kw:
            for kname, w in kw.items():
                kern_rows.append({"fold": fi, "kernel": kname, "weight": round(w, 4)})
    if kern_rows:
        pd.DataFrame(kern_rows).to_csv(
            os.path.join(out_dir, "msbgp_kernel_weights.csv"), index=False)

    # ---- Fit final MSB-GP on ALL data for GEBV + marker effects ----
    log(f"\n  Fitting final MSB-GP on ALL data for GEBV estimation ...")
    final_model = MSBCal(k_inner=args.k_inner, seed=0, **reml_kwargs)
    final_model.fit(X, y)
    # ================================================================ #
    # Marker effects (beta coefficients for QTL discovery)
    # ================================================================ #
    log(f"\n  Computing marker effects ...")
    final_raw = final_model.model_  # underlying MSBGP (un-calibrated)
    marker_ids = None
    try:
        geno_df = pd.read_csv(os.path.join(args.data_dir, "Genotypes.csv"),
                              nrows=0, index_col=0)
        marker_ids = list(geno_df.index)
    except Exception:
        pass
    if marker_ids is None or len(marker_ids) != X.shape[1]:
        marker_ids = [f"Marker_{i+1}" for i in range(X.shape[1])]
    beta = final_raw.get_marker_effects(X)
    # rank by absolute effect
    abs_beta = np.abs(beta)
    beta_order = np.argsort(abs_beta)[::-1]
    marker_df = pd.DataFrame({
        "rank": range(1, X.shape[1] + 1),
        "marker_id": [marker_ids[i] for i in beta_order],
        "beta_effect": np.round(beta[beta_order], 6),
        "abs_effect": np.round(abs_beta[beta_order], 6),
    })
    marker_df.to_csv(os.path.join(out_dir, "marker_effects.csv"), index=False)
    log(f"    Top 10 markers by |effect|:")
    for i in range(min(10, X.shape[1])):
        log(f"      {i+1:>3d}. {marker_df.iloc[i]['marker_id']:>20s}  "
            f"beta={marker_df.iloc[i]['beta_effect']:.6f}")

    # ================================================================ #
    # Variance components (h2, per-kernel s2, residual)
    # ================================================================ #
    h2 = final_raw.get_model_heritability()
    vc_dict, s2e_val = final_raw.get_variance_components()
    total_var = sum(vc_dict.values()) + s2e_val
    vc_rows = [{"component": "additive_genetic_total", "variance": round(float(sum(vc_dict.values())), 6)},
               {"component": "residual", "variance": round(float(s2e_val), 6)},
               {"component": "heritability_h2", "variance": round(h2, 4)}]
    for kname, sval in vc_dict.items():
        vc_rows.append({"component": f"kernel_{kname}",
                        "variance": round(float(sval), 6),
                        "share": round(float(sval / total_var), 4) if total_var > 0 else 0})
    pd.DataFrame(vc_rows).to_csv(
        os.path.join(out_dir, "variance_components.csv"), index=False)
    log(f"    h2 = {h2:.4f}, s2g = {sum(vc_dict.values()):.4f}, s2e = {s2e_val:.4f}")

    # ================================================================ #
    # Per-individual reliability
    # ================================================================ #
    rel = final_raw.get_reliability()
    rel_df = pd.DataFrame({
        "sample_id": kept_ids,
        "GEBV": np.round(final_model.predict(X), 4),
        "reliability": np.round(rel, 4),
    })
    rel_df = rel_df.sort_values("GEBV", ascending=False).reset_index(drop=True)
    rel_df["rank"] = range(1, len(rel_df) + 1)
    rel_df.to_csv(os.path.join(out_dir, "individual_reliability.csv"), index=False)
    log(f"    reliability: mean={rel.mean():.4f}, range=[{rel.min():.4f}, {rel.max():.4f}]")

    # ================================================================ #
    # Selection candidates (ranked GEBV, top N% flagged)
    # ================================================================ #
    gebv_all = final_model.predict(X)

    n_top = max(1, int(round(X.shape[0] * args.top_percent / 100.0)))
    order = np.argsort(gebv_all)[::-1]

    sel_df = pd.DataFrame({
        "rank": range(1, X.shape[0] + 1),
        "sample_id": [kept_ids[i] for i in order],
        "GEBV": np.round(gebv_all[order], 4),
        "observed": np.round(y[order], 4),
        "selected": ["YES" if ri < n_top else "" for ri in range(X.shape[0])],
    })
    sel_df.to_csv(os.path.join(out_dir, "selection_candidates.csv"), index=False)

    top_ids = sel_df[sel_df["selected"] == "YES"]["sample_id"].tolist()
    log(f"  Top {args.top_percent}% selection candidates ({n_top} individuals):")
    for i, sid in enumerate(top_ids[:20], 1):
        log(f"    {i:>3d}. {sid}  GEBV={sel_df.iloc[i-1]['GEBV']:.4f}")
    if len(top_ids) > 20:
        log(f"    ... and {len(top_ids) - 20} more")

    # ================================================================ #
    # Figures
    # ================================================================ #
    log(f"\n  Generating figures ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

        # Fig 1: scatter (OOF predictions)
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.scatter(true_pool, oof_pool, s=12, alpha=0.5, color="#B07AA1",
                    edgecolors="none")
        lims = [min(true_pool.min(), oof_pool.min()),
                max(true_pool.max(), oof_pool.max())]
        ax1.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y = x")
        ax1.set_xlabel("Observed phenotype", fontsize=10)
        ax1.set_ylabel("Predicted GEBV (OOF)", fontsize=10)
        ax1.set_title(f"MSB-GP: {species}\n"
                      f"r = {r_p:.3f}, MSE = {mse:.3f}", fontsize=11)
        ax1.legend(fontsize=8, frameon=False)
        fig1.tight_layout()
        for ext in ("png", "pdf"):
            fig1.savefig(os.path.join(out_dir, f"fig_scatter.{ext}"), dpi=600)
        plt.close(fig1)

        # Fig 2: kernel weights
        if kern_rows:
            df_kw = pd.DataFrame(kern_rows)
            pivot = df_kw.groupby("kernel")["weight"].mean()
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            colors = ["#E69F00", "#56B4E9", "#009E73", "#D55E00"][:len(pivot)]
            ax2.bar(pivot.index, pivot.values, color=colors, edgecolor="white")
            for i, v in enumerate(pivot.values):
                ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9,
                         fontweight="bold")
            ax2.set_ylabel("Kernel share", fontsize=10)
            ax2.set_title(f"MSB-GP kernel decomposition: {species}", fontsize=11)
            ax2.set_ylim(0, min(1.0, pivot.max() + 0.15))
            plt.xticks(rotation=0)
            fig2.tight_layout()
            for ext in ("png", "pdf"):
                fig2.savefig(os.path.join(out_dir, f"fig_kernel_weights.{ext}"), dpi=600)
            plt.close(fig2)

        # Fig 3: calibration
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        ax3.scatter(true_pool, oof_pool, s=10, alpha=0.5, color="#B07AA1",
                    edgecolors="none", label="OOF predictions")
        ax3.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="ideal")
        z = np.polyfit(true_pool, oof_pool, 1)
        xs = np.linspace(lims[0], lims[1], 100)
        ax3.plot(xs, np.polyval(z, xs), color="#E15759", linewidth=1.2,
                 label=f"slope={z[0]:.2f}, int={z[1]:.2f}")
        ax3.set_xlabel("Observed", fontsize=10)
        ax3.set_ylabel("Predicted", fontsize=10)
        ax3.set_title(f"Calibration: {species}", fontsize=11)
        ax3.legend(fontsize=8, frameon=False)
        fig3.tight_layout()
        for ext in ("png", "pdf"):
            fig3.savefig(os.path.join(out_dir, f"fig_calibration.{ext}"), dpi=600)
        plt.close(fig3)

        # Fig 4: selection ranking
        fig4, ax4 = plt.subplots(figsize=(max(6, X.shape[0] * 0.06), 5))
        colors_sel = ["#E15759" if s == "YES" else "#D0D0D0"
                      for s in sel_df["selected"]]
        ax4.bar(range(X.shape[0]), sel_df["GEBV"], color=colors_sel,
                edgecolor="none")
        ax4.set_xlabel("Individual (ranked by GEBV)", fontsize=10)
        ax4.set_ylabel("GEBV", fontsize=10)
        ax4.set_title(f"Selection candidates (top {args.top_percent}% highlighted): {species}",
                      fontsize=11)
        # mark cutoff line
        if n_top < X.shape[0]:
            ax4.axvline(n_top - 0.5, color="black", linewidth=0.8, linestyle="--",
                        alpha=0.6, label=f"top {args.top_percent}% cutoff")
            ax4.legend(fontsize=8, frameon=False)
        fig4.tight_layout()
        for ext in ("png", "pdf"):
            fig4.savefig(os.path.join(out_dir, f"fig_selection.{ext}"), dpi=600)
        plt.close(fig4)

        # Fig 5: Manhattan plot of marker effects
        try:
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            abs_b = np.abs(beta)
            ax5.scatter(range(len(abs_b)), abs_b, s=3, alpha=0.5,
                       color="#0072B2", edgecolors="none")
            top20 = np.argsort(abs_b)[-20:]
            ax5.scatter(top20, abs_b[top20], s=12, color="#E15759",
                       edgecolors="none", zorder=5, label="top 20")
            ax5.set_xlabel("Marker index", fontsize=9)
            ax5.set_ylabel("|marker effect|", fontsize=9)
            ax5.set_title(f"Marker effects (Manhattan): {species}", fontsize=10)
            ax5.legend(fontsize=7, frameon=False)
            fig5.tight_layout()
            for ext in ("png", "pdf"):
                fig5.savefig(os.path.join(out_dir, f"fig_manhattan.{ext}"), dpi=600)
            plt.close(fig5)
        except Exception:
            pass
        log(f"  Figures saved (600 dpi PNG + PDF) to {out_dir}")
    except Exception as e:
        log(f"  Figure error: {e}")

    # ================================================================ #
    # Selection report
    # ================================================================ #
    report_lines = [
        f"=" * 60,
        f"  MSB-GP SELECTION REPORT",
        f"=" * 60,
        f"",
        f"  Species:        {species}",
        f"  Samples:        {X.shape[0]}",
        f"  Markers:        {X.shape[1]}",
        f"  CV scheme:      {args.k}-fold x {args.repeats} repeat(s)",
        f"",
        f"  --- Prediction accuracy ---",
        f"  Pearson r:      {r_p:.4f} ± {std_acc:.4f}",
        f"  MSE:            {mse:.4f}",
        f"  RMSE:           {rmse:.4f}",
        f"  MAE:            {mae:.4f}",
        f"  Calibration:    slope = {model.a_:.4f}, intercept = {model.b_:.4f}",
        f"",
        f"  --- Genetic architecture ---",
    ]
    if kern_rows:
        df_kw = pd.DataFrame(kern_rows)
        pivot = df_kw.groupby("kernel")["weight"].mean()
        dominant = pivot.idxmax()
        report_lines.append(f"  Dominant kernel:  {dominant} ({pivot.max():.1%})")
        if pivot.max() > 0.9:
            report_lines.append(f"  → Pure additive architecture; GBLUP would suffice.")
        elif pivot["additive"] < 0.5 if "additive" in pivot else False:
            report_lines.append(f"  → Substantial non-additive signal detected.")
        else:
            report_lines.append(f"  → Mixed architecture with dominant {dominant} component.")
    report_lines.extend([
        f"",
        f"  --- Selection candidates (top {args.top_percent}%, n={n_top}) ---",
    ])
    for i, sid in enumerate(top_ids, 1):
        report_lines.append(f"    {i:>3d}. {sid}")
    report_lines.extend([
        f"",
        f"  Full ranked list: selection_candidates.csv",
        f"  Predictions:      msbgp_predictions.csv",
        f"  Kernel details:   msbgp_kernel_weights.csv",
        f"=" * 60,
    ])
    with open(os.path.join(out_dir, "selection_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    log(f"\n  selection_report.txt saved")
    log(f"\n=== DONE. MSB-GP results in {out_dir} ===")
    logf.close()


if __name__ == "__main__":
    main()




