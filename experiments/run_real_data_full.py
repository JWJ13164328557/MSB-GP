import os, sys, time

# Confine BLAS/LAPACK/OpenMP BEFORE numpy/scipy import
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from msbgp.bases import GBLUP, RRBLUP, BayesA, BayesC
from msbgp.calibration import MSBCal
from msbgp.ml_baselines import RandomForest
from msbgp.cv import random_kfold

import ast as _ast
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "run_real_data.py"), encoding="utf-8").read()
_t = _ast.parse(_src)
DATA_DIR = None
for node in _t.body:
    if isinstance(node, _ast.Assign) and any(
            isinstance(x, _ast.Name) and x.id == "DATA_DIR" for x in node.targets):
        DATA_DIR = node.value.value
        break
assert DATA_DIR is not None

OUT_DIR = os.path.join(DATA_DIR, "results", "real_data_benchmark_full")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- full-data tuning ----
MAX_MARKERS = 12000     # final marker cap per species (up from 3000)
LOAD_MARKER_CAP = 25000  # random marker pre-sample during load (keeps coding loop fast)
MAF_MIN = 0.05
SPECIES_ORDER = ["Maize", "Rice", "Wheat", "Soybean"]
SPECIES = {
    "Maize":   dict(geno_id_col="Hybrid",  coding="nucleotide", sample_cap=None),
    "Rice":    dict(geno_id_col="Sample",  coding="numeric",    sample_cap=None),
    "Wheat":   dict(geno_id_col="Sample",  coding="nucleotide", sample_cap=None),
    "Soybean": dict(geno_id_col="Sample",  coding="nucleotide", sample_cap=4000),
}
METHOD_ORDER = ["GBLUP", "RR-BLUP", "BayesA", "BayesCpi", "MSB-GP",
                "MSB-GP-Add", "RandomForest"]
N_REPEATS = 2        # independent 5-fold repetitions -> 10 fold-accuracies each
BASE_SEED = 2026
K = 5


def load_genotypes(species_dir, coding):
    gpath = os.path.join(species_dir, "Genotypes.csv")
    print(f"    reading {os.path.basename(gpath)} ...", flush=True)
    raw = pd.read_csv(gpath, index_col=0, dtype=str)
    samples = list(raw.columns)
    if raw.shape[0] > LOAD_MARKER_CAP:
        rng = np.random.default_rng(1234)
        mk = rng.choice(raw.shape[0], size=LOAD_MARKER_CAP, replace=False)
        raw = raw.iloc[mk]
        print(f"    pre-sampled to {raw.shape[0]} markers for load", flush=True)
    if coding == "numeric":
        X = raw.values.astype(np.float32).T
        return X, samples
    # nucleotide coding (2-char codes, fixed-width |S2)
    vals = np.ascontiguousarray(raw.values.astype(bytes))
    n_m, n_s = vals.shape
    b8 = vals.view(np.uint8).reshape(n_m, n_s, 2)  # byte0, byte1 per cell
    X = np.zeros((n_m, n_s), dtype=np.float32)
    for i in range(n_m):
        row = vals[i]
        seen = set()
        for b0 in np.unique(row):
            c0 = chr(b0[0]) if b0 and b0[0] not in (78, 0) else ""
            c1 = chr(b0[1]) if len(b0) > 1 and b0[1] not in (78, 0) else ""
            seen.update([x for x in (c0, c1) if x])
        chset = sorted(seen)
        if len(chset) < 2:
            if not chset:
                continue
            chset = [chset[0], chset[0]]
        code = ord(chset[1])
        byte0 = b8[i, :, 0]
        byte1 = b8[i, :, 1]
        d = ((byte0 == code).astype(np.float32)
             + (byte1 == code).astype(np.float32))
        bad = (byte0 == 78) | (byte1 == 78) | (byte1 == 0)
        d[bad] = np.nan
        X[i] = d
    Xt = X.T
    for mk in range(Xt.shape[1]):
        col = Xt[:, mk]
        valid = col[~np.isnan(col)]
        if valid.size:
            vals_c, counts = np.unique(valid, return_counts=True)
            mode = vals_c[np.argmax(counts)]
            nm = np.isnan(col)
            if nm.any():
                col = col.copy(); col[nm] = mode; Xt[:, mk] = col
    return Xt.astype(np.float32), samples

def load_phenotypes(species_dir, id_col):
    ph = pd.read_csv(os.path.join(species_dir, "Phenotypes.csv"))
    mean = ph.groupby(id_col)["Trait_1"].mean()
    return mean.dropna()


def prepare_species(sp, cfg):
    print(f"\n=== {sp} ===", flush=True)
    species_dir = os.path.join(DATA_DIR, sp)
    X, samples = load_genotypes(species_dir, cfg["coding"])
    print(f"    loaded genotype: {X.shape[0]} samples x {X.shape[1]} markers", flush=True)
    pheno = load_phenotypes(species_dir, cfg["geno_id_col"])
    keep = [si for si in samples if si in pheno.index]
    X = X[np.array([samples.index(si) for si in keep])]
    y = np.array([float(pheno[si]) for si in keep], dtype=float)

    n = X.shape[0]
    p = 0.5 * np.nansum(X, axis=0) / n
    maf = np.minimum(p, 1.0 - p)
    X = X[:, (maf >= MAF_MIN) & (maf <= 0.5)]
    print(f"    after MAF>= {MAF_MIN}: {X.shape[1]} markers", flush=True)

    if X.shape[1] > MAX_MARKERS:
        rng = np.random.default_rng(1234)
        idx = rng.choice(X.shape[1], size=MAX_MARKERS, replace=False)
        X = X[:, idx]
        print(f"    marker-capped (shared across methods) to {X.shape[1]}", flush=True)
    else:
        print(f"    markers used = {X.shape[1]} (all)", flush=True)

    sc = cfg.get("sample_cap")
    if sc and n > sc:
        rng = np.random.default_rng(1234)
        choice = rng.choice(n, size=sc, replace=False)
        X = X[choice]; y = y[choice]
        print(f"    sample-capped to {X.shape[0]} (Soybean n^3 bound)", flush=True)
    else:
        print(f"    samples used = {X.shape[0]} (all)", flush=True)
    return X.copy(), y.copy()


def build_methods():
    return {
        "GBLUP": GBLUP(),
        "RR-BLUP": RRBLUP(),
        "BayesA": BayesA(n_iter=500, burnin=150),
        "BayesCpi": BayesC(n_iter=500, burnin=150),
        "MSB-GP": MSBCal(k_inner=5, seed=0),
        "MSB-GP-Add": MSBCal(kernels=["additive"], k_inner=5, seed=0),
        "RandomForest": RandomForest(n_estimators=120, max_depth=10,
                                     max_features="sqrt", seed=0),
    }


def paired_pvalue(a, b):
    from scipy.stats import ttest_rel
    return float(ttest_rel(a, b).pvalue)


def compact_letters(acc_by_method, alpha=0.05):
    from scipy.stats import ttest_rel
    methods = list(acc_by_method.keys())
    means = {m: float(np.mean(acc_by_method[m])) for m in methods}
    order = sorted(methods, key=lambda m: -means[m])
    letters = {m: "" for m in methods}
    nxt = 65
    for i, m in enumerate(order):
        new = ""
        for j in range(i):
            hi = order[j]
            p = ttest_rel(acc_by_method[m], acc_by_method[hi]).pvalue
            if p >= alpha:
                new += letters[hi]
        if not new:
            new = chr(nxt); nxt += 1
        letters[m] = new
    return letters


def main():
    logf = open(os.path.join(OUT_DIR, "real_data_full_progress.log"), "w",
                encoding="utf-8")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    rows = []
    fold_rows = []
    acc_by = {}
    kernw = {}
    failed = []

    def _dump_partial():
        # Write whatever has been produced so far so a later crash cannot
        # lose species/method results already computed.
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(OUT_DIR, "real_data_full_results.csv"), index=False)
        fr = []
        for (sp, name) in acc_by:
            for i, a in enumerate(acc_by[(sp, name)]):
                fr.append({"species": sp, "method": name,
                           "rep": i // K + 1, "fold": i % K + 1, "accuracy": a})
        if fr:
            pd.DataFrame(fr).to_csv(
                os.path.join(OUT_DIR, "real_data_full_folds.csv"), index=False)

    for sp, cfg in SPECIES.items():
        t0 = time.time()
        try:
            X, y = prepare_species(sp, cfg)
        except Exception as exc:
            log(f"  {sp} LOAD FAILED: {type(exc).__name__}: {exc}")
            failed.append((sp, "<load>", -1))
            continue
        methods = build_methods()
        log(f"--- {sp}: {X.shape[0]} samples x {X.shape[1]} markers, "
            f"{N_REPEATS}x{K}-fold ---")
        for name, method in methods.items():
            rep_acc = []
            for rep in range(N_REPEATS):
                seed = BASE_SEED + rep
                kw_fold = {}
                def on_fold(fi, m):
                    if hasattr(m, "get_kernel_weights") and rep == N_REPEATS - 1:
                        try:
                            kw_fold[fi] = m.get_kernel_weights()
                        except Exception:
                            kw_fold[fi] = None
                try:
                    r = random_kfold(X, y, method, k=K, seed=seed,
                                     on_fold=on_fold)
                except Exception as exc:
                    import traceback
                    log(f"  {name} rep{rep+1} FAILED: "
                        f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    failed.append((sp, name, rep + 1))
                    break
                rep_acc += list(r["accuracy"])
                if rep == N_REPEATS - 1 and kw_fold:
                    kernw[(sp, name)] = kw_fold
            if not rep_acc:
                log(f"  {name:>13s}: SKIPPED (no folds completed)")
                continue
            acc_by[(sp, name)] = rep_acc
            meanv = float(np.mean(rep_acc)); stdv = float(np.std(rep_acc))
            rows.append({"species": sp, "method": name,
                         "accuracy": round(meanv, 4), "std": round(stdv, 4),
                         "n_measures": len(rep_acc)})
            line = (f"  {name:>13s}: acc={meanv:.4f} +- {stdv:.4f} "
                    f"(n={len(rep_acc)})")
            if (sp, name) in kernw:
                kd = kernw[(sp, name)]
                names_ = list(next(v for v in kd.values() if v).keys())
                wmat = np.array([[kd[f][k] for k in names_]
                                 for f in sorted(kd)], float)
                line += "  | kernel shares: " + ", ".join(
                    f"{k}={wmat[:, i].mean():.3f}pm{wmat[:, i].std():.3f}"
                    for i, k in enumerate(names_))
            log(line)
        log(f"  {sp} done in {time.time()-t0:.0f}s\n")
        _dump_partial()

    for (sp, name) in acc_by:
        vals = acc_by[(sp, name)]
        for i, a in enumerate(vals):
            rep = i // K; fi = i % K
            fold_rows.append({"species": sp, "method": name, "rep": rep + 1,
                              "fold": fi + 1, "accuracy": a})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "real_data_full_results.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT_DIR, "real_data_full_folds.csv"),
                                   index=False)
    log("Saved results CSV + folds CSV")

    log("\n===== significance (paired t-test, n=10) =====")
    for sp in SPECIES_ORDER:
        if (sp, "MSB-GP") not in acc_by:
            log(f"  {sp}: MSB-GP missing (failed/skipped) -- skipped")
            continue
        a = acc_by[(sp, "MSB-GP")]
        cand = {m: acc_by[(sp, m)] for m in METHOD_ORDER
                if m != "MSB-GP" and (sp, m) in acc_by}
        best_m = max(cand, key=lambda m: float(np.mean(cand[m])))
        line = (f"  {sp}: MSB-GP={np.mean(a):.4f}pm{np.std(a):.4f}; "
                f"best={best_m}({np.mean(cand[best_m]):.4f}); "
                f"vs GBLUP p={paired_pvalue(a, acc_by[(sp, 'GBLUP')]):.4f}")
        for m in METHOD_ORDER:
            if m == "MSB-GP" or (sp, m) not in acc_by:
                continue
            line += f" | vs {m} p={paired_pvalue(a, acc_by[(sp, m)]):.3f}"
        log(line)
    log("")

    log("\n===== MSB-GP kernel-weight fold distribution (last repeat) =====")
    for (sp, name) in kernw:
        kd = kernw[(sp, name)]
        names_ = list(next(v for v in kd.values() if v).keys())
        log(f"  {sp}/{name}: " + ", ".join(names_))
        for fi in sorted(kd):
            log(f"      fold {fi}: " + ", ".join(f"{k}={kd[fi][k]:.3f}"
                for k in names_ if kd[fi]))
    log("")

    make_plots(df, acc_by)
    log("ALL_DONE")
    logf.close()


def make_plots(df, acc_by):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel

    ok_species = [sp for sp in SPECIES_ORDER
                  if all((sp, m) in acc_by for m in METHOD_ORDER)]
    if not ok_species:
        print("No complete species; skipping plots.")
        return
    letters = {sp: compact_letters({m: acc_by[(sp, m)] for m in METHOD_ORDER})
               for sp in ok_species}
    colors = {"Maize": "#E69F00", "Rice": "#56B4E9", "Wheat": "#009E73",
              "Soybean": "#D55E00"}

    fig, axes = plt.subplots(1, 2, figsize=(19, 7.2))
    ax = axes[0]
    x = np.arange(len(METHOD_ORDER)); width = 0.19
    for i, sp in enumerate(ok_species):
        means = [np.mean(acc_by[(sp, m)]) for m in METHOD_ORDER]
        sems = [np.std(acc_by[(sp, m)]) / np.sqrt(len(acc_by[(sp, m)]))
                for m in METHOD_ORDER]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=sems,
               capsize=3, color=colors[sp], label=sp, error_kw=dict(lw=1.1))
        for mi, m in enumerate(METHOD_ORDER):
            ax.text(x[mi] + (i - 1.5) * width, means[mi] + sems[mi] + 0.008,
                    letters[sp][m], ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Prediction accuracy (Pearson r)")
    ax.set_title("A. Full-data per-species comparison\n"
                 "(bars = SEM over 10 fold-measures; letters = paired t-test, a=0.05)")
    ax.legend(ncol=4, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ymax = max(np.mean(acc_by[(sp, m)]) + np.std(acc_by[(sp, m)]) /
               np.sqrt(len(acc_by[(sp, m)]))
               for sp in ok_species for m in METHOD_ORDER) + 0.09
    ax.set_ylim(0, max(ymax, 0.75))

    ax2 = axes[1]
    deltas = [np.mean(acc_by[(sp, "MSB-GP")]) - np.mean(acc_by[(sp, "GBLUP")])
              for sp in ok_species]
    ax2.bar(ok_species, deltas, color=[colors[s] for s in ok_species],
            alpha=0.85)
    ax2.axhline(0, color="k", lw=1)
    for i, sp in enumerate(ok_species):
        p = ttest_rel(acc_by[(sp, "MSB-GP")], acc_by[(sp, "GBLUP")]).pvalue
        ax2.text(i, deltas[i] + (0.003 if deltas[i] >= 0 else -0.010),
                 f"d={deltas[i]:+.3f}\np={p:.3f}" + ("*" if p < 0.05 else ""),
                 ha="center", va="bottom" if deltas[i] >= 0 else "top",
                 fontsize=9)
    ax2.set_ylabel("MSB-GP - GBLUP accuracy")
    ax2.set_title("B. MSB-GP vs GBLUP (paired, n=10)\n(* p<0.05)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(min(deltas) - 0.05, max(deltas) + 0.06)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "real_data_full_errorbars.png")
    fig.savefig(path, dpi=150)
    print("Saved figure -> %s" % path)


if __name__ == "__main__":
    main()

