import os, sys, time, ctypes, json, subprocess
from ctypes import wintypes

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
from msbgp.dnngp import DNNGP

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

OUT_DIR = os.path.join(DATA_DIR, "results", "real_data_metrics_benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_MARKERS = 12000
LOAD_MARKER_CAP = 25000
MAF_MIN = 0.05
SPECIES_ORDER = ["Maize", "Rice", "Wheat", "Soybean"]
SPECIES = {
    "Maize":   dict(geno_id_col="Hybrid",  coding="nucleotide", sample_cap=None),
    "Rice":    dict(geno_id_col="Sample",  coding="numeric",    sample_cap=None),
    "Wheat":   dict(geno_id_col="Sample",  coding="nucleotide", sample_cap=None),
    "Soybean": dict(geno_id_col="Sample",  coding="nucleotide", sample_cap=4000),
}
METHOD_ORDER = ["GBLUP", "RR-BLUP", "BayesA", "BayesCpi", "MSB-GP",
                "MSB-GP-Add", "RandomForest", "DNNGP"]
N_REPEATS = 2
BASE_SEED = 2026
K = 5


# ---- Windows peak working-set memory (includes numpy buffers) ----
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                                        wintypes.DWORD]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

def _current_working_set_mb():
    h = _k32.GetCurrentProcess()
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    _psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / 1e6


# ---- classification-style metrics ----
def roc_auc(ybin, score):
    from scipy.stats import rankdata
    ybin = np.asarray(ybin, dtype=bool)
    score = np.asarray(score, dtype=float)
    pos = score[ybin]; neg = score[~ybin]
    n1, n2 = float(len(pos)), float(len(neg))
    if n1 == 0 or n2 == 0:
        return np.nan
    joined = np.concatenate([pos, neg])
    ranks = rankdata(joined, method="average")
    U = ranks[:len(pos)].sum() - n1 * (n1 + 1) / 2.0
    return float(U / (n1 * n2))

def auprc(ybin, score):
    ybin = np.asarray(ybin, dtype=bool)
    score = np.asarray(score, dtype=float)
    n_pos = int(ybin.sum())
    if n_pos == 0 or n_pos == len(ybin):
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    yb = ybin[order]
    tp = np.cumsum(yb).astype(float)
    fp = np.cumsum(~yb).astype(float)
    prec = tp / (tp + fp)
    rec = tp / n_pos
    return float(np.trapezoid(prec, rec))

def balanced_accuracy_youden(ybin, score):
    ybin = np.asarray(ybin, dtype=bool)
    score = np.asarray(score, dtype=float)
    if ybin.sum() == 0 or (~ybin).sum() == 0:
        return np.nan
    n_pos = float(ybin.sum()); n_neg = float((~ybin).sum())
    best = 0.0
    for t in np.unique(score):
        pred = score >= t
        tp = (pred & ybin).sum(); tn = (~pred & ~ybin).sum()
        best = max(best, 0.5 * (tp / n_pos + tn / n_neg))
    return float(best)


# ---- data loading (matches run_real_data_full.py) ----
def load_genotypes(species_dir, coding):
    gpath = os.path.join(species_dir, "Genotypes.csv")
    print(f"    reading {os.path.basename(gpath)} ...", flush=True)
    raw = pd.read_csv(gpath, index_col=0, dtype=str)
    samples = list(raw.columns)
    if raw.shape[0] > LOAD_MARKER_CAP:
        rng = np.random.default_rng(1234)
        mk = rng.choice(raw.shape[0], size=LOAD_MARKER_CAP, replace=False)
        raw = raw.iloc[mk]
    if coding == "numeric":
        X = raw.values.astype(np.float32).T
        return X, samples
    vals = np.ascontiguousarray(raw.values.astype(bytes))
    n_m, n_s = vals.shape
    b8 = vals.view(np.uint8).reshape(n_m, n_s, 2)
    X = np.zeros((n_m, n_s), dtype=np.float32)
    for i in range(n_m):
        row = vals[i]; seen = set()
        for b0 in np.unique(row):
            c0 = chr(b0[0]) if b0 and b0[0] not in (78, 0) else ""
            c1 = chr(b0[1]) if len(b0) > 1 and b0[1] not in (78, 0) else ""
            seen.update([x for x in (c0, c1) if x])
        chset = sorted(seen)
        if len(chset) < 2:
            if not chset:
                continue
            chset = [chset[0], chset[0]]
        code = ord(chset[1]); byte0 = b8[i, :, 0]; byte1 = b8[i, :, 1]
        d = ((byte0 == code).astype(np.float32) + (byte1 == code).astype(np.float32))
        bad = (byte0 == 78) | (byte1 == 78) | (byte1 == 0)
        d[bad] = np.nan
        X[i] = d
    Xt = X.T
    for mk in range(Xt.shape[1]):
        col = Xt[:, mk]; valid = col[~np.isnan(col)]
        if valid.size:
            vc, ct = np.unique(valid, return_counts=True)
            mode = vc[np.argmax(ct)]; nm = np.isnan(col)
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
    pheno = load_phenotypes(species_dir, cfg["geno_id_col"])
    keep = [si for si in samples if si in pheno.index]
    X = X[np.array([samples.index(si) for si in keep])]
    y = np.array([float(pheno[si]) for si in keep], dtype=float)
    n = X.shape[0]
    p = 0.5 * np.nansum(X, axis=0) / n
    maf = np.minimum(p, 1.0 - p)
    X = X[:, (maf >= MAF_MIN) & (maf <= 0.5)]
    if X.shape[1] > MAX_MARKERS:
        rng = np.random.default_rng(1234)
        idx = rng.choice(X.shape[1], size=MAX_MARKERS, replace=False)
        X = X[:, idx]
    sc = cfg.get("sample_cap")
    if sc and n > sc:
        rng = np.random.default_rng(1234)
        choice = rng.choice(n, size=sc, replace=False)
        X = X[choice]; y = y[choice]
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
        "DNNGP": DNNGP(),
    }

def make_folds(n, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)

def paired_pvalue(a, b):
    from scipy.stats import ttest_rel
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(ttest_rel(a, b).pvalue)

def compact_letters(acc_by_method, alpha=0.05):
    from scipy.stats import ttest_rel
    methods = list(acc_by_method.keys())
    means = {m: float(np.mean(acc_by_method[m])) for m in methods}
    order = sorted(methods, key=lambda m: -means[m])
    letters = {m: "" for m in methods}
    nxt = 65
    for i, m in enumerate(order):
        newch = ""
        for j in range(i):
            hi = order[j]
            try:
                p = ttest_rel(acc_by_method[m], acc_by_method[hi]).pvalue
            except Exception:
                p = float("nan")
            if not np.isnan(p) and p >= alpha:
                newch += letters[hi]
        if not newch:
            newch = chr(nxt); nxt += 1
        letters[m] = newch
    return letters


def _run_method_worker(sp, name, work_dir):
    """Run one (species, method) in an isolated subprocess; return its dict.

    Per-method isolation fixes the memory bug where all methods shared one
    process, so later methods measured only a small delta above an already
    elevated baseline.  The worker reports PeakWorkingSetSize (true cumulative
    peak) of its own fresh process.
    """
    os.makedirs(work_dir, exist_ok=True)
    out_json = os.path.join(work_dir, (sp + "__" + name).replace("/", "_"))
    if os.path.isfile(out_json):
        with open(out_json, encoding="utf-8") as f:
            return json.load(f)
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "metric_worker.py")
    cmd = [sys.executable, worker, sp, name, out_json]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("worker failed for %s/%s: %s" % (
            sp, name, (r.stderr or "")[-2000:]))
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def main():
    logf = open(os.path.join(OUT_DIR, "real_data_metrics_progress.log"), "w",
                encoding="utf-8")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    rows = []; fold_rows = []
    met_by = {}      # (sp, method) -> {'bac':[...], 'auc':[...], 'auprc':[...]}
    res_by = {}      # (sp, method) -> {'runtime_s':.., 'peak_mem_mb':..}
    work_dir = os.path.join(OUT_DIR, "_worker_out")

    for sp in SPECIES_ORDER:
        log(f"=== {sp} ===")
        for name in METHOD_ORDER:
            t0 = time.time()
            rep = _run_method_worker(sp, name, work_dir)
            met = rep["met"]
            elapsed = float(rep["runtime"])
            peak = float(rep["peak_mem"])
            met_by[(sp, name)] = met
            res_by[(sp, name)] = {"runtime": elapsed, "peak_mem": peak}

            for m_ in ("bac", "auc", "auprc"):
                rows.append({"species": sp, "method": name, "metric": m_,
                             "mean": round(float(np.mean(met[m_])), 4),
                             "std": round(float(np.std(met[m_])), 4),
                             "n": len(met[m_]), "unit": "score"})
                for fr in rep["fold_rows"]:
                    fold_rows.append({"species": sp, "method": name,
                                      "metric": m_, "rep": fr["rep"],
                                      "fold": fr["fold"], "value": fr[m_]})
            rows.append({"species": sp, "method": name, "metric": "runtime",
                         "mean": round(elapsed, 2), "std": 0.0, "n": 1,
                         "unit": "s"})
            rows.append({"species": sp, "method": name, "metric": "peak_mem",
                         "mean": round(peak, 1), "std": 0.0, "n": 1,
                         "unit": "MB"})
            log(f"  {name:>13s}: BAC={np.mean(met['bac']):.4f} "
                f"AUC={np.mean(met['auc']):.4f} "
                f"AUPRC={np.mean(met['auprc']):.4f} "
                f"| {elapsed:.1f}s, {peak:.0f}MB "
                f"(wall {(time.time()-t0):.1f}s)")
        log(f"  {sp} done\n")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "real_data_metrics_results.csv"),
                              index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT_DIR, "real_data_metrics_folds.csv"),
                                   index=False)
    log("Saved metrics CSVs")

    log("\n===== significance: MSB-GP vs GBLUP per metric/species =====")
    for m_ in ("bac", "auc", "auprc"):
        for sp in SPECIES_ORDER:
            a = met_by[(sp, "MSB-GP")][m_]
            b = met_by[(sp, "GBLUP")][m_]
            log(f"  {sp}/{m_}: MSB-GP={np.mean(a):.4f}pm{np.std(a):.4f} "
                f"GBLUP={np.mean(b):.4f}pm{np.std(b):.4f} "
                f"p={paired_pvalue(a,b):.4f}")
    make_plots(rows, met_by, res_by)
    log("ALL_DONE")
    logf.close()


def make_plots(rows, met_by, res_by):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["bac", "auc", "auprc"]
    colors = {"Maize": "#E69F00", "Rice": "#56B4E9", "Wheat": "#009E73",
              "Soybean": "#D55E00"}

    def acc_by_metric(m):
        out = {}
        for sp in SPECIES_ORDER:
            for meth in METHOD_ORDER:
                out[(sp, meth)] = met_by[(sp, meth)][m]
        return out

    # ---------- Figure 1: BAC / AUC / AUPRC ----------
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.6))
    x = np.arange(len(METHOD_ORDER)); width = 0.19
    # (dynamic y-limits computed per panel below)
    for ai, m in enumerate(metrics):
        acc = acc_by_metric(m)
        letters = {sp: compact_letters({meth: acc[(sp, meth)]
                                        for meth in METHOD_ORDER})
                   for sp in SPECIES_ORDER}
        ax = axes[ai]
        ymax = 0
        for i, sp in enumerate(SPECIES_ORDER):
            means = [float(np.mean(acc[(sp, meth)])) for meth in METHOD_ORDER]
            sems = [float(np.std(acc[(sp, meth)])) / np.sqrt(len(acc[(sp, meth)]))
                    for meth in METHOD_ORDER]
            ax.bar(x + (i - 1.5) * width, means, width, yerr=sems, capsize=3,
                   color=colors[sp], label=sp, error_kw=dict(lw=1.1))
        # (lo/hi computed from all values below)
            for mi, meth in enumerate(METHOD_ORDER):
                ax.text(x[mi] + (i - 1.5) * width,
                        means[mi] + sems[mi] + 0.008, letters[sp][meth],
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
        ax.set_ylabel("value")
        ax.set_title({"bac": "A. Balanced Accuracy (BAC)",
                      "auc": "B. Area under ROC (AUC)",
                      "auprc": "C. Area under PR (AUPRC)"}[m] +
                     "\n(bars = SEM over 10 fold-measures; letters = paired t-test, a=0.05)")
        ax.legend(ncol=4, loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        lo = max(0.0, min([float(np.mean(acc[(sp, meth)])) for sp in SPECIES_ORDER for meth in METHOD_ORDER]) - 0.08)
        hi = min(1.0, max([float(np.mean(acc[(sp, meth)])) + float(np.std(acc[(sp, meth)])) / np.sqrt(len(acc[(sp, meth)])) for sp in SPECIES_ORDER for meth in METHOD_ORDER]) + 0.10)
        ax.set_ylim(lo, hi)
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"real_data_metrics_{fmt}.{fmt}"),
                    dpi=600, bbox_inches="tight")
    print("Saved metric figures", flush=True)

    # ---------- Figure 2: runtime + memory ----------
    fig2, axes2 = plt.subplots(1, 2, figsize=(19, 6.4))
    x = np.arange(len(METHOD_ORDER)); width = 0.19
    for ai, (metric, unit, ylab, title) in enumerate((
            ("runtime", "s", "Runtime (s)", "A. Runtime (fit+predict, full CV)"),
            ("peak_mem", "MB", "Peak memory (MB)", "B. Peak memory (full CV)"))):
        ax = axes2[ai]
        ymax = 0
        for i, sp in enumerate(SPECIES_ORDER):
            vals = [res_by[(sp, meth)][metric] for meth in METHOD_ORDER]
            ax.bar(x + (i - 1.5) * width, vals, width, color=colors[sp], label=sp)
            ymax = max(ymax, max(vals))
            for mi, v in enumerate(vals):
                ax.text(x[mi] + (i - 1.5) * width, v + ymax * 0.008, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(ncol=4, loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, ymax * 1.18 + 1e-9)
    fig2.tight_layout()
    for fmt in ("png", "pdf"):
        fig2.savefig(os.path.join(OUT_DIR, f"real_data_resources_{fmt}.{fmt}"),
                     dpi=600, bbox_inches="tight")
    print("Saved resource figures", flush=True)


if __name__ == "__main__":
    main()


