# -*- coding: utf-8 -*-
"""5x5-fold repeated CV benchmark on real data (all methods, calibrated MSB-GP).

Usage:  python run_5x5.py <Species>
Output: results/real_data_5x5_points_<Species>.csv   (fold-level, resumable)
        results/real_data_5x5_summary_<Species>.csv  (per-method aggregates)
"""
import os, sys, time, ctypes
from ctypes import wintypes

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_real_data_full", os.path.join(HERE, "run_real_data_full.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from msbgp.calibration import MSBCal
from msbgp.cv import _make_folds

N_REPEATS = 5          # 5 independent repetitions
K = 5                  # 5-fold
BASE_SEED = mod.BASE_SEED


class _Mem:
    def __init__(self):
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        self.PMC = PMC
        self.psapi = ctypes.WinDLL("psapi")
        self.k32 = ctypes.WinDLL("kernel32")
        self.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]

    def peak_mb(self):
        pmc = self.PMC(); pmc.cb = ctypes.sizeof(self.PMC)
        self.psapi.GetProcessMemoryInfo(self.k32.GetCurrentProcess(),
                                        ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 1e6


def regression_metrics(yt, yp):
    r = float(np.corrcoef(yt, yp)[0, 1])
    mse = float(np.mean((yt - yp) ** 2))
    return r, mse, float(np.sqrt(mse)), float(np.mean(np.abs(yt - yp)))


def roc_auc(yt, yp):
    order = np.argsort(yp)
    ranks = np.empty(len(yp)); ranks[order] = np.arange(1, len(yp) + 1)
    n1 = int(yt.sum()); n0 = len(yt) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[yt == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(yt, yp):
    order = np.argsort(-yp)
    yts = yt[order]
    tp = np.cumsum(yts); fp = np.cumsum(1 - yts)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(yt.sum()), 1)
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def bac_youden(yt, yp):
    thr = np.median(yp)
    pred = (yp > thr).astype(int)
    tp = int(((yt == 1) & (pred == 1)).sum())
    tn = int(((yt == 0) & (pred == 0)).sum())
    n1 = int((yt == 1).sum()); n0 = int((yt == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return 0.5 * (tp / n1 + tn / n0)


def build_methods_5x5():
    d = mod.build_methods()
    d["MSB-GP"] = MSBCal(k_inner=5, seed=0)   # published = calibrated
    return d


def main():
    sp = sys.argv[1]
    cfg = mod.SPECIES[sp]
    X, y = mod.prepare_species(sp, cfg)
    ybin = (y > float(np.median(y))).astype(int)
    n = X.shape[0]

    pts_path = os.path.join(ROOT, "results", f"real_data_5x5_points_{sp}.csv")
    done = set()
    if os.path.isfile(pts_path):
        old = pd.read_csv(pts_path)
        done = {(int(r.rep), int(r.fold), str(r.method)) for r in old.itertuples()}
        points = old.to_dict("records")
    else:
        points = []
    mem = _Mem()
    methods = build_methods_5x5()
    summary_rows = []
    for name, method in methods.items():
        t0 = time.time()
        for rep in range(N_REPEATS):
            seed = BASE_SEED + rep
            folds = _make_folds(n, K, seed)
            for fi, order in enumerate(folds):
                if (rep + 1, fi + 1, name) in done:
                    continue
                test = np.sort(order)
                train = np.setdiff1d(np.arange(n), test)
                method.fit(X[train], y[train])
                yp = np.asarray(method.predict(X[test])).ravel()
                yt = y[test]
                r_, mse_, rmse_, mae_ = regression_metrics(yt, yp)
                ytb = ybin[test]
                points.append({"species": sp, "method": name,
                               "rep": rep + 1, "fold": fi + 1,
                               "pearson": r_, "mse": mse_, "rmse": rmse_,
                               "mae": mae_, "auc": roc_auc(ytb, yp),
                               "auprc": auprc(ytb, yp),
                               "bac": bac_youden(ytb, yp)})
        elapsed = time.time() - t0
        peak = mem.peak_mb()
        dfm = pd.DataFrame([p for p in points if p["method"] == name])
        agg = dfm.groupby(["species", "method"])[
            ["pearson", "mse", "rmse", "mae", "auc", "auprc", "bac"]].agg(["mean", "std"])
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        agg = agg.reset_index()
        agg["runtime_s"] = round(elapsed, 1)
        agg["peak_mem_MB"] = round(peak, 1)
        summary_rows.append(agg)
        pd.DataFrame(points).to_csv(pts_path, index=False)
        print(f"[{sp}] {name}: {len(dfm)} fold-points, "
              f"wall={elapsed:.0f}s, peak={peak:.0f}MB", flush=True)

    out = pd.concat(summary_rows, ignore_index=True)
    out.to_csv(os.path.join(ROOT, "results", f"real_data_5x5_summary_{sp}.csv"),
               index=False)
    pd.DataFrame(points).to_csv(pts_path, index=False)
    print(f"5X5_{sp.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
