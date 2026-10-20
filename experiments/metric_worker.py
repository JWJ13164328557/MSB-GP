# -*- coding: utf-8 -*-
"""Isolated per-method worker for the real-data metrics benchmark."""
import os, sys, time, json, ctypes
from ctypes import wintypes

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # experiments/
_ROOT = os.path.dirname(_HERE)                          # project root (msbgp)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_benchmark_metrics as M
from run_benchmark_metrics import (SPECIES, N_REPEATS, BASE_SEED, K,
                                   prepare_species, build_methods, make_folds,
                                   roc_auc, auprc, balanced_accuracy_youden)


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


def peak_working_set_mb():
    h = _k32.GetCurrentProcess()
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    _psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 1e6


def main():
    sp = sys.argv[1]
    name = sys.argv[2]
    out_json = sys.argv[3]
    cfg = SPECIES[sp]

    X, y = prepare_species(sp, cfg)
    thr_label = float(np.median(y))
    ybin = (y > thr_label).astype(int)

    methods = build_methods()
    method = methods[name]

    t_start = time.time()
    met = {"bac": [], "auc": [], "auprc": []}
    fold_rows = []
    for rep in range(N_REPEATS):
        seed = BASE_SEED + rep
        folds = make_folds(X.shape[0], K, seed)
        for fi, test in enumerate(folds):
            test = np.sort(test)
            train = np.setdiff1d(np.arange(X.shape[0]), test)
            method.fit(X[train], y[train])
            yhat = np.asarray(method.predict(X[test])).ravel()
            yt = ybin[test]
            met["auc"].append(roc_auc(yt, yhat))
            met["auprc"].append(auprc(yt, yhat))
            met["bac"].append(balanced_accuracy_youden(yt, yhat))
            fold_rows.append({"rep": rep + 1, "fold": fi + 1,
                              "auc": met["auc"][-1], "auprc": met["auprc"][-1],
                              "bac": met["bac"][-1]})
    elapsed = time.time() - t_start
    peak = peak_working_set_mb()

    report = {
        "species": sp, "method": name,
        "runtime": elapsed, "peak_mem": peak,
        "met": {k: [float(v) for v in met[k]] for k in met},
        "fold_rows": fold_rows,
        "n_samples": int(X.shape[0]), "n_markers": int(X.shape[1]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f)
    print(json.dumps({"ok": True, "sp": sp, "method": name,
                      "runtime": round(elapsed, 2),
                      "peak_mem": round(peak, 1)}), flush=True)


if __name__ == "__main__":
    main()
