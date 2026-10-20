# -*- coding: utf-8 -*-
"""Capture MSB-GP adaptive kernel weights across the 9 simulation architectures.

Loads the exact saved datasets (simulation_datasets/*.npz, seed 2026) and
runs MSB-GP through the same 3-repeat x 5-fold CV used by the main simulation
benchmark, recording the learned kernel shares on every fold via the
``on_fold`` callback.

Output:
  simulation_kernel_weights.csv  (architecture, arch_label, repeat, fold,
                                     kernel, weight)
"""
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import simulation_engine as eng
from msbgp.calibration import MSBCal
from msbgp.cv import random_kfold

RES = eng.resolve_out_dir()
DATA_DIR = os.path.join(RES, "simulation_datasets")
OUT = os.path.join(RES, "simulation_kernel_weights.csv")
LOG = os.path.join(RES, "simulation_kernel_weights.log")

K = 5
REPEATS = 3
SEED = 2026


def main():
    logf = open(LOG, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    seeds = [SEED + rep * 100 for rep in range(REPEATS)]
    rows = []
    log("=" * 60)
    log("MSB-GP kernel weights across simulation architectures")
    log("=" * 60)
    for arch in eng.ARCHITECTURES:
        npz = os.path.join(DATA_DIR, f"{arch.replace(':','')}.npz")
        d = np.load(npz)
        X, y = d["X"], d["y"]
        log(f"\n{arch} ({eng.ARCH_LABELS[arch]})  X={X.shape}")
        t0 = time.time()
        for rep, seed in enumerate(seeds):
            kw_fold = {}

            def on_fold(fi, m, _rep=rep):
                if hasattr(m, "get_kernel_weights"):
                    try:
                        kw_fold[fi] = m.get_kernel_weights()
                    except Exception:
                        pass

            model = MSBCal(k_inner=5, seed=0)
            random_kfold(X.astype(float), y, model, k=K, seed=seed,
                         on_fold=on_fold)
            for fi, kw in kw_fold.items():
                for kname, wval in kw.items():
                    rows.append({
                        "architecture": arch,
                        "arch_label": eng.ARCH_LABELS[arch],
                        "repeat": rep,
                        "fold": fi,
                        "kernel": kname,
                        "weight": float(wval),
                    })
            msg = ", ".join(f"{k}={v:.3f}" for k, v in
                            sorted(kw_fold.get(0, {}).items()))
            log(f"  rep{rep} fold0: {msg}")
        log(f"  ({time.time()-t0:.1f}s)")
        pd.DataFrame(rows).to_csv(OUT, index=False)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    log(f"\nSaved -> {OUT}")
    log("ALL_DONE")
    logf.close()


if __name__ == "__main__":
    main()
