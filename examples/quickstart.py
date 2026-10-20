"""Minimal MSB-GP / MSBCal quick-start example."""
import os
import sys

# Allow running this file directly from the package root
# ("python examples/quickstart.py") without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from msbgp import MSBCal, MSBGP, simulate_gs_data, pearson_corr


def _split(X, y, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_te = int(round(len(y) * test_frac))
    te, tr = idx[:n_te], idx[n_te:]
    return X[tr], X[te], y[tr], y[te]


def main():
    d = simulate_gs_data(n=180, m=400, architecture="epistatic", h2=0.55, seed=1)
    X, y = d["X"], d["y"]
    X_tr, X_te, y_tr, y_te = _split(X, y)

    model = MSBCal(k_inner=3).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("Pearson r:", pearson_corr(y_te, pred))
    print("Calibration a, b:", model.a_, model.b_)
    print("Kernel weights:", model.get_kernel_weights())

    # Uncalibrated multi-kernel model, useful for ablation/diagnostics.
    raw = MSBGP()
    raw.fit(X_tr, y_tr)
    pred_raw = raw.predict(X_te)
    print("Uncalibrated Pearson r:", pearson_corr(y_te, pred_raw))


if __name__ == "__main__":
    main()
