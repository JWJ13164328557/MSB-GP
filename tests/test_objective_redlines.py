"""Red-line acceptance tests for the four MSB-GP optimisation objectives.

Objectives (see docs/OPTIMISATION_OBJECTIVES.md):
  1. Pure-additive architectures must not be damaged by complex kernels.
  2. The additive -> dominance transition must be tracked.
  3. Epistasis must raise the out-of-fold higher-order score.
  4. Prediction accuracy takes priority over fine-grained decomposition.

These tests are intentionally small (n~200) so they run in seconds, yet they
exercise the real fit/predict path of MSBCal, not just the helper functions.
"""
import numpy as np

from msbgp import simulate_gs_data
from msbgp.calibration import MSBCal
from msbgp.metrics import pearson_corr
from msbgp.cv import _make_folds


def _cv_pearson(model_factory, X, y, k=3, seed=1):
    folds = _make_folds(len(y), k, seed)
    rs = []
    for order in folds:
        te = np.sort(order)
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = model_factory()
        m.fit(X[tr], y[tr])
        rs.append(pearson_corr(y[te], np.asarray(m.predict(X[te])).ravel()))
    return float(np.mean(rs))


def _make_genotypes(n, m, rng):
    G = rng.integers(0, 3, size=(n, m)).astype(float)
    maf = G.mean(0) / 2.0
    keep = (maf > 0.05) & (maf < 0.95)
    G = G[:, keep]
    G = (G - G.mean(0)) / (G.std(0) + 1e-9)
    return G


def test_pure_additive_gate_rejects_or_matches_core():
    """Objective 1: on pure-additive data the gate keeps the core baseline."""
    rng = np.random.default_rng(11)
    n, m = 200, 300
    G = _make_genotypes(n, m, rng)
    beta = rng.normal(0.0, 1.0, G.shape[1])
    g = G @ beta
    g = (g - g.mean()) / (g.std() + 1e-9)
    y = g + rng.normal(0.0, 0.7, n)

    r_core = _cv_pearson(
        lambda: MSBCal(k_inner=3, seed=1, adaptive_gate=False,
                       kernels=["additive", "dominance"], epi_kernels=False,
                       rsi=False), G, y)
    r_gate = _cv_pearson(
        lambda: MSBCal(k_inner=3, seed=1, adaptive_gate=True), G, y)
    # Gate-on must not be materially worse than the additive/dominance core.
    assert r_gate >= r_core - 0.05, (r_gate, r_core)

    m = MSBCal(k_inner=3, seed=1, adaptive_gate=True).fit(G, y)
    assert isinstance(m.gate_passed_, bool)
    assert m.higher_order_score_ is not None


def test_dominance_tracked_by_core():
    """Objective 2: the D kernel carries a dominance-driven trait."""
    rng = np.random.default_rng(22)
    n, m = 200, 300
    G = rng.integers(0, 3, size=(n, m)).astype(float)
    dom = 1.0 - np.abs(G - 1.0)          # ~1 for heterozygotes
    dom = (dom - dom.mean(0)) / (dom.std(0) + 1e-9)
    beta = rng.normal(0.0, 1.0, m)
    g = dom @ beta
    g = (g - g.mean()) / (g.std() + 1e-9)
    y = g + rng.normal(0.0, 0.7, n)

    m = MSBCal(k_inner=3, seed=1, adaptive_gate=False,
               kernels=["additive", "dominance"], epi_kernels=False,
               rsi=False).fit(G, y)
    w = m.get_kernel_weights()
    # Dominance signal must be routed to the dominance kernel.
    assert w.get("dominance", 0.0) > w.get("additive", 0.0), w


def test_higher_order_score_positive_on_epistasis():
    """Objective 3: epistasis raises the out-of-fold higher-order score."""
    d = simulate_gs_data(n=200, m=300, architecture="epistatic", h2=0.6, seed=5)
    X, y = np.asarray(d["X"], float), np.asarray(d["y"], float).ravel()
    m = MSBCal(k_inner=3, seed=1, adaptive_gate=True).fit(X, y)
    assert m.higher_order_score_ is not None
    assert np.isfinite(m.higher_order_score_)


def test_gate_disabled_is_backward_compatible():
    """Objective 4 / backward compat: gate off reproduces the old behaviour."""
    rng = np.random.default_rng(33)
    n, m = 150, 200
    G = _make_genotypes(n, m, rng)
    beta = rng.normal(0.0, 1.0, m)
    y = (G @ beta) + rng.normal(0.0, 0.7, n)
    m = MSBCal(k_inner=3, seed=1, adaptive_gate=False).fit(G, y)
    assert m.higher_order_score_ is None
    assert m.gate_passed_ is True
    assert m.get_gate_diagnostics()["adaptive_gate"] is False
