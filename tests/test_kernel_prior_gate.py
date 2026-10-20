"""Regression tests for the evidence-constrained nonlinear kernel gate.

The gate is MSB-GP default behaviour and keeps the kernel shares identified by
penalising the total mass assigned to non-core (non-additive, non-dominance)
kernels.  These tests lock in (a) the analytic gradient of the gate and (b) its
qualitative effect on a purely additive trait.
"""
import numpy as np

from msbgp import MSBGP, simulate_gs_data
from msbgp.kernels import KernelState


def _prepare_gate_model(X, y, strength=0.05):
    model = MSBGP(kernel_prior="nonlinear", kernel_prior_strength=strength,
                  kernel_correlation_threshold=None)
    model.kernel_state_ = KernelState().fit(X)
    Klist = model._make_kernels(X)
    model.kernel_names_ = [name for name, _ in Klist]
    model._kernel_prior_mode_ = "nonlinear"
    model._nonlinear_idx_ = tuple(
        i for i, nm in enumerate(model.kernel_names_)
        if str(nm).lower() not in {"additive", "dominance"})
    model._kappa_eff_ = strength * len(y)
    return model, Klist


def test_gate_gradient_matches_finite_difference():
    d = simulate_gs_data(n=60, m=40, architecture="polygenic", h2=0.7, seed=7)
    X, y = d["X"], d["y"]
    model, Klist = _prepare_gate_model(X, y, strength=0.05)
    assert model._nonlinear_idx_, "expected nonlinear kernels in the library"
    yc = y - y.mean()
    n_k = len(Klist)
    theta = np.concatenate([[0.0], 0.2 * np.ones(n_k - 1), [0.2]])
    _, gradient, _ = model._neg_reml(theta, Klist, yc)
    eps = 1e-5
    numeric = []
    for i in range(len(theta)):
        high, low = theta.copy(), theta.copy()
        high[i] += eps
        low[i] -= eps
        numeric.append((model._neg_reml(high, Klist, yc)[0]
                        - model._neg_reml(low, Klist, yc)[0]) / (2 * eps))
    assert np.max(np.abs(gradient - np.asarray(numeric))) < 1e-4


def test_gate_shrinks_nonlinear_mass_on_additive_data():
    d = simulate_gs_data(n=120, m=80, architecture="polygenic", h2=0.8, seed=11)
    X, y = d["X"], d["y"]

    def nl_mass(weights):
        return sum(v for k, v in weights.items()
                   if str(k).lower() not in {"additive", "dominance"})

    nogate = MSBGP(kernel_prior=None,
                   kernel_correlation_threshold=None).fit(X, y)
    gated = MSBGP(kernel_prior="nonlinear", kernel_prior_strength=0.05,
                  kernel_correlation_threshold=None).fit(X, y)

    w_nogate = nogate.get_kernel_weights()
    w_gated = gated.get_kernel_weights()
    assert abs(sum(w_gated.values()) - 1.0) < 1e-8
    # The gate must not increase the nonlinear mass, and should push the
    # additive share up on a purely additive trait.
    assert nl_mass(w_gated) <= nl_mass(w_nogate) + 1e-6
    assert w_gated.get("additive", 0.0) >= w_nogate.get("additive", 0.0) - 1e-6
