import numpy as np

from msbgp import MSBGP, simulate_gs_data


def test_gls_projection_and_optimizer_diagnostics():
    d = simulate_gs_data(n=60, m=30, architecture="polygenic",
                         h2=0.6, seed=11)
    model = MSBGP(rsi_stability_repeats=2).fit(d["X"], d["y"])
    assert model.Py_.shape == (model.n_,)
    # P is idempotent and annihilates the intercept design matrix.
    assert np.max(np.abs(model.Py_.sum())) < 1e-8
    assert np.isfinite(model.gls_intercept_)
    assert np.isfinite(model.optim_gradient_norm_)
    assert model.optim_niter_ >= 0
    assert isinstance(model.optim_boundary_parameters_, dict)


def test_reliability_uses_gls_projection():
    d = simulate_gs_data(n=45, m=25, architecture="polygenic",
                         h2=0.5, seed=12)
    model = MSBGP(rsi=False, rsi_stability_repeats=1).fit(d["X"], d["y"])
    rel = model.get_reliability()
    assert rel.shape == (model.n_,)
    assert np.isfinite(rel).all()
    assert np.all(rel >= 0.0) and np.all(rel <= 1.0)


def test_strict_rsi_stability_has_no_fallback():
    d = simulate_gs_data(n=80, m=80, architecture="epistatic",
                         h2=0.6, seed=13)
    model = MSBGP(rsi_stability_repeats=4,
                  rsi_stability_fraction=0.75).fit(d["X"], d["y"])
    diag = model.get_rsi_stability()
    assert diag["nominal_threshold"] == 3
    assert diag["effective_threshold"] == 3
    if diag["selected_terms"]:
        assert all(count >= 3 for count in diag["counts"].values())
    else:
        assert "RSI" not in model.get_kernel_weights()
