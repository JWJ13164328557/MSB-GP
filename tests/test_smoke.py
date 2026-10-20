import numpy as np
from msbgp import GBLUP, MSBCal, MSBGP, RRBLUP, RandomForest, simulate_gs_data


def _split(X, y, seed=0, test_frac=0.25):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_te = int(round(len(y) * test_frac))
    te = idx[:n_te]
    tr = idx[n_te:]
    return X[tr], X[te], y[tr], y[te]


def test_basic_models_fit_and_predict():
    d = simulate_gs_data(n=70, m=60, architecture="polygenic", h2=0.6, seed=1)
    X, y = d["X"], d["y"]
    X_tr, X_te, y_tr, y_te = _split(X, y)

    for model in [GBLUP(), RRBLUP(n_lambdas=5), MSBGP(), MSBCal(k_inner=2), RandomForest(n_estimators=10)]:
        pred = model.fit(X_tr, y_tr).predict(X_te)
        assert pred.shape == (len(y_te),)
        assert np.isfinite(pred).all()


def test_rsi_and_calibration():
    d = simulate_gs_data(n=90, m=120, architecture="epistatic", h2=0.65, seed=3)
    X, y = d["X"], d["y"]
    X_tr, X_te, y_tr, y_te = _split(X, y)

    model = MSBCal(k_inner=2).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    assert pred.shape == (len(y_te),)
    assert np.isfinite(pred).all()
    assert np.isfinite(model.a_) and np.isfinite(model.b_)

    weights = model.get_kernel_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-8

    arch_diag = model.get_architecture_diagnostics()
    assert {"kernel_weights", "higher_order_score", "gate", "sparse_epistasis"} <= set(arch_diag)
    sparse_diag = arch_diag["sparse_epistasis"]
    assert {"counts", "terms", "n_terms", "enabled", "selected"} <= set(sparse_diag)
    assert set(sparse_diag["counts"]) == {"AA", "AD", "DD"}
    # With strict stability selection RSI is optional: it is included only
    # when at least one interaction reaches the nominal frequency threshold.
    diagnostics = model.get_rsi_stability()
    assert "frequencies" in diagnostics
    if diagnostics["selected_terms"]:
        assert diagnostics["effective_threshold"] == diagnostics["nominal_threshold"]
        assert all(diagnostics["frequencies"][term] >= 0.6 for term in diagnostics["selected_terms"])

    additive = MSBGP(kernels=["additive"]).fit(X_tr, y_tr)
    assert "RSI" not in additive.get_kernel_weights()
    assert len(getattr(additive, "rsi_terms_", [])) == 0


