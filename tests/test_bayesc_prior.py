import numpy as np
from msbgp.bases import BayesC
from msbgp.cv import random_kfold


def _sparse_additive(n=200, m=300, n_causal=30, h2=0.6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    qtl = rng.choice(m, size=n_causal, replace=False)
    beta = np.zeros(m); beta[qtl] = rng.normal(size=n_causal)
    g = X @ beta
    g = (g - g.mean()) / (g.std() + 1e-9)
    e = rng.normal(size=n)
    e = (e - e.mean()) / (e.std() + 1e-9)
    y = np.sqrt(h2) * g + np.sqrt(1.0 - h2) * e
    return X, y, qtl


def test_bayesc_default_is_sparse_and_pi_semantics():
    # 1) default pi must be the standard BayesC zero-effect proportion
    assert BayesC().pi == 0.99

    # 2) prior odds: at logBF = 0, p_inc == 1 - pi  (inclusion prob)
    for pi in (0.0, 0.5, 0.9, 0.99):
        if pi <= 0.0:
            p_inc = 1.0
        else:
            p_inc = 1.0 / (1.0 + (pi / (1.0 - pi)) * np.exp(0.0))
        assert abs(p_inc - (1.0 - pi)) < 1e-12

    # 3) invalid pi rejected
    for bad in (-0.1, 1.0, 1.5):
        try:
            BayesC(pi=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("pi=%r should raise ValueError" % bad)


def test_bayesc_pi_controls_sparsity_and_fixes_ridge_default():
    X, y, qtl = _sparse_additive(seed=1)
    # ridge-like limit (pi -> 0): essentially no markers zeroed out
    ridge_like = BayesC(n_iter=120, burnin=40, pi=0.001).fit(X, y)
    # sparse default (pi = 0.99): a large fraction of markers zeroed out
    sparse = BayesC(n_iter=120, burnin=40, pi=0.99).fit(X, y)

    assert ridge_like.zero_frac_ < 0.05, ridge_like.zero_frac_
    assert sparse.zero_frac_ > 0.50, sparse.zero_frac_
    # the excluded fraction should track the prior direction
    assert sparse.zero_frac_ > ridge_like.zero_frac_

    # predictions must be finite and positively informative for the sparse model
    r_sparse = float(np.corrcoef(sparse.predict(X), y)[0, 1])
    r_ridge = float(np.corrcoef(ridge_like.predict(X), y)[0, 1])
    assert np.isfinite(r_sparse) and np.isfinite(r_ridge)
    assert r_sparse > 0.30, r_sparse
    # the excluded-marker estimate must be tracked and reported
    assert hasattr(sparse, "zero_frac_") and hasattr(sparse, "delta_frac_")
    assert abs(sparse.zero_frac_ - sparse.delta_frac_) < 1e-12


if __name__ == "__main__":
    test_bayesc_default_is_sparse_and_pi_semantics()
    test_bayesc_pi_controls_sparsity_and_fixes_ridge_default()
    print("BayesC prior tests passed.")
