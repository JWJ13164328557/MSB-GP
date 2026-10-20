import gc
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse as sp

from msbgp import MSBGP, MSBCal, simulate_gs_data
from msbgp.data import (save_genotype_memmap, load_genotype_memmap,
                        validate_genotype_memmap, genotype_memmap_to_csr)
from msbgp.kernels import KernelState


def test_reml_gradient_matches_finite_difference():
    d = simulate_gs_data(n=32, m=25, architecture="polygenic", h2=0.6, seed=4)
    X, y = d["X"], d["y"]
    model = MSBGP(rsi=False, kernel_correlation_threshold=None)
    model.kernel_state_ = KernelState().fit(X)
    Klist = model._make_kernels(X)
    yc = y - y.mean()
    # theta = [log_v_total, w_1..w_{n_k-1}, log_s2e]; length depends on
    # the kernel library (the default library now includes AA/AD/DD).
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
    assert np.max(np.abs(gradient - np.asarray(numeric))) < 5e-5


def test_kernel_state_training_and_cross_shapes():
    d = simulate_gs_data(n=40, m=35, architecture="epistatic", h2=0.5, seed=5)
    X = d["X"]
    state = KernelState().fit(X[:-5])
    for name in ["additive", "dominance", "gaussian", "poly", "AA", "AD", "DD"]:
        Ktr = state.training_kernel(name)
        Kcross = state.cross_kernel(name, X[-5:])
        assert Ktr.shape == (35, 35)
        assert Kcross.shape == (35, 5)
        assert np.isfinite(Ktr).all() and np.isfinite(Kcross).all()
        assert abs(np.mean(np.diag(Ktr)) - 1.0) < 1e-8


def test_diagnostics_and_reliability():
    d = simulate_gs_data(n=70, m=45, architecture="polygenic", h2=0.6, seed=6)
    model = MSBGP(rsi_stability_repeats=2).fit(d["X"], d["y"])
    rel = model.get_reliability()
    weights = model.get_kernel_weights()
    assert rel.shape == (len(d["y"]),)
    assert abs(sum(weights.values()) - 1.0) < 1e-8
    assert all(np.isfinite(v) for v in model.get_kernel_correlations().values())
    assert 0.0 <= model.get_model_heritability() <= 1.0




def test_training_cross_kernel_consistency():
    """Training and cross kernels must use identical fitted state."""
    d = simulate_gs_data(n=45, m=30, architecture="epistatic", h2=0.55,
                         seed=17)
    X = d["X"]
    Xtr = X[:-6]
    Xva = Xtr[:6]
    state = KernelState().fit(Xtr)
    for name in ["additive", "dominance", "gaussian", "poly", "AA", "AD", "DD"]:
        Ktr = state.training_kernel(name)
        Kcross = state.cross_kernel(name, Xva)
        assert np.allclose(Kcross, Ktr[:, :6], rtol=1e-10, atol=1e-12), name


def test_shared_kernel_state_supports_model_and_calibration_variants():
    """Shared KernelState reuse must not change shape/validity of fits."""
    d = simulate_gs_data(n=60, m=40, architecture="polygenic", h2=0.55,
                         seed=23)
    X, y = d["X"], d["y"]
    state = MSBGP(rsi=False)._build_kernel_state(X, y)
    variants = [
        MSBGP(kernels=["additive"], rsi=False,
              kernel_correlation_threshold=None),
        MSBGP(kernels=["additive", "dominance"], rsi=False,
              kernel_correlation_threshold=None),
        MSBGP(kernels=["additive", "dominance", "sparse_additive"], rsi=False,
              kernel_correlation_threshold=None),
    ]
    preds = []
    for model in variants:
        model.fit(X, y, _kernel_state=state)
        pred = np.asarray(model.predict(X[:12]), dtype=float).ravel()
        assert pred.shape == (12,)
        assert np.isfinite(pred).all()
        assert model.kernel_state_ is state
        preds.append(pred)
    assert len(preds) == 3



def test_sparse_genotype_input_matches_dense():
    """Sparse CSR genotypes must follow the same numerical path as dense X."""
    d = simulate_gs_data(n=48, m=35, architecture="polygenic", h2=0.55,
                         seed=77)
    X, y = d["X"], d["y"]
    X_sp = sp.csr_matrix(X)
    variants = [
        ["additive"],
        ["additive", "dominance"],
        ["additive", "dominance", "gaussian", "poly"],
    ]
    for kernels in variants:
        dense = MSBGP(kernels=kernels, rsi=False, em_iters=40,
                      kernel_correlation_threshold=None,
                      kernel_prior=None, kernel_floor=None).fit(X, y)
        sparse = MSBGP(kernels=kernels, rsi=False, em_iters=40,
                       kernel_correlation_threshold=None,
                       kernel_prior=None, kernel_floor=None).fit(X_sp, y)
        pred_dense = dense.predict(X[:7])
        pred_sparse = sparse.predict(X_sp[:7])
        assert np.allclose(pred_sparse, pred_dense, rtol=1e-9, atol=1e-9)
        wd = dense.get_kernel_weights()
        ws = sparse.get_kernel_weights()
        for name in wd:
            assert abs(wd[name] - ws[name]) < 1e-9

def test_additive_only_msbgp_near_gblup():
    """With one additive kernel, MSBGP should reduce to an additive GBLUP fit."""
    d = simulate_gs_data(n=90, m=55, architecture="polygenic", h2=0.65,
                         seed=31)
    X, y = d["X"], d["y"]
    model = MSBGP(kernels=["additive"], rsi=False, em_iters=120,
                  kernel_correlation_threshold=None,
                  kernel_prior=None, kernel_floor=None).fit(X, y)
    pred = model.predict(X)
    K = model.kernel_state_.training_kernel("additive")
    V = (model.s2_[0] * K +
         model.s2e_ * np.eye(K.shape[0], dtype=float))
    V.flat[::K.shape[0] + 1] += 1e-8
    Vinv_y = np.linalg.solve(V, y)
    Vinv_1 = np.linalg.solve(V, np.ones_like(y))
    mu = float(np.ones_like(y) @ Vinv_y /
               (np.ones_like(y) @ Vinv_1))
    Py = Vinv_y - Vinv_1 * (np.ones_like(y) @ Vinv_y) / (np.ones_like(y) @ Vinv_1)
    gblup = mu + model.s2_[0] * K @ Py
    corr = float(np.corrcoef(pred, gblup)[0, 1])
    assert corr > 0.999

def test_cross_kernel_cache_matches_uncached_and_respects_input():
    """Cross-kernel caching must be exact and keyed by the validation input."""
    d = simulate_gs_data(n=54, m=36, architecture="epistatic", h2=0.5,
                         seed=41)
    X = d["X"]
    Xtr = X[:-8]
    Xva = np.asarray(X[-8:], dtype=float)
    state = KernelState().fit(Xtr)

    names = ["additive", "dominance", "poly", "AA", "AD", "DD"]
    weights = np.linspace(0.5, 1.0, len(names))
    uncached = state.weighted_cross_kernel(names, weights, Xva)

    state.enable_cross_kernel_cache()
    try:
        first = state.cross_kernel("additive", Xva)
        second = state.cross_kernel("additive", Xva)
        cached = state.weighted_cross_kernel(names, weights, Xva)
        Xva_other = Xva.copy()
        other = state.weighted_cross_kernel(names, weights, Xva_other)
    finally:
        state.clear_cross_kernel_cache()

    assert first is second
    assert np.allclose(cached, uncached, rtol=1e-12, atol=1e-12)
    assert np.allclose(other, uncached, rtol=1e-12, atol=1e-12)
    assert abs(float(np.max(np.abs(other - uncached)))) < 1e-12

def test_candidate_stack_prediction_uses_shared_cross_cache_path():
    """Forced candidate stack must fit and predict through the cached path."""
    from msbgp import MSBCal

    d = simulate_gs_data(n=45, m=30, architecture="polygenic", h2=0.55,
                         seed=123)
    X, y = d["X"], d["y"]
    model = MSBCal(
        kernels=["additive", "dominance"],
        rsi=False,
        adaptive_gate=True,
        candidate_stack=True,
        stack_min_gain=-1.0,
        n_inner_repeats=1,
        n_ensemble=1,
        auto_sparse_additive=False,
        k_inner=3,
        seed=123,
    ).fit(X, y)
    assert model.candidate_stack_enabled_ is True
    pred = np.asarray(model.predict(X[:7]), dtype=float).ravel()
    assert pred.shape == (7,)
    assert np.isfinite(pred).all()
    for models in model.candidate_models_.values():
        assert models
        states = [getattr(m, "kernel_state_", None) for m in models]
        assert all(state is states[0] for state in states)



def test_native_msbcal_default_has_no_external_model_calls():
    """Default MSB-GP optimization must remain native multi-kernel REML."""
    d = simulate_gs_data(n=70, m=45, architecture="polygenic", h2=0.6, seed=211)
    model = MSBCal(
        kernels="auto",
        rsi=False,
        auto_sparse_additive=False,
        k_inner=3,
        n_inner_repeats=1,
        n_ensemble=1,
        seed=211,
        candidate_stack=True,
        stack_min_gain=-1.0,
    ).fit(d["X"], d["y"])
    assert model.aux_candidate_names_ == []
    names = set(getattr(model, "candidate_stack_names_", []))
    assert names.issubset({"core", "sparse", "epi", "full"})
    assert not names.intersection({"gblup", "rr_blup", "mean_gmkl", "fh_gmkl", "cka_gmkl"})


def test_native_epi_candidate_evaluated_on_auto_kernels():
    """Auto MSB-GP evaluates the native explicit AA/AD/DD candidate."""
    d = simulate_gs_data(n=90, m=50, architecture="epistatic", h2=0.55, seed=212)
    model = MSBCal(
        kernels="auto",
        rsi=False,
        auto_sparse_additive=False,
        k_inner=3,
        n_inner_repeats=1,
        n_ensemble=1,
        seed=212,
        aux_candidates=(),
    ).fit(d["X"], d["y"])
    assert model.aux_candidate_names_ == []
    assert model.epi_candidate_score_ is not None
    assert np.isfinite(model.epi_candidate_score_)

def test_memmap_genotype_input_matches_dense():
    """Int8 memmap genotypes must match the dense numerical path exactly."""
    d = simulate_gs_data(n=52, m=38, architecture="epistatic", h2=0.55,
                         seed=91)
    X, y = d["X"], d["y"]
    td = Path(__file__).resolve().parent / "_tmp_memmap_dense"
    shutil.rmtree(td, ignore_errors=True)
    td.mkdir(parents=True, exist_ok=True)
    try:
        path = td / "X.dat"
        Xm = save_genotype_memmap(X, path, dtype=np.int8, overwrite=True,
                                  validate=True)
        assert str(Xm.dtype) == "int8"
        validate_genotype_memmap(Xm, require_integer=True, allow_missing=False)
        models = [
            MSBGP(kernels=["additive", "dominance"], rsi=False, em_iters=40,
                  kernel_correlation_threshold=None,
                  kernel_prior=None, kernel_floor=None),
            MSBGP(kernels=["additive", "dominance", "gaussian", "poly"],
                  rsi=False, em_iters=40, kernel_correlation_threshold=None,
                  kernel_prior=None, kernel_floor=None),
        ]
        for dense_model, mem_model in zip(models, [
                MSBGP(kernels=["additive", "dominance"], rsi=False, em_iters=40,
                      kernel_correlation_threshold=None,
                      kernel_prior=None, kernel_floor=None),
                MSBGP(kernels=["additive", "dominance", "gaussian", "poly"],
                      rsi=False, em_iters=40, kernel_correlation_threshold=None,
                      kernel_prior=None, kernel_floor=None)]):
            dense_model.fit(X, y)
            mem_model.fit(Xm, y)
            pd = dense_model.predict(X[:9])
            pm = mem_model.predict(Xm[:9])
            assert np.allclose(pm, pd, rtol=1e-9, atol=1e-9)
            wd = dense_model.get_kernel_weights()
            wm = mem_model.get_kernel_weights()
            for name in wd:
                assert abs(wd[name] - wm[name]) < 1e-9
        csr = genotype_memmap_to_csr(Xm, chunk_size=8)
        assert np.allclose(csr.toarray(), X.astype(np.float32))
    finally:
        try:
            if hasattr(Xm, "_mmap"):
                Xm._mmap.close()
        except (UnboundLocalError, AttributeError, ValueError):
            pass
        try:
            del Xm
        except UnboundLocalError:
            pass
        gc.collect()
        shutil.rmtree(td, ignore_errors=True)


def test_memmap_validation_rejects_bad_codes_and_nan():
    td = Path(__file__).resolve().parent / "_tmp_memmap_bad"
    shutil.rmtree(td, ignore_errors=True)
    td.mkdir(parents=True, exist_ok=True)
    try:
        path = td / "bad.dat"
        bad = np.array([[0, 1, 3], [2, 0, 1]], dtype=np.int8)
        try:
            save_genotype_memmap(bad, path, dtype=np.int8, overwrite=True,
                                 validate=True)
        except ValueError:
            pass
        else:
            raise AssertionError("bad 0/1/2 coding was not rejected")
        nan_path = td / "nan.dat"
        nan_x = np.array([[0.0, 1.0], [np.nan, 2.0]])
        try:
            validate_genotype_memmap(nan_x, require_integer=True,
                                     allow_missing=False)
        except ValueError:
            pass
        else:
            raise AssertionError("NaN genotype was not rejected")
        assert nan_path.parent.exists()
    finally:
        gc.collect()
        shutil.rmtree(td, ignore_errors=True)


def test_reml_multistart_diagnostics_and_objective():
    """REML multistart is deterministic and never worse than single-start."""
    d = simulate_gs_data(n=45, m=32, architecture="polygenic", h2=0.55,
                         seed=93)
    single = MSBGP(kernels=["additive", "gaussian"], rsi=False, em_iters=60,
                   kernel_correlation_threshold=None,
                   kernel_prior=None, kernel_floor=None,
                   reml_init="single").fit(d["X"], d["y"])
    multi = MSBGP(kernels=["additive", "gaussian"], rsi=False, em_iters=60,
                  kernel_correlation_threshold=None,
                  kernel_prior=None, kernel_floor=None,
                  reml_init="multi_start", reml_multistart=3).fit(d["X"], d["y"])
    assert multi.optim_n_starts_ == 3
    assert len(multi.optim_multistart_results_) == 3
    assert multi.optim_start_label_
    finite_single = [float(r.fun) for r in single.optim_multistart_results_
                     if np.isfinite(float(r.fun))]
    finite_multi = [float(r.fun) for r in multi.optim_multistart_results_
                    if np.isfinite(float(r.fun))]
    assert finite_single and finite_multi
    assert min(finite_multi) <= min(finite_single) + 1e-8


