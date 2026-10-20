# -*- coding: utf-8 -*-
"""Multi-kernel genomic prediction baselines (GMKL).

Three lightweight multi-kernel baselines are provided for benchmarking:

* ``MeanGMKL`` -- equal-weight mean fusion of candidate kernels followed by
  kernel ridge regression.
* ``FHGMKL``   -- fold-wise heuristic weights: each candidate kernel is scored
  by leave-one-out predictive correlation, then positive correlations are
  normalised into kernel weights.
* ``CKAGMKL``  -- centered kernel alignment (CKA) weights between each kernel
  and the phenotype kernel.

These baselines are intentionally simpler than MSB-GP: they do not estimate
per-kernel variance components by a Bayesian marginal likelihood and do not
apply posterior calibration.
"""

import numpy as np

from .kernels import (additive_kernel, dominance_kernel, gaussian_kernel,
                      polynomial_kernel)

__all__ = ["MeanGMKL", "FHGMKL", "CKAGMKL"]


_ALPHAS = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                    1e-2, 3e-2, 1e-1, 3e-1, 1.0])


def _loocv_kernel_ridge(K, y):
    """Select lambda by closed-form LOOCV and return dual coefficients."""
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    eigvals, eigvecs = np.linalg.eigh(K)
    yt = eigvecs.T @ y
    best_mse, best_alpha, best_coef = np.inf, _ALPHAS[0], None
    # Reference denominator for standardised lambda.
    ref = max(float(np.mean(np.diag(K))), 1.0)
    for rel in _ALPHAS:
        lam = rel * ref
        denom = eigvals + lam
        coef_eig = yt / denom
        pred = eigvecs @ (eigvals * coef_eig)
        h_diag = eigvecs @ (eigvals / denom * eigvecs.T).T[:, 0] if False else \
            np.einsum("ij,j,ij->i", eigvecs, eigvals / denom, eigvecs)
        resid = y - pred
        loo_pred = y - resid / np.maximum(1.0 - h_diag, 1e-8)
        mse = float(np.mean((loo_pred - y) ** 2))
        if np.isfinite(mse) and mse < best_mse:
            best_mse, best_alpha = mse, lam
            best_coef = eigvecs @ coef_eig
    if best_coef is None:
        best_coef = np.zeros(n)
    return best_alpha, best_coef


def _pearson(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _cka(K, y):
    """CKA between a candidate kernel and the linear phenotype kernel."""
    n = K.shape[0]
    one = np.ones((n, 1))
    Kc = K - one @ np.mean(K, axis=0, keepdims=True) \
         - np.mean(K, axis=1, keepdims=True) @ one.T \
         + np.full((n, n), np.mean(K))
    y = np.asarray(y, dtype=float).ravel()
    yc = y - y.mean()
    Y = np.outer(yc, yc)
    denom = np.linalg.norm(Kc, ord="fro") * np.linalg.norm(Y, ord="fro")
    if denom < 1e-20:
        return 0.0
    return float(np.sum(Kc * Y) / denom)


def _normalise_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if w.sum() <= 1e-12:
        return np.full_like(w, 1.0 / len(w))
    return w / w.sum()


class _BaseGMKL:
    """Shared kernel-fusion ridge model."""

    _kernel_names = ("additive", "dominance", "gaussian", "poly2", "poly3")

    def __init__(self, kernels=None):
        self.kernels = kernels
        self.weights_ = None
        self.alpha_ = None

    def _kernel_list(self, X):
        X = np.asarray(X, dtype=float)
        names = list(self.kernels) if self.kernels is not None \
            else list(self._kernel_names)
        out = []
        for name in names:
            key = str(name).lower()
            if key == "additive":
                out.append(("additive", additive_kernel(X)))
            elif key == "dominance":
                out.append(("dominance", dominance_kernel(X)))
            elif key == "gaussian":
                out.append(("gaussian", gaussian_kernel(X, sigma=None)))
            elif key == "poly2":
                out.append(("poly2", polynomial_kernel(X, degree=2)))
            elif key == "poly3":
                out.append(("poly3", polynomial_kernel(X, degree=3)))
            else:
                raise ValueError(f"Unknown GMKL kernel: {name}")
        return out

    def _cross_kernels(self, X_tr, X_va):
        X_all = np.vstack([np.asarray(X_tr), np.asarray(X_va)])
        n_tr = len(X_tr)
        out = []
        for name, _ in self.Ktr_:
            key = name.lower()
            if key == "additive":
                K_all = additive_kernel(X_all)
            elif key == "dominance":
                K_all = dominance_kernel(X_all)
            elif key == "gaussian":
                K_all = gaussian_kernel(X_all, sigma=None)
            elif key == "poly2":
                K_all = polynomial_kernel(X_all, degree=2)
            elif key == "poly3":
                K_all = polynomial_kernel(X_all, degree=3)
            else:
                raise ValueError(f"Unknown GMKL kernel: {name}")
            out.append(K_all[:n_tr, n_tr:])
        return out

    def _fit_mixed(self, Kmix, y):
        self.mu_ = float(np.mean(y))
        yc = y - self.mu_
        self.alpha_, self.dual_ = _loocv_kernel_ridge(Kmix, yc)

    def predict(self, X_va):
        if not hasattr(self, "dual_"):
            raise RuntimeError("fit() must be called before predict()")
        Kcv = self._cross_kernels(self.X_, np.asarray(X_va))
        Kmix_cv = sum(w * K for w, K in zip(self.weights_, Kcv))
        return self.mu_ + Kmix_cv.T @ self.dual_


class MeanGMKL(_BaseGMKL):
    """Equal-weight multi-kernel ridge regression."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.Ktr_ = self._kernel_list(X)
        self.weights_ = np.full(len(self.Ktr_), 1.0 / len(self.Ktr_))
        Kmix = sum(w * K for w, (_, K) in zip(self.weights_, self.Ktr_))
        self.X_ = X
        self._fit_mixed(Kmix, y)
        return self


class FHGMKL(_BaseGMKL):
    """Fold-wise/fixed-heuristic weighted multi-kernel ridge regression.

    For each kernel, alpha is chosen by LOOCV and its out-of-fold prediction
    correlation with y is used as a non-negative fusion weight.  This is a
    transparent fixed-heuristic baseline, not a full marginal-likelihood MKL.
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.Ktr_ = self._kernel_list(X)
        scores = []
        for _, K in self.Ktr_:
            _, coef = _loocv_kernel_ridge(K, y - y.mean())
            oof = K @ coef
            scores.append(max(_pearson(oof, y), 0.0))
        self.weights_ = _normalise_weights(np.asarray(scores))
        Kmix = sum(w * K for w, (_, K) in zip(self.weights_, self.Ktr_))
        self.X_ = X
        self._fit_mixed(Kmix, y)
        return self


class CKAGMKL(_BaseGMKL):
    """Centered-kernel-alignment weighted multi-kernel ridge regression."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.Ktr_ = self._kernel_list(X)
        scores = np.asarray([_cka(K, y) for _, K in self.Ktr_])
        self.weights_ = _normalise_weights(scores)
        Kmix = sum(w * K for w, (_, K) in zip(self.weights_, self.Ktr_))
        self.X_ = X
        self._fit_mixed(Kmix, y)
        return self
