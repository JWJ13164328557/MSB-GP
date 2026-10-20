# -*- coding: utf-8 -*-
"""Residual sparse epistasis (RSI) kernel construction for MSB-GP.

The RSI layer is used internally by :class:`msbgp.msbgp.MSBGP` when
``kernels="auto"``.  It does *not* create a new public model.  The idea is:

1. Rank markers by their marginal association with the residual phenotype.
2. Construct candidate AA/AD/DD interaction features among top markers.
3. Keep only a small sparse set of interactions.
4. Convert the selected interactions into one kernel for the multi-kernel model.

This avoids diluting the epistatic signal with all M(M-1)/2 genome-wide
interactions and is therefore much more suitable for small-to-moderate n.
"""

from itertools import combinations

import numpy as np
from scipy import sparse as sp


def _is_sparse(X):
    return sp.issparse(X)


def _as_float_matrix(X):
    if _is_sparse(X):
        return X.astype(float, copy=False)
    return np.asarray(X, dtype=float)


def _to_dense(X):
    if _is_sparse(X):
        return np.asarray(X.toarray(), dtype=float)
    return np.asarray(X, dtype=float)


def _col_values(X, idx):
    col = X[:, int(idx)]
    if _is_sparse(col):
        return np.asarray(col.toarray()).ravel().astype(float)
    return np.asarray(col, dtype=float).ravel()


__all__ = ["build_rsi_terms", "rsi_feature_matrix", "rsi_kernel_from_terms"]


def _standardize_cols(A):
    A = _to_dense(A)
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (A - mu) / sd


def dominance_features(X):
    """Centered/scaled binary heterozygote indicators."""
    X = _to_dense(X)
    return _standardize_cols((X == 1).astype(float))


def _abs_corr(A, y):
    """Column-wise absolute Pearson correlation, tolerant to zero variance."""
    A = _to_dense(A)
    y = np.asarray(y, dtype=float).ravel()
    yc = y - y.mean()
    Ac = A - A.mean(axis=0, keepdims=True)
    num = np.sum(Ac * yc[:, None], axis=0)
    den = np.sqrt(np.sum(Ac ** 2, axis=0) * np.sum(yc ** 2))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-15)
    return np.abs(r)


def build_rsi_terms(X, y, top_markers=60, interactions_per_type=50,
                    max_terms=150, random_state=0):
    """Select sparse AA/AD/DD interaction terms.

    Returns a list of ``(kind, i, j)`` tuples, where ``i`` and ``j`` are
    marker indices.  Kinds are ``AA``, ``AD`` and ``DD``.
    """
    X = _as_float_matrix(X)
    y = np.asarray(y, dtype=float).ravel()
    n, m = X.shape
    top_markers = max(3, min(int(top_markers), m))
    interactions_per_type = max(1, int(interactions_per_type))

    # Marginal sure-independence screening for additive and dominance effects.
    Z = _standardize_cols(X)
    W = dominance_features(X)
    resid = y - y.mean()
    score_a = _abs_corr(Z, resid)
    score_d = _abs_corr(W, resid)
    idx_a = np.argsort(score_a)[-top_markers:]
    idx_d = np.argsort(score_d)[-top_markers:]

    cand = []
    cand.extend(("AA", int(i), int(j)) for i, j in combinations(idx_a, 2))
    cand.extend(("AD", int(i), int(j)) for i in idx_a for j in idx_d)
    cand.extend(("DD", int(i), int(j)) for i, j in combinations(idx_d, 2))

    # Score interactions in blocks to avoid a very large temporary matrix.
    scores = np.empty(len(cand), dtype=float)
    block = 2048
    for s in range(0, len(cand), block):
        block_terms = cand[s:s + block]
        H = rsi_feature_matrix(X, block_terms)
        scores[s:s + block] = _abs_corr(H, resid)

    # Take the strongest terms from each type and cap the total number.
    by_kind = {"AA": [], "AD": [], "DD": []}
    for score, term in zip(scores, cand):
        by_kind[term[0]].append((float(score), term))
    terms = []
    for kind in ("AA", "AD", "DD"):
        arr = sorted(by_kind[kind], key=lambda x: -x[0])
        terms.extend(term for _, term in arr[:interactions_per_type])
    if len(terms) > max_terms:
        # Keep the globally strongest capped set; ties are deterministic.
        scored = [(next(s for s, t in zip(scores, cand) if t == t0), t0)
                  for t0 in terms]
        terms = [t for _, t in sorted(scored, key=lambda x: (-x[0], x[1]))[:max_terms]]
    return terms


def rsi_feature_matrix(X, terms):
    """Construct raw interaction columns defined by ``terms``."""
    X = _as_float_matrix(X)
    if not terms:
        return np.empty((X.shape[0], 0), dtype=float)
    cols = []
    for kind, i, j in terms:
        i = int(i); j = int(j)
        xi = _col_values(X, i)
        xj = _col_values(X, j)
        if kind == "AA":
            v = xi * xj
        elif kind == "AD":
            v = xi * (xj == 1)
        elif kind == "DD":
            v = (xi == 1) * (xj == 1)
        else:
            raise ValueError(f"Unknown RSI interaction type: {kind}")
        cols.append(v)
    return np.column_stack(cols)


def rsi_kernel_from_terms(X_train, y_train, X_all, terms):
    """Build training and cross RSI kernels with training-derived scaling.

    The function returns ``(K_train, K_cross, scale)``.  ``K_cross`` is the
    train-by-validation block of the full RSI kernel, scaled consistently with
    ``K_train``.
    """
    X_train = _as_float_matrix(X_train)
    X_all = _as_float_matrix(X_all)
    n_tr = X_train.shape[0]
    H = rsi_feature_matrix(X_all, terms)
    if H.shape[1] == 0:
        return (np.zeros((n_tr, n_tr)), np.zeros((n_tr, X_all.shape[0] - n_tr)), 1.0)
    mu = H[:n_tr].mean(axis=0)
    sd = H[:n_tr].std(axis=0)
    sd[sd < 1e-12] = 1.0
    H = (H - mu) / sd
    K_all = H @ H.T / max(1, H.shape[1])
    scale = float(np.mean(np.diag(K_all[:n_tr, :n_tr])))
    if not np.isfinite(scale) or scale <= 1e-15:
        scale = 1.0
    K_all /= scale
    return K_all[:n_tr, :n_tr], K_all[:n_tr, n_tr:], scale
