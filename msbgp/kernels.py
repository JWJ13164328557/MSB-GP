"""Kernel functions and a leakage-free kernel state for genomic prediction.

The public kernel functions share a single genotype-dosage convention.  The
:class:`KernelState` object added here is used by MSB-GP so that every quantity
estimated from markers (allele frequency, kernel normaliser, bandwidth,
polynomial centre, interaction-feature scaling) is fitted on the training set
only and reused when forming cross-kernels.
"""

import warnings

import numpy as np
from scipy import sparse as sp



def _is_sparse(X):
    """Return True for scipy sparse genotype/dosage matrices."""
    return sp.issparse(X)


def _is_memmap(X):
    """Return True for numpy memory-mapped genotype matrices."""
    return isinstance(X, np.memmap)


def _as_float_matrix(X):
    """Return a numeric matrix without densifying sparse/memmap inputs.

    Dense in-memory arrays are normalised to float64 for numerical stability.
    Sparse matrices are kept sparse.  Memory-mapped arrays are intentionally
    returned as-is so downstream kernel builders can stream them instead of
    silently materialising a full float64 copy at the API boundary.
    """
    if _is_sparse(X):
        return X.astype(float, copy=False)
    if _is_memmap(X):
        return X
    return np.asarray(X, dtype=float)


def _sumsq_dense(X, block_size=8192):
    """Sum of squared entries without a full dense temporary for memmaps."""
    if not _is_memmap(X):
        return float(np.sum(np.asarray(X, dtype=float) ** 2))
    total = 0.0
    for start in range(0, X.shape[0], int(block_size)):
        block = np.asarray(X[start:start + int(block_size)], dtype=float)
        total += float(np.sum(block * block))
    return total


def _to_dense(X):
    """Convert sparse or array-like input to a dense float64 numpy array."""
    if _is_sparse(X):
        return np.asarray(X.toarray(), dtype=float)
    return np.asarray(X, dtype=float)


def _col_mean(X):
    """Column means as a 1-D float64 vector for dense or sparse inputs."""
    if _is_sparse(X):
        return np.asarray(X.mean(axis=0)).ravel().astype(float)
    return np.asarray(X.mean(axis=0), dtype=float).ravel()


def _col_std(X):
    """Population column standard deviations for dense or sparse inputs."""
    if _is_sparse(X):
        mean = _col_mean(X)
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(float)
        return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return np.asarray(X.std(axis=0), dtype=float).ravel()


def _row_sumsq(X):
    """Row-wise squared norms without densifying sparse inputs."""
    if _is_sparse(X):
        return np.asarray(X.multiply(X).sum(axis=1)).ravel().astype(float)
    X = np.asarray(X, dtype=float)
    return np.einsum("ij,ij->i", X, X)


def _centered_gram_raw(X, mu):
    """Return (X - 1 mu^T)(X - 1 mu^T)^T for dense, sparse, or memmap X."""
    mu = np.asarray(mu, dtype=float).ravel()
    if _is_sparse(X) or _is_memmap(X):
        X = X.tocsr().astype(float) if _is_sparse(X) else X
        K = _to_dense(X @ X.T) if _is_sparse(X) else np.asarray(X @ X.T, dtype=float)
        row = np.asarray(X @ mu.reshape(-1, 1)).ravel()
        K = K - row[:, None] - row[None, :] + float(np.dot(mu, mu))
        return K
    X = np.asarray(X, dtype=float)
    Z = X - mu
    return Z @ Z.T


def _centered_cross_raw(X_left, X_right, mu):
    """Return (X_left - 1 mu^T)(X_right - 1 mu^T)^T for dense/sparse/memmap X."""
    mu = np.asarray(mu, dtype=float).ravel()
    if _is_sparse(X_left) or _is_sparse(X_right) or _is_memmap(X_left) or _is_memmap(X_right):
        left = X_left.tocsr().astype(float) if _is_sparse(X_left) else X_left
        right = X_right.tocsr().astype(float) if _is_sparse(X_right) else X_right
        K = _to_dense(left @ right.T) if (_is_sparse(left) or _is_sparse(right)) else np.asarray(left @ right.T, dtype=float)
        left_row = np.asarray(left @ mu.reshape(-1, 1)).ravel()
        right_row = np.asarray(right @ mu.reshape(-1, 1)).ravel()
        K = K - left_row[:, None] - right_row[None, :] + float(np.dot(mu, mu))
        return K
    left = np.asarray(X_left, dtype=float)
    right = np.asarray(X_right, dtype=float)
    return (left - mu) @ (right - mu).T


__all__ = [
    "additive_kernel",
    "dominance_kernel",
    "gaussian_kernel",
    "polynomial_kernel",
    "epistasis_kernel",
    "build_kernel_library",
    "KernelState",
]


def _standardize(X, center=True, scale=False):
    """Center (and optionally scale) marker columns."""
    X = _as_float_matrix(X)
    if _is_sparse(X) and center:
        # Centering a sparse matrix creates a dense result; keep that explicit
        # here because public standalone kernels are not the memory-critical
        # MSB-GP path.  KernelState uses sparse-preserving formulas instead.
        X = _to_dense(X)
    if center:
        X = X - _col_mean(X) if not _is_sparse(X) else X
    if scale:
        if _is_sparse(X):
            s = _col_std(X)
            X = X @ sp.diags(np.divide(1.0, np.where(s == 0, 1.0, s)))
        else:
            s = np.asarray(X.std(axis=0), dtype=float)
            s[s == 0] = 1.0
            X = X / s
    return X


def _normalize_relationship(K):
    """Scale a raw kernel so its average diagonal equals one."""
    K = np.asarray(K, dtype=float)
    d = np.trace(K) / K.shape[0]
    if d == 0:
        return K
    return K / d


def additive_kernel(X, allele_freq=None, center=True):
    """VanRaden genomic relationship matrix."""
    X = _as_float_matrix(X)
    if allele_freq is None:
        p = 0.5 * _col_mean(X)
    else:
        p = np.asarray(allele_freq, dtype=float).ravel()
    denom = 2.0 * np.sum(p * (1.0 - p))
    if denom == 0:
        denom = 1.0
    raw = _centered_gram_raw(X, 2.0 * p)
    return _normalize_relationship(raw / denom)


def dominance_kernel(X):
    """Dominance relationship matrix using the coding of Su et al. (2012)."""
    # Dominance codes are non-zero at essentially every marker after
    # centering, so this standalone dense kernel intentionally materialises X.
    X = _to_dense(X)
    n, m = X.shape
    p = 0.5 * (X.sum(axis=0) / n)
    W = np.zeros_like(X, dtype=float)
    W[X == 0] = -2.0 * (p ** 2)[np.where(X == 0)[1]]
    W[X == 1] = (2.0 * p * (1.0 - p))[np.where(X == 1)[1]]
    W[X == 2] = -((1.0 - p) ** 2)[np.where(X == 2)[1]]
    denom = np.sum((2.0 * p * (1.0 - p)) ** 2)
    if denom == 0:
        denom = 1.0
    return _normalize_relationship((W @ W.T) / denom)


def gaussian_kernel(X, sigma=1.0, center=False):
    """Radial-basis kernel with training-set median bandwidth by default."""
    X = _as_float_matrix(X)
    if center:
        X = _standardize(X)
    if _is_sparse(X):
        rn = _row_sumsq(X)
        sq = rn[:, None] + rn[None, :] - 2.0 * _to_dense(X @ X.T)
    else:
        sq = ((X ** 2).sum(axis=1)[:, None] + (X ** 2).sum(axis=1)[None, :]
              - 2.0 * (X @ X.T))
    sq = np.maximum(sq, 0.0)
    if sigma is None:
        triu = sq[np.triu_indices(sq.shape[0], 1)]
        sigma = np.median(triu) if triu.size else 1.0
        sigma = max(float(sigma), 1e-6)
    return _normalize_relationship(np.exp(-sq / (2.0 * float(sigma) ** 2)))


def polynomial_kernel(X, degree=2, coef0=1.0, center=True):
    """Polynomial kernel; degree two contains pairwise interaction terms."""
    X = _as_float_matrix(X)
    if center:
        X = _standardize(X)
        gram = _to_dense(X @ X.T)
    elif _is_sparse(X):
        gram = _to_dense(X @ X.T)
    else:
        gram = X @ X.T
    return _normalize_relationship((gram + coef0) ** degree)


def epistasis_kernel(X, type="AA"):
    """Hadamard-product kernels for AA, AD or DD epistasis."""
    A = additive_kernel(X)
    D = dominance_kernel(X)
    t = str(type).upper()
    if t == "AA":
        return _normalize_relationship(A * A)
    if t == "AD":
        return _normalize_relationship(A * D)
    if t == "DD":
        return _normalize_relationship(D * D)
    raise ValueError("epistasis type must be one of: AA, AD, DD")


class KernelState:
    """Training-only state needed to build train and cross-kernels.

    This avoids recomputing allele frequencies,
    bandwidths and polynomial centres on ``train + validation``.  It therefore
    prevents a subtle form of cross-validation leakage and guarantees that
    training and validation kernels are on the same fitted scale.
    """

    def __init__(self):
        self.X_train_ = None
        self.residualize_ = set()
        self._deflation_ = {}
        self.rsi_terms_ = []
        self.rsi_mu_ = None
        self.rsi_sd_ = None
        self.rsi_H_train_ = None
        self.rsi_scale_ = 1.0
        # Dedicated pairwise interaction-feature kernels (AA/AD/DD).
        self.epi_pairs_ = {}
        self.epi_H_train_ = {}
        self.epi_feature_mu_ = {}
        self.epi_feature_sd_ = {}
        self.epi_add_mu_ = None
        self.epi_add_sd_ = None
        self.epi_dom_mu_ = None
        self.epi_dom_sd_ = None
        # When non-empty, these selected pair terms replace all-pair AA/AD/DD.
        self.epi_terms_ = []
        self.epi_feature_stats_ = {}
        self.Z_std_train_ = None
        self.D_std_train_ = None
        # Sparse-additive (screened-subset) kernel state.
        self.sparse_add_idx_ = np.array([], dtype=int)
        self.sparse_add_scale_ = 1.0
        self.sparse_add_top_markers_ = 0
        # Optional temporary cache used only while several MSB-GP variants are
        # fitted on the same training split.  It is explicitly cleared by the
        # caller so final models do not retain all n-by-n kernels.
        self._training_kernel_cache_enabled_ = False
        self._training_kernel_cache_ = {}
        self._cross_kernel_cache_enabled_ = False
        self._cross_kernel_cache_ = {}

    def enable_training_kernel_cache(self):
        """Enable a temporary per-kernel training cache for shared-state fits."""
        self._training_kernel_cache_enabled_ = True
        self._training_kernel_cache_ = {}
        return self

    def clear_training_kernel_cache(self):
        """Disable and release temporary training-kernel arrays."""
        self._training_kernel_cache_enabled_ = False
        self._training_kernel_cache_ = {}
        return self

    def enable_cross_kernel_cache(self):
        """Enable a temporary cross-kernel cache for repeated predictions."""
        self._cross_kernel_cache_enabled_ = True
        self._cross_kernel_cache_ = {}
        return self

    def clear_cross_kernel_cache(self):
        """Disable and release temporary cross-kernel arrays."""
        self._cross_kernel_cache_enabled_ = False
        self._cross_kernel_cache_ = {}
        return self

    @staticmethod
    def _squared_distances(X_tr, X_other=None):
        X_tr = _as_float_matrix(X_tr)
        if X_other is None:
            rn = _row_sumsq(X_tr)
            cross = _to_dense(X_tr @ X_tr.T) if _is_sparse(X_tr) else X_tr @ X_tr.T
            sq = rn[:, None] + rn[None, :] - 2.0 * cross
        else:
            X_other = _as_float_matrix(X_other)
            cross = (_to_dense(X_tr @ X_other.T)
                     if (_is_sparse(X_tr) or _is_sparse(X_other))
                     else X_tr @ X_other.T)
            sq = _row_sumsq(X_tr)[:, None] + _row_sumsq(X_other)[None, :] - 2.0 * cross
        return np.maximum(sq, 0.0)

    @staticmethod
    def _additive_parts(X):
        X = _as_float_matrix(X)
        p = np.clip(0.5 * _col_mean(X), 1e-8, 1.0 - 1e-8)
        Z = None if (_is_sparse(X) or _is_memmap(X)) else X - 2.0 * p
        denom = max(2.0 * np.sum(p * (1.0 - p)), 1e-12)
        return p, Z, denom

    @staticmethod
    def _dominance_parts(X):
        # The dominance coding is dense after centering.  Keep this conversion
        # explicit and local; sparse input still reduces I/O and raw-genotype
        # storage before kernel construction.
        X = _to_dense(X)
        p = np.clip(0.5 * X.mean(axis=0), 1e-8, 1.0 - 1e-8)
        W = np.zeros_like(X, dtype=float)
        W[X == 0] = -2.0 * (p ** 2)[np.where(X == 0)[1]]
        W[X == 1] = (2.0 * p * (1.0 - p))[np.where(X == 1)[1]]
        W[X == 2] = -((1.0 - p) ** 2)[np.where(X == 2)[1]]
        denom = max(np.sum((2.0 * p * (1.0 - p)) ** 2), 1e-12)
        return W, denom

    @staticmethod
    def _abs_corr(A, y):
        A = np.asarray(A, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        yc = y - y.mean()
        Ac = A - A.mean(axis=0, keepdims=True)
        num = np.sum(Ac * yc[:, None], axis=0)
        den = np.sqrt(np.sum(Ac ** 2, axis=0) * np.sum(yc ** 2))
        return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-15)

    def fit(self, X, residualize=None, y=None, epi_top_markers=0,
            sparse_additive_top_markers=0):
        X = _as_float_matrix(X)
        # Keep a read-only reference instead of an unconditional copy.  The
        # matrix is only read after fitting; calibration/candidate stacking now
        # shares one KernelState across model variants fitted on the same split,
        # so this avoids one full n-by-m duplicate on large datasets.
        self.X_train_ = X
        n, m = X.shape
        # Genotype coding contract: additive/dominance/epistasis kernels follow
        # the 0/1/2 dosage convention.  Record whether the matrix satisfies it
        # so kernel selection can warn instead of silently building an
        # all-zero dominance/epistasis kernel from continuous dosages.
        if _is_sparse(X):
            values = X.data
            self.genotypes_are_integer_coded_ = bool(np.all(np.isin(values, (0.0, 1.0, 2.0))))
        else:
            self.genotypes_are_integer_coded_ = bool(np.all(np.isin(X, (0.0, 1.0, 2.0))))
        self.residualize_ = {str(r).upper() for r in (residualize or [])}
        self._deflation_ = {}

        p_add, Z_add, denom_add = self._additive_parts(X)
        self.p_add_, self.denom_add_ = p_add, denom_add
        self.Z_add_train_ = Z_add
        # Trace(Raw) = trace(ZZ')/denom = sum(Z**2)/denom.  Avoid a
        # persistent n-by-n raw additive matrix; it can be reconstructed from
        # the centred training genotype matrix on demand.
        if Z_add is None:
            # Exact trace without materialising the centred n-by-m genotype
            # matrix: trace(ZZ') = sum_ij (X_ij - 2p_j)^2.
            x2 = float(X.multiply(X).sum()) if _is_sparse(X) else _sumsq_dense(X)
            c = 2.0 * p_add
            xp = float(np.sum(np.asarray(X @ c.reshape(-1, 1)).ravel())) if _is_sparse(X) else float(np.sum(X * c))
            trace_raw = x2 + float(n) * float(np.sum(c * c)) - 2.0 * xp
        else:
            trace_raw = float(np.sum(Z_add * Z_add))
        self.additive_scale_ = max(trace_raw / denom_add / n, 1e-12)

        W_dom, denom_dom = self._dominance_parts(X)
        self.W_dom_train_ = W_dom
        self.denom_dom_ = denom_dom
        self.dominance_scale_ = max(
            float(np.sum(W_dom * W_dom) / denom_dom / n), 1e-12)

        # Dedicated AA/AD/DD kernels use explicit pairwise interaction-feature
        # maps rather than Hadamard products of genomic relationship matrices.
        # This matches the simulation generative model: products of standardised
        # additive codes (AA), additive-by-heterozygote products (AD), and
        # heterozygote-by-heterozygote products (DD).  All feature centring,
        # scaling, and marker screening are fitted on the training set only.
        add_mu = _col_mean(X)
        add_sd = _col_std(X)
        add_sd[add_sd < 1e-12] = 1.0
        self.epi_add_mu_ = add_mu
        self.epi_add_sd_ = add_sd
        if _is_sparse(X):
            # Pairwise AA/AD features are estimated on selected markers; this
            # dense standardised view is built only after the sparse input has
            # already reduced raw genotype storage.
            self.Z_std_train_ = _to_dense((X - add_mu) / add_sd)
        else:
            self.Z_std_train_ = (X - add_mu) / add_sd

        if _is_sparse(X):
            D_raw = _to_dense((X == 1.0).astype(float))
        else:
            D_raw = (X == 1.0).astype(float)
        dom_mu = D_raw.mean(axis=0)
        dom_sd = D_raw.std(axis=0)
        dom_sd[dom_sd < 1e-12] = 1.0
        self.epi_dom_mu_ = dom_mu
        self.epi_dom_sd_ = dom_sd
        self.D_std_train_ = (D_raw - dom_mu) / dom_sd

        requested = int(epi_top_markers or 0)
        if requested <= 0:
            k_top = int(m)
        else:
            k_top = int(min(requested, m))
        self.epi_top_markers_ = int(k_top)
        all_idx = np.arange(m)
        self.epi_idx_AA_ = all_idx.copy()
        self.epi_idx_DD_ = all_idx.copy()
        self.epi_idx_AD_a_ = all_idx.copy()
        self.epi_idx_AD_d_ = all_idx.copy()
        if k_top > 0 and k_top < m:
            if y is not None:
                y_arr = np.asarray(y, dtype=float).ravel()
                score_a = np.abs(self._abs_corr(self.Z_std_train_, y_arr))
                score_d = np.abs(self._abs_corr(self.D_std_train_, y_arr))
            else:
                score_a = self.Z_std_train_.std(axis=0)
                score_d = self.D_std_train_.std(axis=0)
            self.epi_idx_AA_ = np.argsort(score_a)[-k_top:]
            self.epi_idx_AD_a_ = self.epi_idx_AA_.copy()
            self.epi_idx_DD_ = np.argsort(score_d)[-k_top:]
            self.epi_idx_AD_d_ = self.epi_idx_DD_.copy()

        # Sparse-additive screening: select the top markers by training-only
        # marginal association so a dedicated low-rank additive kernel can
        # capture sparse / oligogenic additive signal on top of the dense
        # polygenic background.  Selection uses only training phenotypes.
        self._fit_sparse_additive(X, y, sparse_additive_top_markers)

        sq = self._squared_distances(X)
        triu = sq[np.triu_indices(n, 1)]
        self.gaussian_sigma_ = max(float(np.median(triu)) if triu.size else 1.0,
                                   1e-6)
        G_train_raw = np.exp(-sq / (2.0 * self.gaussian_sigma_ ** 2))
        self.gaussian_scale_ = max(float(np.mean(np.diag(G_train_raw))), 1e-12)

        self.poly_mu_ = _col_mean(X)
        P_train_raw = (_centered_gram_raw(X, self.poly_mu_) + 1.0) ** 2
        self.poly_scale_ = max(float(np.mean(np.diag(P_train_raw))), 1e-12)

        for typ in ("AA", "AD", "DD"):
            raw = self._raw_epi_train(typ)
            setattr(self, f"epistasis_scale_{typ}_",
                    max(float(np.mean(np.diag(raw))), 1e-12))
        return self

    def _fit_sparse_additive(self, X, y, top):
        """Screen markers for the sparse-additive subset kernel (train-only).

        The selected subset is the ``top`` markers with the largest absolute
        training-set correlation between the standardised additive code and the
        phenotype.  Ties are broken deterministically by index.  When ``top``
        is not positive or no phenotype is available the component is disabled.
        """
        n, m = X.shape
        top = int(top or 0)
        self.sparse_add_idx_ = np.array([], dtype=int)
        self.sparse_add_scale_ = 1.0
        self.sparse_add_top_markers_ = 0
        if top <= 0:
            return
        k = int(min(top, m))
        if k <= 0:
            return
        Z_std = getattr(self, "Z_std_train_", None)
        if Z_std is None or y is None:
            # fall back to spread-based ranking if no phenotype is available
            score = Z_std.std(axis=0) if Z_std is not None else np.ones(m)
        else:
            score = np.abs(self._abs_corr(Z_std, np.asarray(y, dtype=float).ravel()))
        # deterministic: rank by score, break ties by ascending index
        order = np.lexsort((np.arange(m), -score))
        idx = np.sort(order[:k])
        self.sparse_add_idx_ = idx.astype(int)
        self.sparse_add_top_markers_ = int(k)
        raw = self._sub_additive_train(idx)
        self.sparse_add_scale_ = max(float(np.mean(np.diag(raw))), 1e-12)

    def _epi_train_parts(self, typ):
        typ = str(typ).upper()
        if typ == "AA":
            return self.Z_std_train_[:, self.epi_idx_AA_]
        if typ == "DD":
            return self.D_std_train_[:, self.epi_idx_DD_]
        if typ == "AD":
            return (self.Z_std_train_[:, self.epi_idx_AD_a_],
                    self.D_std_train_[:, self.epi_idx_AD_d_])
        raise ValueError(f"Unknown epistasis type {typ}")

    def _sparse_epi_feature_matrix(self, X, typ, fit=False):
        """Build selected-pair interaction features with training-only scaling."""
        from .rsi import rsi_feature_matrix
        typ = str(typ).upper()
        terms = [t for t in self.epi_terms_ if str(t[0]).upper() == typ]
        H = rsi_feature_matrix(X, terms)
        if H.shape[1] == 0:
            return H
        if fit:
            mu = H.mean(axis=0)
            sd = H.std(axis=0)
            sd[sd < 1e-12] = 1.0
            self.epi_feature_stats_[typ] = (mu, sd)
        else:
            mu, sd = self.epi_feature_stats_.get(
                typ, (np.zeros(H.shape[1]), np.ones(H.shape[1])))
        return (H - mu) / sd

    def set_epi_terms(self, terms):
        """Use a selected sparse AA/AD/DD interaction set instead of all pairs."""
        clean = []
        for term in (terms or []):
            try:
                kind, i, j = term
            except Exception:
                continue
            kind = str(kind).upper()
            if kind in {"AA", "AD", "DD"}:
                clean.append((kind, int(i), int(j)))
        self.epi_terms_ = clean
        self.epi_feature_stats_ = {}
        self._training_kernel_cache_ = {}
        self._cross_kernel_cache_ = {}
        for typ in ("AA", "AD", "DD"):
            H = self._sparse_epi_feature_matrix(self.X_train_, typ, fit=True)
            if H.shape[1] == 0:
                setattr(self, f"epistasis_scale_{typ}_", 1.0)
                continue
            raw = H @ H.T / max(1, H.shape[1])
            setattr(self, f"epistasis_scale_{typ}_",
                    max(float(np.mean(np.diag(raw))), 1e-12))
        return self

    def _epi_cross_parts(self, typ, X_va):
        typ = str(typ).upper()
        X_va = np.asarray(X_va, dtype=float)
        Z_va = (X_va - self.epi_add_mu_) / self.epi_add_sd_
        D_raw = (X_va == 1.0).astype(float)
        D_va = (D_raw - self.epi_dom_mu_) / self.epi_dom_sd_
        if typ == "AA":
            return Z_va[:, self.epi_idx_AA_]
        if typ == "DD":
            return D_va[:, self.epi_idx_DD_]
        if typ == "AD":
            return (Z_va[:, self.epi_idx_AD_a_],
                    D_va[:, self.epi_idx_AD_d_])
        raise ValueError(f"Unknown epistasis type {typ}")

    @staticmethod
    def _pair_product_kernel(typ, left, right):
        """All-pairs product kernel without materialising pair features.

        The simulation generative model defines AA/DD as products of two
        distinct loci and AD as an additive-code-by-dominance-code product at
        two distinct loci.  These identities compute the same kernels directly
        from marker matrices:

            K_AA = (Z Z^T) o (Z Z^T) - (Z o Z)(Z o Z)^T
            K_DD = (D D^T) o (D D^T) - (D o D)(D o D)^T
            K_AD = (Z Z^T) o (D D^T) - (Z o D)(Z o D)^T

        where ``o`` is Hadamard multiplication.  The corresponding cross-kernel
        blocks follow by replacing the right-hand marker matrix with the
        validation matrix.  This is algebraically identical to summing over all
        ordered pairs of distinct markers but avoids constructing the enormous
        m(m-1)/2-dimensional pair-feature matrices.
        """
        typ = str(typ).upper()
        if typ in {"AA", "DD"}:
            A, B = left, right
            G = A @ B.T
            corr = (A ** 2) @ (B ** 2).T
            return G * G - corr
        if typ == "AD":
            (A, D), (B, E) = left, right
            return (A @ B.T) * (D @ E.T) - ((A * D) @ (B * E).T)
        raise ValueError(f"Unknown epistasis type {typ}")

    def _raw_epi_train(self, typ):
        typ = str(typ).upper()
        if self.epi_terms_:
            H = self._sparse_epi_feature_matrix(self.X_train_, typ, fit=False)
            return H @ H.T / max(1, H.shape[1])
        if typ == "AA":
            Z = self._epi_train_parts("AA")
            return self._pair_product_kernel("AA", Z, Z)
        if typ == "DD":
            D = self._epi_train_parts("DD")
            return self._pair_product_kernel("DD", D, D)
        if typ == "AD":
            Z, D = self._epi_train_parts("AD")
            return self._pair_product_kernel("AD", (Z, D), (Z, D))
        raise ValueError(f"Unknown epistasis type {typ}")

    def set_rsi(self, terms, H_train=None):
        """Register selected RSI terms and their training-only scaling."""
        self.rsi_terms_ = list(terms or [])
        self.rsi_mu_ = None
        self.rsi_sd_ = None
        self.rsi_H_train_ = None
        self.rsi_scale_ = 1.0
        if self.rsi_terms_:
            from .rsi import rsi_feature_matrix
            H = rsi_feature_matrix(self.X_train_, self.rsi_terms_)
            mu = H.mean(axis=0)
            sd = H.std(axis=0)
            sd[sd < 1e-12] = 1.0
            self.rsi_mu_, self.rsi_sd_ = mu, sd
            Hs = (H - mu) / sd
            K = Hs @ Hs.T / max(1, Hs.shape[1])
            self.rsi_scale_ = max(float(np.mean(np.diag(K))), 1e-12)
            if H_train is not None:
                self.rsi_H_train_ = Hs
        return self

    def _sub_additive_train(self, idx):
        idx = np.asarray(idx, dtype=int)
        X = self.X_train_[:, idx]
        p = np.clip(0.5 * _col_mean(X), 1e-8, 1 - 1e-8)
        denom = max(2.0 * float(np.sum(p * (1.0 - p))), 1e-12)
        return _centered_gram_raw(X, 2.0 * p) / denom

    def _sub_dominance_train(self, idx):
        idx = np.asarray(idx, dtype=int)
        X = self.X_train_[:, idx]
        p = np.clip(0.5 * _col_mean(X), 1e-8, 1 - 1e-8)
        W = self._dominance_parts(X)[0]
        denom = max(np.sum((2.0 * p * (1.0 - p)) ** 2), 1e-12)
        return W @ W.T / denom

    def _base_training_kernel(self, name):
        name = str(name)
        upper = name.upper()
        if name in ("additive",):
            raw = _centered_gram_raw(self.X_train_, 2.0 * self.p_add_)
            return raw / (self.denom_add_ * self.additive_scale_)
        if upper in ("ADDITIVE_SPARSE", "ADD_SPARSE", "SPARSE_ADDITIVE"):
            if self.sparse_add_idx_.size == 0:
                n = self.X_train_.shape[0]
                return np.zeros((n, n))
            raw = self._sub_additive_train(self.sparse_add_idx_)
            return raw / self.sparse_add_scale_
        if name == "dominance":
            return (self.W_dom_train_ @ self.W_dom_train_.T) / (
                self.denom_dom_ * self.dominance_scale_)
        if name == "gaussian":
            sq = self._squared_distances(self.X_train_)
            raw = np.exp(-sq / (2.0 * self.gaussian_sigma_ ** 2))
            return raw / self.gaussian_scale_
        if name == "poly":
            raw = (_centered_gram_raw(self.X_train_, self.poly_mu_) + 1.0) ** 2
            return raw / self.poly_scale_
        if upper in {"AA", "AD", "DD"}:
            raw = self._raw_epi_train(upper)
            return raw / getattr(self, f"epistasis_scale_{upper}_")
        raise ValueError(f"Unknown kernel {name}")

    def _is_residualized(self, name):
        upper = str(name).upper()
        return upper in self.residualize_ and upper != "ADDITIVE"

    def _deflation_coeff(self, name):
        upper = str(name).upper()
        if upper not in self._deflation_:
            key = upper if upper in {"AA", "AD", "DD"} else name
            Kx = self._base_training_kernel(key)
            Ka = self._base_training_kernel("additive")
            num = float(np.sum(Kx * Ka))
            den = float(np.sum(Ka * Ka))
            coeff = num / den if den > 0.0 else 0.0
            self._deflation_[upper] = float(np.clip(coeff, 0.0, 1.0))
        return self._deflation_[upper]

    def training_kernel(self, name):
        cache_enabled = bool(getattr(self, "_training_kernel_cache_enabled_", False))
        cache = getattr(self, "_training_kernel_cache_", None)
        cache_key = str(name)
        if cache_enabled and cache is not None and cache_key in cache:
            return cache[cache_key]
        if str(name).upper() == "RSI":
            if not self.rsi_terms_:
                n = self.X_train_.shape[0]
                return np.zeros((n, n))
            if self.rsi_H_train_ is None:
                from .rsi import rsi_feature_matrix
                H = rsi_feature_matrix(self.X_train_, self.rsi_terms_)
                H = (H - self.rsi_mu_) / self.rsi_sd_
            else:
                H = self.rsi_H_train_
            return H @ H.T / max(1, H.shape[1]) / self.rsi_scale_
        K = self._base_training_kernel(name)
        if self._is_residualized(name):
            coef = self._deflation_coeff(name)
            K = K - coef * self._base_training_kernel("additive")
        if cache_enabled and cache is not None:
            cache[cache_key] = K
        return K

    def _rsi_cross_kernel(self, X_va):
        X_va = _as_float_matrix(X_va)
        if not self.rsi_terms_:
            return np.zeros((self.X_train_.shape[0], X_va.shape[0]))
        from .rsi import rsi_feature_matrix
        H_va = rsi_feature_matrix(X_va, self.rsi_terms_)
        H_va = (H_va - self.rsi_mu_) / self.rsi_sd_
        if self.rsi_H_train_ is None:
            H_tr = rsi_feature_matrix(self.X_train_, self.rsi_terms_)
            H_tr = (H_tr - self.rsi_mu_) / self.rsi_sd_
        else:
            H_tr = self.rsi_H_train_
        return H_tr @ H_va.T / max(1, H_tr.shape[1]) / self.rsi_scale_

    @staticmethod
    def _cross_cache_key(name, X_va):
        return (str(name), id(X_va), tuple(X_va.shape))

    def cross_kernel(self, name, X_va):
        X_va = _as_float_matrix(X_va)
        cache_enabled = bool(getattr(self, "_cross_kernel_cache_enabled_", False))
        cache = getattr(self, "_cross_kernel_cache_", None)
        key = self._cross_cache_key(name, X_va)
        if cache_enabled and cache is not None and key in cache:
            return cache[key]
        if str(name).upper() == "RSI":
            K = self._rsi_cross_kernel(X_va)
        else:
            base_blocks = self._cross_base_blocks(X_va)
            K = self._cross_kernel_with_base(name, X_va, base_blocks)
        if cache_enabled and cache is not None:
            cache[key] = K
        return K

    def weighted_cross_kernel(self, names, weights, X_va):
        """Weighted sum of cross-kernels with one A/D base computation."""
        names = list(names)
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != len(names):
            raise ValueError("names and weights must have the same length")
        X_va = _as_float_matrix(X_va)
        out = np.zeros((self.X_train_.shape[0], X_va.shape[0]), dtype=float)
        base_blocks = None
        cache_enabled = bool(getattr(self, "_cross_kernel_cache_enabled_", False))
        cache = getattr(self, "_cross_kernel_cache_", None)
        for name, w in zip(names, weights):
            w = float(w)
            if not np.isfinite(w) or w == 0.0:
                continue
            key = self._cross_cache_key(name, X_va)
            if cache_enabled and cache is not None and key in cache:
                out += w * cache[key]
                continue
            upper = str(name).upper()
            if upper == "RSI":
                K = self._rsi_cross_kernel(X_va)
            else:
                if base_blocks is None:
                    base_blocks = self._cross_base_blocks(X_va)
                K = self._cross_kernel_with_base(name, X_va, base_blocks)
            if cache_enabled and cache is not None:
                cache[key] = K
            out += w * K
        return out

    def _sub_additive_cross(self, idx, X_va):
        idx = np.asarray(idx, dtype=int)
        X_tr = self.X_train_[:, idx]
        X_va = _as_float_matrix(X_va)[:, idx]
        p = np.clip(0.5 * _col_mean(X_tr), 1e-8, 1 - 1e-8)
        denom = max(2.0 * float(np.sum(p * (1.0 - p))), 1e-12)
        return _centered_cross_raw(X_tr, X_va, 2.0 * p) / denom

    def _sub_dominance_cross(self, idx, X_va):
        idx = np.asarray(idx, dtype=int)
        X_tr = self.X_train_[:, idx]
        X_va = _to_dense(_as_float_matrix(X_va)[:, idx])
        p = np.clip(0.5 * _col_mean(X_tr), 1e-8, 1 - 1e-8)
        W_tr = self._dominance_parts(X_tr)[0]
        W_va = np.zeros((X_va.shape[0], idx.size), dtype=float)
        W_va[X_va == 0] = (-2.0 * (p ** 2))[np.where(X_va == 0)[1]]
        W_va[X_va == 1] = (2.0 * p * (1.0 - p))[np.where(X_va == 1)[1]]
        W_va[X_va == 2] = (-((1.0 - p) ** 2))[np.where(X_va == 2)[1]]
        denom = max(np.sum((2.0 * p * (1.0 - p)) ** 2), 1e-12)
        return W_tr @ W_va.T / denom

    def _cross_base_blocks(self, X_va):
        """Return normalized additive/dominance cross blocks in one pass."""
        X_va = _as_float_matrix(X_va)
        A_cross = _centered_cross_raw(
            self.X_train_, X_va, 2.0 * self.p_add_
        ) / (self.denom_add_ * self.additive_scale_)

        W_tr = getattr(self, "W_dom_train_", None)
        if W_tr is None:
            W_tr, _ = self._dominance_parts(self.X_train_)
        X_va_dom = _to_dense(X_va)
        p_dom = np.clip(0.5 * _col_mean(self.X_train_), 1e-8, 1.0 - 1e-8)
        W_va = np.zeros((X_va_dom.shape[0], p_dom.size), dtype=float)
        W_va[X_va_dom == 0] = -2.0 * (p_dom ** 2)[np.where(X_va_dom == 0)[1]]
        W_va[X_va_dom == 1] = (2.0 * p_dom * (1.0 - p_dom))[np.where(X_va_dom == 1)[1]]
        W_va[X_va_dom == 2] = -((1.0 - p_dom) ** 2)[np.where(X_va_dom == 2)[1]]
        D_cross = (W_tr @ W_va.T) / (self.denom_dom_ * self.dominance_scale_)
        return {"additive": A_cross, "dominance": D_cross}

    def _cross_kernel_with_base(self, name, X_va, base_blocks):
        raw = self._raw_cross_kernel(name, X_va, base_blocks=base_blocks)
        if self._is_residualized(name):
            coef = self._deflation_coeff(name)
            raw = raw - coef * self._raw_cross_kernel(
                "additive", X_va, base_blocks=base_blocks)
        return raw

    def _raw_cross_kernel(self, name, X_va, base_blocks=None):
        if base_blocks is None:
            upper0 = str(name).upper()
            need_base = (name in ("additive", "dominance")
                         or self._is_residualized(name))
            base_blocks = self._cross_base_blocks(X_va) if need_base else {}
        A_cross = base_blocks.get("additive")
        D_cross = base_blocks.get("dominance")

        name = str(name)
        upper = name.upper()
        if name == "additive":
            return A_cross
        if upper in ("ADDITIVE_SPARSE", "ADD_SPARSE", "SPARSE_ADDITIVE"):
            if self.sparse_add_idx_.size == 0:
                return np.zeros((self.X_train_.shape[0], X_va.shape[0]))
            raw = self._sub_additive_cross(self.sparse_add_idx_, X_va)
            return raw / self.sparse_add_scale_
        if name == "dominance":
            return D_cross
        if name == "gaussian":
            sq = self._squared_distances(self.X_train_, X_va)
            raw = np.exp(-sq / (2.0 * self.gaussian_sigma_ ** 2))
            return raw / self.gaussian_scale_
        if name == "poly":
            raw = (_centered_cross_raw(self.X_train_, X_va, self.poly_mu_) + 1.0) ** 2
            return raw / self.poly_scale_
        if upper in {"AA", "AD", "DD"}:
            if self.epi_terms_:
                H_tr = self._sparse_epi_feature_matrix(self.X_train_, upper, fit=False)
                H_va = self._sparse_epi_feature_matrix(X_va, upper, fit=False)
                if H_tr.shape[1] == 0:
                    cross = np.zeros((self.X_train_.shape[0], X_va.shape[0]))
                else:
                    cross = H_tr @ H_va.T / max(1, H_tr.shape[1])
                return cross / getattr(self, f"epistasis_scale_{upper}_")
            if upper == "AA":
                Z_tr = self._epi_train_parts("AA")
                Z_va = self._epi_cross_parts("AA", X_va)
                cross = self._pair_product_kernel("AA", Z_tr, Z_va)
            elif upper == "DD":
                D_tr = self._epi_train_parts("DD")
                D_va = self._epi_cross_parts("DD", X_va)
                cross = self._pair_product_kernel("DD", D_tr, D_va)
            else:
                Z_tr, D_tr = self._epi_train_parts("AD")
                Z_va, D_va = self._epi_cross_parts("AD", X_va)
                cross = self._pair_product_kernel(
                    "AD", (Z_tr, D_tr), (Z_va, D_va))
            return cross / getattr(self, f"epistasis_scale_{upper}_")
        raise ValueError(f"Unknown kernel {name}")

    def _dominance_cross(self, X_va):
        W_tr, _ = self._dominance_parts(self.X_train_)
        X_va = _to_dense(_as_float_matrix(X_va))
        p_dom = np.clip(0.5 * _col_mean(self.X_train_), 1e-8, 1.0 - 1e-8)
        W_va = np.zeros_like(X_va, dtype=float)
        W_va[X_va == 0] = -2.0 * (p_dom ** 2)[np.where(X_va == 0)[1]]
        W_va[X_va == 1] = (2.0 * p_dom * (1.0 - p_dom))[np.where(X_va == 1)[1]]
        W_va[X_va == 2] = -((1.0 - p_dom) ** 2)[np.where(X_va == 2)[1]]
        return W_tr @ W_va.T / self.denom_dom_

    def build_training_library(self, additive=True, dominance=True,
                               gaussian=True, polynomial=True, epistasis=None,
                               additive_sparse=False):
        epistasis = [] if epistasis in (None, False) else (
            [epistasis] if isinstance(epistasis, str) else epistasis)
        need_integer = bool(dominance) or len(epistasis) > 0
        if need_integer and not getattr(self, "genotypes_are_integer_coded_", True):
            warnings.warn(
                "Genotypes are not 0/1/2-integer coded; dominance and "
                "epistasis (AA/AD/DD) kernels are undefined for continuous "
                "dosages and will contribute ~0. Pass raw 0/1/2 genotypes to "
                "use these kernels.",
                RuntimeWarning, stacklevel=2)
        lib = []
        if additive:
            lib.append(("additive", self.training_kernel("additive")))
        if additive_sparse:
            lib.append(("additive_sparse",
                        self.training_kernel("additive_sparse")))
        if dominance:
            lib.append(("dominance", self.training_kernel("dominance")))
        if gaussian:
            lib.append(("gaussian", self.training_kernel("gaussian")))
        if polynomial:
            lib.append(("poly", self.training_kernel("poly")))
        for typ in epistasis:
            typ = str(typ).upper()
            lib.append((typ, self.training_kernel(typ)))
        return lib


def build_kernel_library(X, additive=True, dominance=True, gaussian=True,
                         polynomial=True, gaussian_sigma=None, poly_degree=2,
                         epistasis=False):
    """Assemble a normalised training-kernel library."""
    lib = []
    if additive:
        lib.append(("additive", additive_kernel(X)))
    if dominance:
        lib.append(("dominance", dominance_kernel(X)))
    if gaussian:
        lib.append(("gaussian", gaussian_kernel(X, sigma=gaussian_sigma)))
    if polynomial:
        lib.append(("polynomial" if False else "poly",
                    polynomial_kernel(X, degree=poly_degree)))
    if epistasis:
        if isinstance(epistasis, str):
            epistasis = [epistasis]
        for e in epistasis:
            name = str(e).upper()
            lib.append((name, epistasis_kernel(X, name)))
    return lib




