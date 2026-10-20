"""Baseline Genomic Selection methods used for benchmarking.

Implements the classical BLUP-family and Bayesian methods that the new MSB-GP
algorithm is designed to outperform or generalise:

  * GBLUP  -- mixed model with genomic relationship matrix (VanRaden 2008)
  * RRBLUP -- marker based ridge regression (equivalent to GBLUP)
  * BayesA -- t / scaled-t prior on marker variances (Meuwissen et al. 2001)
  * BayesC -- mixture prior: markers either have zero effect or a common
              variance (Habier et al. 2011)

All estimators expose a common sklearn-like interface:

    model.fit(X_tr, y_tr)
    yhat = model.predict(X_va)
    model.get_variance_components()
"""

import numpy as np

from .kernels import additive_kernel

__all__ = ["GBLUP", "RRBLUP", "BayesA", "BayesC"]




class GBLUP:
    """Genomic Best Linear Unbiased Prediction.

    Solves  y = mu + g + e  with  g ~ N(0, G*s2g), e ~ N(0, I*s2e)  using a
    REML profile over the noise-to-signal ratio lambda = s2e/s2g, then
    predicts breeding values of validation individuals through the G-matrix
    (VanRaden 2008).

    Numerically efficient: the spectral decomposition of G is done once and
    the REML profile is evaluated in the eigen-space (O(n) per candidate).
    """

    def __init__(self):
        pass

    def _reml_spectral(self, loglam, G, y):
        n = G.shape[0]
        lam = float(np.exp(loglam))
        d = self.D_ + lam
        # log|V|
        logdet = np.sum(np.log(d))
        # project y onto eigen-space
        Ut = self.U_.T
        ty = Ut @ y
        # GLS intercept via weights 1/d
        X = Ut @ np.ones(n)
        XtVX = np.sum(X ** 2 / d)
        XtVy = np.sum(X * ty / d)
        mu = XtVy / XtVX if XtVX > 0 else 0.0
        r_ty = ty - mu * X
        rVr = np.sum(r_ty ** 2 / d)
        return -0.5 * (logdet + (n - 1.0) * np.log(rVr / (n - 1.0)))

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).ravel()
        self.mu_ = y_train.mean()
        yc = y_train - self.mu_
        G = additive_kernel(X_train)
        n = G.shape[0]
        # spectral decomposition (symmetric)
        self.D_, self.U_ = np.linalg.eigh(G)
        self.D_ = np.maximum(self.D_, 1e-10)
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(
            lambda l: -self._reml_spectral(l, G, yc),
            bounds=(np.log(1e-6), np.log(1e6)), method="bounded")
        lam = float(np.exp(res.x))
        # residual variance in eigen-space
        d = self.D_ + lam
        ty = self.U_.T @ yc
        X = self.U_.T @ np.ones(n)
        XtVX = np.sum(X ** 2 / d)
        XtVy = np.sum(X * ty / d)
        mu = XtVy / XtVX
        r_ty = ty - mu * X
        rVr = np.sum(r_ty ** 2 / d)
        self.s2e_ = rVr / (n - 1.0)
        self.s2g_ = self.s2e_ / lam
        self.lambda_ = lam
        self.G_tr_ = G
        self.yc_ = yc
        self._last_X = X_train
        self.Ut_ = self.U_.T
        return self

    def predict(self, X_va):
        if not hasattr(self, "G_tr_"):
            raise RuntimeError("fit() must be called before predict()")
        X_va = np.asarray(X_va, dtype=float)
        X_tr = self._last_X
        X_all = np.vstack([X_tr, X_va])
        p = 0.5 * (X_all.sum(axis=0) / X_all.shape[0])
        Z_tr = X_tr - 2.0 * p
        Z_va = X_va - 2.0 * p
        denom = 2.0 * np.sum(p * (1.0 - p))
        G_tv = (Z_tr @ Z_va.T) / denom
        n = self.G_tr_.shape[0]
        # V^{-1} y via spectral form
        Vinvy = self.U_ @ ((self.Ut_ @ self.yc_) / (self.D_ + self.lambda_))
        g_va = G_tv.T @ Vinvy
        return self.mu_ + g_va


class RRBLUP:
    """Ridge-regression BLUP on marker effects (equivalent to GBLUP).

    Estimates marker effects with an L2 penalty tuned by cross-validation,
    then predicts y = X beta + mu.

    Efficient dual (n x n) form for the n < m regime: instead of solving the
    (m x m) system recomputed for every lambda, we precompute the (n x n)
    Gram matrix once (``X X^T``) and use the Woodbury identity
        beta = X^T (X X^T + lambda I)^{-1} y,
    so lambda tuning only needs cheap (n x n) solves.
    """

    def __init__(self, n_lambdas=30):
        self.n_lambdas = n_lambdas
        self.beta_ = None
        self.mu_ = 0.0

    def _ridge_dual(self, XXt, y, lmbda):
        """Solve RR-BLUP in dual form: alpha = (XX^T + lam I)^{-1} y."""
        n = XXt.shape[0]
        A = XXt + lmbda * np.eye(n)
        return np.linalg.solve(A, y)

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).ravel()
        self.mu_ = y.mean()
        yc = y - self.mu_
        n, m = X.shape
        # precompute Gram once
        XXt = X @ X.T
        # candidate lambdas relative to the trace scale of XXt
        lmbdas = np.logspace(-6, 6, self.n_lambdas) * (np.trace(XXt) / n)

        best = None
        best_alpha = None
        if n >= 10:
            rng = np.random.default_rng(0)
            idx = rng.permutation(n)
            k = 5
            folds = np.array_split(idx, k)
            for lam in lmbdas:
                errs = []
                for f in folds:
                    tr = np.setdiff1d(np.arange(n), f)
                    a = self._ridge_dual(XXt[np.ix_(tr, tr)], yc[tr], lam)
                    # Use the precomputed Gram (XXt = X X^T) so the inner CV no
                    # longer materialises large float64 marker-slice copies.
                    pred = XXt[np.ix_(f, tr)] @ a
                    errs.append(np.mean((yc[f] - pred) ** 2))
                e = np.mean(errs)
                if best is None or e < best:
                    best = e
                    best_alpha = lam
        else:
            best_alpha = lmbdas[len(lmbdas) // 2]

        # final fit on full training set with best lambda
        a = self._ridge_dual(XXt, yc, best_alpha)
        self.beta_ = X.T @ a
        self.X_ = X
        self.lambda_ = best_alpha
        return self

    def predict(self, X_va):
        X = np.asarray(X_va, dtype=float)
        return self.mu_ + X @ self.beta_


class BayesA:
    """BayesA: independent t / scaled-t prior on each marker variance.

    Uses a Gibbs sampler over marker effects beta_j ~ N(0, s2b_j) with
    s2b_j | s2b ~ scaled-inverse-chi2(nua, s2b), and a common residual variance.
    """
    def __init__(self, n_iter=1500, burnin=300, nua=4.0):
        self.n_iter = n_iter
        self.burnin = burnin
        self.nua = nua
        self.beta_ = None
        self.mu_ = 0.0

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).ravel()
        n, m = X.shape
        self.mu_ = y.mean()
        yc = y - self.mu_
        rng = np.random.default_rng(42)
        # data-driven priors (scale to the observed phenotype variance)
        var_y = float(np.var(yc)) + 1e-8
        nu_e = 4.0
        S_e = var_y * (1.0 - 0.5)          # residual prior ~ half the variance
        # initialize
        beta = np.zeros(m)
        s2b = np.full(m, var_y / m)
        s2e = var_y * 0.5
        # scale marker columns
        Xc = X
        # mean residual
        resid = yc.copy()
        # gamma prior scale
        nub = self.nua
        Sb = var_y / m
        n_samples = (self.n_iter - self.burnin)
        beta_sum = np.zeros(m)
        Xm = X.T                                # markers as contiguous rows (m,n)
        xjx = (X ** 2).sum(axis=0) + 1e-10      # column sums of squares (once)
        for it in range(self.n_iter):
            for j in range(m):
                xj = Xm[j]
                rj = resid + xj * beta[j]       # add back current beta_j
                xxs = xjx[j]
                rhs = xj @ rj
                C = xxs + s2e / s2b[j]
                mu_j = rhs / C
                var_j = s2e / C
                beta[j] = rng.normal(mu_j, np.sqrt(var_j + 1e-12))
                resid = rj - xj * beta[j]       # remove new beta_j
            rss = resid @ resid
            s2e = (nu_e * S_e + rss) / (rng.chisquare(nu_e + n) + 1e-12)
            s2b = (nub * Sb + beta ** 2) / (rng.chisquare(nub + 1, size=m) + 1e-12)
            if it >= self.burnin:
                beta_sum += beta

        self.beta_ = beta_sum / n_samples
        return self

    def predict(self, X_va):
        X = np.asarray(X_va, dtype=float)
        return self.mu_ + X @ self.beta_


class BayesC:
    """BayesC spike-slab mixture prior (Habier et al. 2011).

    A fraction ``pi`` of markers are assumed to have *exactly zero* effect
    (the spike), while the remaining ``1 - pi`` markers share a common effect
    variance ``s2b`` (the slab).  ``pi`` therefore follows the standard BayesC
    convention: it is the prior probability of a *zero* effect, i.e. the
    expected proportion of markers excluded from the model.

    Parameters
    ----------
    n_iter, burnin : int
        Total Gibbs iterations and number discarded as burn-in.
    pi : float
        Prior probability that a marker has zero effect (``0 <= pi < 1``).
        The default ``pi=0.99`` implies ~1% prior inclusion, i.e. a sparse
        spike-slab prior appropriate for most genomic architectures.
        ``pi=0.0`` reduces BayesC to a ridge-like Gaussian prior over all
        markers (prior inclusion probability = 1).

    Notes
    -----
    The prior *inclusion* probability is ``1 - pi`` and the prior odds of
    exclusion vs inclusion are ``pi / (1 - pi)``.  Here ``pi`` is the zero-effect proportion, so the prior inclusion
    probability is ``1 - pi``; ``pi=0.999`` would be effectively a ridge
    (99.9% of markers included).
    """
    def __init__(self, n_iter=1500, burnin=300, pi=0.99):
        if not (0.0 <= pi < 1.0):
            raise ValueError("pi must be in [0, 1); got %r" % (pi,))
        self.n_iter = n_iter
        self.burnin = burnin
        self.pi = pi
        self.beta_ = None
        self.mu_ = 0.0

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).ravel()
        n, m = X.shape
        self.mu_ = y.mean()
        yc = y - self.mu_
        rng = np.random.default_rng(7)
        var_y = float(np.var(yc)) + 1e-8
        nu_e = nu_b = 4.0
        S_e = var_y * 0.5
        S_b = var_y / m
        beta = np.zeros(m)
        delta = np.zeros(m, dtype=bool)
        s2b = var_y / m
        resid = yc.copy()
        pi0 = float(self.pi)          # prior prob. of zero effect (spike)
        pi_incl = 1.0 - pi0           # prior prob. of inclusion (slab)
        n_samples = self.n_iter - self.burnin
        beta_sum = np.zeros(m)
        # precompute
        Xm = X.T                                  # markers as contiguous rows (m,n)
        xjx = (X ** 2).sum(axis=0)                # column sums of squares
        for it in range(self.n_iter):
            rss = resid @ resid
            s2e = (nu_e * S_e + rss) / (rng.chisquare(nu_e + n) + 1e-12)
            # marker effect variance (scaled-inv-chi2) over currently-included set
            ninc = int(delta.sum())
            if ninc > 0:
                s2b = (nu_b * S_b + (beta ** 2).sum()) / (rng.chisquare(nu_b + ninc) + 1e-12)
            for j in range(m):
                xj = Xm[j]
                rj = resid + xj * beta[j]
                xxs = xjx[j] + 1e-10
                rhs = xj @ rj
                C = xxs + s2e / s2b
                mu_j = rhs / C
                var_j = s2e / C
                logBF = 0.5 * (np.log(xxs / C) + (rhs ** 2) / (s2e * xxs) - (rhs ** 2) / (s2e * C))
                # prior odds of exclusion vs inclusion = pi0 / (1 - pi0)
                if pi0 <= 0.0:
                    p_inc = 1.0                      # ridge limit
                else:
                    p_inc = 1.0 / (1.0 + (pi0 / pi_incl) * np.exp(-logBF))
                if rng.uniform() < p_inc:
                    delta[j] = True
                    beta[j] = rng.normal(mu_j, np.sqrt(var_j + 1e-12))
                else:
                    delta[j] = False
                    beta[j] = 0.0
                resid = rj - xj * beta[j]
            if it >= self.burnin:
                beta_sum += beta

        self.beta_ = beta_sum / n_samples
        # fraction of markers excluded from the model (zero-effect estimate)
        self.zero_frac_ = 1.0 - delta.sum() / m
        self.delta_frac_ = self.zero_frac_   # backward-compatible alias
        return self

    def predict(self, X_va):
        X = np.asarray(X_va, dtype=float)
        return self.mu_ + X @ self.beta_
