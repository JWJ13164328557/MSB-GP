# -*- coding: utf-8 -*-
"""Post-hoc affine calibration + prediction-first higher-order gate.

Why calibration matters
-----------------------
Kernel-based genomic prediction models can produce predictions whose RANKING
is correct but whose absolute location/scale is biased.  Empirically, MSB-GP
on Rice showed a +3.4 unit mean shift (intercept 17.7 vs 14.3 for linear
baselines) caused by a dominant Gaussian-kernel component, inflating MSE by
2.26x while leaving Pearson r untouched.  An affine recalibration

    y_pred_cal = a * y_pred + b

removes exactly this bias without changing the ranking (a > 0), hence AUC,
BAC, AUPRC and Pearson r are all preserved; only MSE/RMSE/MAE improve.

Leakage-free estimation
-----------------------
The coefficients (a, b) MUST be estimated from training data only.  We use an
inner K-fold cross-validation on the training set to obtain out-of-fold
predictions, then fit ordinary least squares y ~ a * y_oof + b.  Test labels
are never touched.

Prediction-first higher-order gate
----------------------------------
Four optimisation objectives govern the current MSB-GP iteration:

1. Pure-additive architectures must not be damaged by complex kernels.
2. The additive -> dominance transition must be tracked correctly.
3. When epistasis is present the model must recognise an *increase* in
   higher-order signal rather than absorb it into the additive baseline.
4. Prediction accuracy takes priority over fine-grained variance
   decomposition.

Objectives 1+3 are enforced by an out-of-fold *higher-order gate*.  For every
inner fold we fit both the full kernel library and a biological **core** model
(additive + dominance only).  The gate score is

    higher_order_score = r(full_oof, y) - r(core_oof, y)

which is a leakage-free, prediction-based measure of how much the nonlinear /
interaction kernels actually buy.  When the score does not exceed
``gate_threshold`` the final model is refitted with the core kernels only, so
a pure-additive dataset can never be degraded by Gaussian/poly/epistatic
kernels (objective 1).  Because dominance is part of the core set the
additive -> dominance transition is always preserved (objective 2), and the
score itself increases with the true epistatic fraction (objective 3).

Cost
----
Kernel construction dominates the cost and depends only on X, but inner fits
re-run variance-component optimisation on sub-matrices.  With the gate enabled
the inner CV fits both the full and the core library, roughly doubling the
inner cost; the affine step alone observed 2.3x-7x overhead on real data.  The
gate is a no-op on already well-calibrated, purely additive data (it returns
the core model, which equals GBLUP under the same additive kernel).
"""

import numpy as np
from scipy.optimize import lsq_linear

from .msbgp import MSBGP
from .cv import _make_folds
from .metrics import pearson_corr

__all__ = ["MSBCal"]

_DEFAULT_K_INNER = 5
_CORE_KERNEL_ALIASES = ("additive", "dominance", "additive_sparse",
                        "add_sparse", "sparse_additive")
_SPARSE_KERNEL_ALIASES = {"additive_sparse", "add_sparse", "sparse_additive"}


def _safe_scale_vec(a):
    """Column scales that are safe for constant columns."""
    a = np.asarray(a, dtype=float)
    scale = np.nanstd(a, axis=0)
    scale = np.asarray(scale, dtype=float)
    scale[~np.isfinite(scale)] = 1.0
    scale[scale < 1e-12] = 1.0
    return scale


def _robust_center_scale(a):
    """Robust center/scale used to align final candidates to their OOF scale."""
    a = np.asarray(a, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    center = float(np.median(a))
    q25, q75 = np.percentile(a, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if (not np.isfinite(scale)) or scale < 1e-12:
        scale = float(np.std(a))
    if (not np.isfinite(scale)) or scale < 1e-12:
        scale = 1.0
    return center, scale


def _match_robust_distribution(source, target):
    """Affine map that matches source center/scale to a target distribution."""
    source = np.asarray(source, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    cs, ss = _robust_center_scale(source)
    ct, st = _robust_center_scale(target)
    alpha = float(st / (ss + 1e-12))
    beta = float(ct - alpha * cs)
    if not np.isfinite(alpha) or alpha <= 0.0:
        alpha = 1.0
        beta = float(ct - cs)
    return alpha, beta


def _fit_nonnegative_stack(A, y, ridge=1e-4):
    """Fit a non-negative, sum-to-one stack on standardized OOF columns.

    The stack is deliberately small and constrained.  Candidate predictions
    are standardized first so that a candidate with a larger raw scale cannot
    dominate only because of units.  The weights are non-negative and
    normalized, which keeps the final predictor as a convex combination of
    MSB-GP candidate models and prevents unstable sign-flipping blends.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if A.ndim == 1:
        A = A[:, None]
    n, k = A.shape
    if n == 0 or k == 0:
        raise ValueError("empty stack design")

    means = np.mean(A, axis=0)
    scales = _safe_scale_vec(A)
    Z = (A - means) / scales
    y_center = float(np.mean(y))
    target = y - y_center
    y_scale = float(np.std(target))
    if not np.isfinite(y_scale) or y_scale < 1e-12:
        y_scale = 1.0
    target = target / y_scale

    ridge = float(max(0.0, ridge))
    if ridge > 0.0:
        Z_fit = np.vstack([Z, np.sqrt(ridge) * np.eye(k, dtype=float)])
        y_fit = np.concatenate([target, np.zeros(k, dtype=float)])
    else:
        Z_fit = Z
        y_fit = target

    try:
        res = lsq_linear(Z_fit, y_fit, bounds=(0.0, np.inf),
                         max_iter=1000, lsmr_tol="auto")
        weights = np.asarray(res.x, dtype=float)
    except Exception:
        weights = np.zeros(k, dtype=float)

    weights[~np.isfinite(weights)] = 0.0
    weights = np.maximum(weights, 0.0)
    if float(np.sum(weights)) <= 1e-12:
        corr = np.empty(k, dtype=float)
        for j in range(k):
            corr[j] = abs(pearson_corr(Z[:, j], target))
        weights = np.zeros(k, dtype=float)
        weights[int(np.nanargmax(corr))] = 1.0
    weights = weights / float(np.sum(weights))
    return means, scales, weights


def _apply_nonnegative_stack(A, means, scales, weights):
    """Apply a fitted standardized non-negative stack."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    scales = np.asarray(scales, dtype=float).copy()
    scales[scales < 1e-12] = 1.0
    Z = (A - np.asarray(means, dtype=float)) / scales
    return Z @ np.asarray(weights, dtype=float).ravel()


def _common_model_kernel_state(model_groups):
    """Return a shared KernelState when every non-empty model group uses it."""
    states = []
    for models in model_groups:
        if not models:
            return None
        local = [getattr(m, "kernel_state_", None) for m in models]
        if any(state is None for state in local):
            return None
        first = local[0]
        if any(state is not first for state in local):
            return None
        states.append(first)
    if not states:
        return None
    first = states[0]
    if any(state is not first for state in states):
        return None
    return first


def _cross_fit_stack_oof(A, y, fold_ids, ridge=1e-4):
    """Cross-fitted stack score and all-data stack parameters."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    fold_ids = np.asarray(fold_ids, dtype=int).ravel()
    if A.ndim == 1:
        A = A[:, None]
    n = len(y)
    oof = np.empty(n, dtype=float)
    unique_folds = np.unique(fold_ids)
    for f in unique_folds:
        test = np.where(fold_ids == f)[0]
        train = np.setdiff1d(np.arange(n), test, assume_unique=False)
        if len(train) < max(20, A.shape[1] + 5):
            continue
        means, scales, weights = _fit_nonnegative_stack(A[train], y[train],
                                                        ridge=ridge)
        oof[test] = _apply_nonnegative_stack(A[test], means, scales, weights)

    if not np.all(np.isfinite(oof)):
        means, scales, weights = _fit_nonnegative_stack(A, y, ridge=ridge)
        oof = _apply_nonnegative_stack(A, means, scales, weights)
    means, scales, weights = _fit_nonnegative_stack(A, y, ridge=ridge)
    return oof, means, scales, weights


def _has_sparse_kernel(spec):
    """Return True when a user kernel specification contains a sparse additive kernel."""
    if isinstance(spec, str):
        return spec.strip().lower() in _SPARSE_KERNEL_ALIASES
    return any(str(k).strip().lower() in _SPARSE_KERNEL_ALIASES for k in spec)


def _as_kernel_list(spec):
    """Normalise a kernel specification to a plain list of names."""
    if isinstance(spec, str):
        return [spec]
    return [str(k) for k in spec]


def _core_kernel_list(spec):
    """Return the biological core kernel names for a user kernel spec.

    The core is {additive, dominance} (plus a sparse-additive variant if the
    user requested one).  Gaussian, polynomial, AA/AD/DD and RSI are treated
    as higher-order kernels that must earn their place through the gate.
    """
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("auto", "", "default"):
            return ["additive", "dominance"]
        return [spec]
    names = [str(x) for x in spec]
    core = [n for n in names if n.lower() in _CORE_KERNEL_ALIASES]
    return core or ["additive", "dominance"]


class MSBCal:
    """MSB-GP with leakage-free affine calibration and a higher-order gate.

    Parameters
    ----------
    kernels : str or list, default="auto"
        Passed to :class:`msbgp.msbgp.MSBGP`.  Use e.g.
        ``["additive", "dominance", "poly"]`` to exclude the Gaussian kernel.
    k_inner : int, default=5
        Number of inner CV folds used to build out-of-fold training
        predictions for the calibration fit (and the gate score).
    seed : int, default=0
        Seed for the inner fold split.
    adaptive_gate : bool, default=True
        When True, enable the prediction-first higher-order gate.  The final
        model falls back to the additive+dominance core whenever the inner
        out-of-fold accuracy is not improved by the full kernel library.
    gate_threshold : float, default=0.0
        Minimum out-of-fold Pearson improvement required to keep the
        higher-order kernels.  ``0.0`` keeps them only when they help at all.

    Attributes
    ----------
    a_, b_ : float
        Learned slope / intercept of the affine map.
    higher_order_score_ : float or None
        ``r(full_oof, y) - r(core_oof, y)`` (None when the gate is disabled).
    gate_passed_ : bool
        Whether the higher-order kernels were retained.
    model_ : MSBGP
        The final model fitted on the full training data.

    Examples
    --------
    >>> from msbgp import MSBCal
    >>> model = MSBCal().fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)          # already calibrated
    >>> model.a_, model.b_                      # inspect the correction
    >>> model.higher_order_score_, model.gate_passed_
    """

    def __init__(self, kernels="auto", k_inner=_DEFAULT_K_INNER, seed=0,
                 n_inner_repeats=1,
                 n_ensemble=1, ensemble_sample_fraction=0.80,
                 residualize=True, rsi=False, rsi_gate=False,
                 epi_kernels=True, epi_top_markers=0, epi_in_core=True,
                 epi_sparse_residual=True,
                 sparse_additive_top_markers="auto",
                 auto_sparse_additive=True,
                 sparse_gate_threshold=0.04,
                 sparse_gate_threshold_high=0.12,
                 sparse_high_dominance_share=0.30,
                 sparse_full_gate_margin=0.005,
                 sparse_additive_min_markers=20,
                 sparse_additive_max_markers=200,
                 sparse_additive_fraction=0.10,
                 sparse_additive_min_samples=300,
                 sparse_additive_min_features=500,
                 kernel_prior="nonlinear", kernel_prior_strength=0.02,
                 kernel_prior_add_share=0.5,
                 kernel_floor=None, kernel_floor_strength=0.05,
                 adaptive_gate=True, gate_threshold=0.0,
                 candidate_stack=True, stack_min_gain=0.005,
                 stack_ridge=1e-4,
                 aux_candidates=(),
                 aux_candidate_min_gain=0.0,
                 aux_candidate_min_score=0.02,
                 aux_candidate_min_samples=250,
                 epi_sparse_top_markers=50,
                 epi_sparse_interactions_per_type=20,
                 epi_sparse_max_terms=60,
                 reml_init="single", reml_multistart=1,
                 reml_max_eval=None, reml_gtol=None):
        self.kernels = kernels
        self.k_inner = k_inner
        self.seed = seed
        self.n_inner_repeats = n_inner_repeats
        self.n_ensemble = n_ensemble
        self.ensemble_sample_fraction = ensemble_sample_fraction
        self.residualize = residualize
        self.rsi = rsi
        self.rsi_gate = rsi_gate
        self.epi_kernels = bool(epi_kernels)
        self.epi_top_markers = int(epi_top_markers)
        self.epi_in_core = bool(epi_in_core)
        self.epi_sparse_residual = bool(epi_sparse_residual)
        self.sparse_additive_top_markers = sparse_additive_top_markers
        self.auto_sparse_additive = bool(auto_sparse_additive)
        self.sparse_gate_threshold = float(sparse_gate_threshold)
        self.sparse_gate_threshold_high = float(sparse_gate_threshold_high)
        self.sparse_high_dominance_share = float(sparse_high_dominance_share)
        self.sparse_full_gate_margin = float(sparse_full_gate_margin)
        self.sparse_additive_min_markers = int(sparse_additive_min_markers)
        self.sparse_additive_max_markers = int(sparse_additive_max_markers)
        self.sparse_additive_fraction = float(sparse_additive_fraction)
        self.sparse_additive_min_samples = int(sparse_additive_min_samples)
        self.sparse_additive_min_features = int(sparse_additive_min_features)
        self.kernel_prior = kernel_prior
        self.kernel_prior_strength = kernel_prior_strength
        self.kernel_prior_add_share = kernel_prior_add_share
        self.kernel_floor = kernel_floor
        self.kernel_floor_strength = kernel_floor_strength
        self.adaptive_gate = bool(adaptive_gate)
        self.gate_threshold = float(gate_threshold)
        self.candidate_stack = bool(candidate_stack)
        self.stack_min_gain = float(stack_min_gain)
        self.stack_ridge = float(stack_ridge)
        self.aux_candidates = aux_candidates
        self.aux_candidate_min_gain = float(aux_candidate_min_gain)
        self.aux_candidate_min_score = float(aux_candidate_min_score)
        self.aux_candidate_min_samples = int(aux_candidate_min_samples)
        self.epi_sparse_top_markers = int(epi_sparse_top_markers)
        self.epi_sparse_interactions_per_type = int(epi_sparse_interactions_per_type)
        self.epi_sparse_max_terms = int(epi_sparse_max_terms)
        self.reml_init = str(reml_init).strip().lower()
        if self.reml_init not in {"single", "heuristic", "multi_start", "multistart", "auto"}:
            raise ValueError(
                "reml_init must be one of 'single', 'heuristic', or 'multi_start'"
            )
        self.reml_multistart = max(1, int(reml_multistart))
        self.reml_max_eval = None if reml_max_eval is None else max(1, int(reml_max_eval))
        self.reml_gtol = None if reml_gtol is None else float(reml_gtol)
        self.epi_sparse_terms_ = []

    # ------------------------------------------------------------------ #
    def _resolved_sparse_top(self, n_markers):
        """Resolve the sparse-additive marker subset size for this training set."""
        spec = self.sparse_additive_top_markers
        if spec is None or spec is False:
            return 0
        if isinstance(spec, str):
            key = spec.strip().lower()
            if key in {"", "none", "off", "false", "0"}:
                return 0
            if key in {"auto", "default"}:
                frac = float(np.clip(self.sparse_additive_fraction, 0.0, 1.0))
                top = int(round(frac * int(n_markers)))
                top = max(int(self.sparse_additive_min_markers), top)
                top = min(int(self.sparse_additive_max_markers), top)
                return int(min(int(n_markers), max(0, top)))
            try:
                spec = int(float(key))
            except ValueError as exc:
                raise ValueError(
                    "sparse_additive_top_markers must be an int, 'auto', or None"
                ) from exc
        top = int(spec)
        if top <= 0:
            return 0
        return int(min(int(n_markers), top))

    def _auto_sparse_eligible(self, n_samples, n_features, sparse_top):
        """Guard the automatic sparse rescue against small-sample overfitting.

        The sparse-additive candidate screens markers on the training fold.  On
        small pilot datasets (for example n < 300 and m < 500), the selected
        markers can be fold-specific noise, so the rescue must be opt-in rather
        than automatic.  Users can still request it explicitly through
        ``kernels=["additive", "additive_sparse"]`` and a fixed marker count.
        """
        return bool(
            int(sparse_top) > 0 and
            int(n_samples) >= self.sparse_additive_min_samples and
            int(n_features) >= self.sparse_additive_min_features
        )

    def _resolved_aux_candidates(self):
        """Normalise the internal candidate-bank names.

        Auxiliary model calls are opt-in only and disabled by default.  The
        standard MSB-GP algorithm is self-contained; when explicitly enabled,
        these candidates are treated as external benchmark/fallback models and
        are not part of the native multi-kernel fit.
        """
        spec = self.aux_candidates
        if spec is None or spec is False:
            return []
        if isinstance(spec, str):
            spec = [s.strip() for s in spec.split(",") if s.strip()]
        aliases = {
            "gblup": "gblup",
            "rr": "rr_blup",
            "rr-blup": "rr_blup",
            "rr_blup": "rr_blup",
            "rrblup": "rr_blup",
            "mean": "mean_gmkl",
            "mean_gmkl": "mean_gmkl",
            "mean-gmkl": "mean_gmkl",
            "fh": "fh_gmkl",
            "fh_gmkl": "fh_gmkl",
            "fh-gmkl": "fh_gmkl",
            "cka": "cka_gmkl",
            "cka_gmkl": "cka_gmkl",
            "cka-gmkl": "cka_gmkl",
        }
        out = []
        for name in spec:
            key = str(name).strip().lower()
            key = aliases.get(key, key)
            if key and key not in out:
                out.append(key)
        return out

    @staticmethod
    def _make_aux_candidate(name):
        """Instantiate one lightweight candidate from the internal model bank."""
        key = str(name).strip().lower()
        if key == "gblup":
            from .bases import GBLUP
            return GBLUP()
        if key == "rr_blup":
            from .bases import RRBLUP
            return RRBLUP()
        if key == "mean_gmkl":
            from .gmkls import MeanGMKL
            return MeanGMKL()
        if key == "fh_gmkl":
            from .gmkls import FHGMKL
            return FHGMKL()
        if key == "cka_gmkl":
            from .gmkls import CKAGMKL
            return CKAGMKL()
        raise ValueError(f"Unknown internal MSB-GP auxiliary candidate: {name}")

    def _fit_aux_oof(self, X, y, seed):
        """Leakage-free inner-fold OOF predictions for auxiliary candidates."""
        names = self._resolved_aux_candidates()
        if not names:
            return {}
        n = len(y)
        folds = _make_folds(n, self.k_inner, seed)
        oof = {name: np.empty(n, dtype=float) for name in names}
        for order in folds:
            test = np.sort(order)
            train = np.setdiff1d(np.arange(n), test)
            for name in names:
                try:
                    m = self._make_aux_candidate(name)
                    m.fit(X[train], y[train])
                    oof[name][test] = np.asarray(m.predict(X[test]), dtype=float).ravel()
                except Exception:
                    oof[name][test] = np.nan
        out = {}
        for name, arr in oof.items():
            if np.all(np.isfinite(arr)):
                out[name] = arr
        return out

    @staticmethod
    def _fit_aux_final(name, X, y):
        m = MSBCal._make_aux_candidate(name)
        m.fit(X, y)
        return m

    def _select_sparse_epi_terms(self, X_train, y_train, core_model):
        """Select sparse AA/AD/DD terms from training-only core residuals."""
        from .rsi import build_rsi_terms
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).ravel()
        pred = np.asarray(core_model.predict(X_train), dtype=float).ravel()
        residual = y_train - pred
        return build_rsi_terms(
            X_train, residual,
            top_markers=min(int(self.epi_sparse_top_markers), X_train.shape[1]),
            interactions_per_type=int(self.epi_sparse_interactions_per_type),
            max_terms=int(self.epi_sparse_max_terms),
            random_state=int(self.seed),
        )

    def _new_base(self, core_only=False, sparse_variant=False,
                  epi_variant=False, sparse_top=0, epi_terms=None):
        if core_only:
            kernels = _core_kernel_list(self.kernels)
            epi_kernels = False
            rsi = False
            pass_sparse_top = 0
        elif sparse_variant and epi_variant:
            kernels = ["additive", "additive_sparse", "dominance",
                       "AA", "AD", "DD"]
            epi_kernels = False
            rsi = False
            pass_sparse_top = int(sparse_top or 0)
        elif epi_variant:
            # Native higher-order candidate: additive + dominance + explicit
            # AA/AD/DD epistasis only.  This lets the leakage-free selector
            # recognise genuine interaction signal without relying on the
            # flexible Gaussian/Poly approximations as the only route to
            # higher-order variation.
            kernels = ["additive", "dominance", "AA", "AD", "DD"]
            epi_kernels = False
            rsi = False
            pass_sparse_top = 0
        elif sparse_variant:
            bases = _as_kernel_list(self.kernels)
            if not _has_sparse_kernel(bases):
                insert_at = 1 if bases and bases[0].lower() == "additive" else len(bases)
                bases = bases[:insert_at] + ["additive_sparse"] + bases[insert_at:]
            kernels = bases
            # For the public ``auto`` selector, evaluate sparse additive as a
            # conservative core extension.  Nonlinear and epistasis kernels are
            # kept in the competing full model instead, so sparse additive is
            # selected only when it improves accuracy without relying on the
            # high-variance kernels that the higher-order gate may reject.
            if str(self.kernels).strip().lower() in {"auto", "default", ""}:
                kernels = ["additive", "additive_sparse", "dominance"]
            epi_kernels = False
            rsi = self.rsi
            pass_sparse_top = int(sparse_top or 0)
        else:
            kernels = self.kernels
            epi_kernels = self.epi_kernels
            rsi = self.rsi
            pass_sparse_top = int(sparse_top or 0) if _has_sparse_kernel(self.kernels) else 0
        return MSBGP(kernels=kernels, residualize=self.residualize,
                     rsi=rsi, rsi_gate=(self.rsi_gate if rsi else False),
                     epi_kernels=epi_kernels, epi_in_core=self.epi_in_core,
                     epi_top_markers=self.epi_top_markers,
                     epi_terms=epi_terms,
                     sparse_additive_top_markers=pass_sparse_top,
                     kernel_prior=self.kernel_prior,
                     kernel_prior_strength=self.kernel_prior_strength,
                     kernel_prior_add_share=self.kernel_prior_add_share,
                     kernel_floor=self.kernel_floor,
                     kernel_floor_strength=self.kernel_floor_strength,
                     reml_init=self.reml_init,
                     reml_multistart=self.reml_multistart,
                     reml_max_eval=self.reml_max_eval,
                     reml_gtol=self.reml_gtol)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _affine_coeffs(oof, y):
        """Rank-preserving affine coefficients from out-of-fold predictions."""
        n = len(y)
        A = np.vstack([np.asarray(oof, dtype=float), np.ones(n)]).T
        a, b = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)[0]
        a = float(a)
        b = float(b)
        # Rank-preserving guard.  When a kernel is near-singular REML can
        # inflate the raw prediction scale so much that the OLS slope collapses
        # toward zero (or turns negative), destroying the absolute scale and
        # MSE while leaving only a faint ranking signal.  In that regime fall
        # back to a variance-matching (still strictly rank-preserving, a > 0)
        # affine map instead of the degenerate OLS solution.
        a_var = float(np.std(y) / (np.std(oof) + 1e-12))
        if (not np.isfinite(a)) or a <= 0.0 or a < 0.05 * a_var:
            a = a_var
            b = float(np.mean(y) - a * np.mean(oof))
        return a, b

    # ------------------------------------------------------------------ #
    def _fit_calibration_once(self, X, y, seed, sparse_top=0):
        """One leakage-free inner CV.

        Returns ``(a, b, oof, oof_core, oof_sparse, oof_epi,
        core_dom_share, fold_ids)`` where ``oof_core`` is the core-only
        out-of-fold prediction, ``oof_sparse`` is the sparse-additive candidate
        prediction, and ``oof_epi`` is the explicit AA/AD/DD epistasis
        candidate prediction.  Each candidate can be ``None`` when the
        corresponding gate is disabled.  ``core_dom_share`` is the mean
        dominance mass of the core model and is used to prevent the
        sparse-additive rescue from replacing a genuine dominance/epistasis
        signal.
        """
        n = len(y)
        folds = _make_folds(n, self.k_inner, seed)
        want_core = bool(self.adaptive_gate)
        auto_kernel_spec = str(self.kernels).strip().lower() in {"auto", "default", ""}
        want_sparse = bool(
            self.adaptive_gate and self.auto_sparse_additive and
            auto_kernel_spec and int(sparse_top or 0) > 0
        )
        want_epi = bool(
            self.adaptive_gate and self.epi_kernels and auto_kernel_spec
        )
        oof = np.empty(n)
        oof_core = np.empty(n) if want_core else None
        oof_sparse = np.empty(n) if want_sparse else None
        oof_epi = np.empty(n) if want_epi else None
        fold_ids = np.empty(n, dtype=int)
        core_dom_shares = []
        for fold_idx, order in enumerate(folds):
            test = np.sort(order)
            fold_ids[test] = int(fold_idx)
            train = np.setdiff1d(np.arange(n), test)
            # Build the training-only kernel state once per inner fold and
            # reuse it across full/core/sparse-additive variants.  These models
            # differ in which kernels receive REML weights, but their allele
            # frequencies, normalisers, genotype matrices, and optional marker
            # screening are identical on a fixed training split.
            state_builder = (
                self._new_base(sparse_variant=True, sparse_top=sparse_top)
                if want_sparse else
                self._new_base(core_only=False, sparse_top=sparse_top)
            )
            shared_state = state_builder._build_kernel_state(X[train], y[train])
            shared_state.enable_training_kernel_cache()
            try:
                m = self._new_base(core_only=False, sparse_top=sparse_top)
                m.fit(X[train], y[train], _kernel_state=shared_state)
                oof[test] = np.asarray(m.predict(X[test])).ravel()
                if want_core:
                    mc = self._new_base(core_only=True, sparse_top=0)
                    mc.fit(X[train], y[train], _kernel_state=shared_state)
                    oof_core[test] = np.asarray(mc.predict(X[test])).ravel()
                    try:
                        w_core = mc.get_kernel_weights()
                        total = float(sum(max(0.0, float(v)) for v in w_core.values()))
                        if total > 0.0:
                            core_dom_shares.append(float(w_core.get("dominance", 0.0)) / total)
                    except Exception:
                        pass
                if want_sparse:
                    ms = self._new_base(sparse_variant=True, sparse_top=sparse_top)
                    ms.fit(X[train], y[train], _kernel_state=shared_state)
                    oof_sparse[test] = np.asarray(ms.predict(X[test])).ravel()
                if want_epi:
                    if self.epi_sparse_residual:
                        try:
                            core_for_epi = mc
                        except NameError:
                            core_for_epi = self._new_base(core_only=True, sparse_top=0)
                            core_for_epi.fit(X[train], y[train], _kernel_state=shared_state)
                        terms = self._select_sparse_epi_terms(
                            X[train], y[train], core_for_epi)
                        epi_state = state_builder._build_kernel_state(
                            X[train], y[train])
                        epi_state.set_epi_terms(terms)
                        epi_state.enable_training_kernel_cache()
                        try:
                            me = self._new_base(epi_variant=True, epi_terms=terms)
                            me.fit(X[train], y[train], _kernel_state=epi_state)
                            oof_epi[test] = np.asarray(me.predict(X[test])).ravel()
                        finally:
                            epi_state.clear_training_kernel_cache()
                    else:
                        me = self._new_base(epi_variant=True)
                        me.fit(X[train], y[train], _kernel_state=shared_state)
                        oof_epi[test] = np.asarray(me.predict(X[test])).ravel()
            finally:
                shared_state.clear_training_kernel_cache()
        a, b = self._affine_coeffs(oof, y)
        core_dom_share = float(np.mean(core_dom_shares)) if core_dom_shares else 0.0
        return (a, b, oof, oof_core, oof_sparse, oof_epi,
                core_dom_share, fold_ids)

    # ------------------------------------------------------------------ #
    def _fit_final(self, X, y, core_only, sparse_variant=False,
                   epi_variant=False, sparse_top=0, epi_terms=None,
                   _kernel_state=None):
        """Fit the final (possibly bagged) model on all training data."""
        models = []
        if int(self.n_ensemble) > 1:
            rng = np.random.default_rng(int(self.seed))
            n_sub = max(10, int(round(len(y) * float(self.ensemble_sample_fraction))))
            for _ in range(int(self.n_ensemble)):
                idx = np.sort(rng.choice(len(y), size=n_sub, replace=False))
                m = self._new_base(core_only=core_only,
                                   sparse_variant=sparse_variant,
                                   epi_variant=epi_variant,
                                   sparse_top=sparse_top,
                                   epi_terms=epi_terms)
                # Bagged members see different subsamples, so they cannot share
                # a precomputed KernelState.
                m.fit(X[idx], y[idx])
                models.append(m)
        if models:
            self.ensemble_models_ = models
            self.model_ = models[0]
        else:
            self.ensemble_models_ = []
            self.model_ = self._new_base(core_only=core_only,
                                         sparse_variant=sparse_variant,
                                         epi_variant=epi_variant,
                                         sparse_top=sparse_top,
                                         epi_terms=epi_terms)
            self.model_.fit(X, y, _kernel_state=_kernel_state)

    def _fit_final_candidates(self, X, y, variants, sparse_top=0):
        """Fit final models for the internal candidate stack.

        The public model remains MSB-GP.  This helper keeps ``core``,
        ``sparse``, ``epi`` and ``full`` native candidate predictors alive so
        their OOF-learned convex weights can be applied at prediction time
        without leakage.
        """
        candidate_models = {}
        shared_state = None
        aux_names = set(self._resolved_aux_candidates())
        variants = [str(v) for v in variants]
        msb_variants = [v for v in variants if v not in aux_names]
        # Keep one native MSB-GP fit alive for kernel-weight diagnostics even
        # when the OOF selector chooses an auxiliary candidate as the deployed
        # predictor.
        diagnostic_variants = list(msb_variants)
        if not diagnostic_variants:
            diagnostic_variants = ["full"]
        epi_terms = None
        epi_state = None
        if int(self.n_ensemble) <= 1 and diagnostic_variants:
            state_builder = (
                self._new_base(
                    sparse_variant=("sparse" in diagnostic_variants),
                    epi_variant=("epi" in diagnostic_variants),
                    sparse_top=sparse_top)
            )
            shared_state = state_builder._build_kernel_state(X, y)
        if "epi" in diagnostic_variants and self.epi_sparse_residual:
            core_state = self._new_base(core_only=True)._build_kernel_state(X, y)
            core_for_epi = self._new_base(core_only=True)
            core_for_epi.fit(X, y, _kernel_state=core_state)
            epi_terms = self._select_sparse_epi_terms(X, y, core_for_epi)
            self.epi_sparse_terms_ = [tuple(t) for t in epi_terms]
            if int(self.n_ensemble) <= 1:
                epi_state = self._new_base(
                    epi_variant=True, epi_terms=epi_terms)._build_kernel_state(X, y)
                epi_state.set_epi_terms(epi_terms)
        try:
            if shared_state is not None:
                shared_state.enable_training_kernel_cache()
            for variant in variants:
                if variant in aux_names:
                    m = self._fit_aux_final(variant, X, y)
                    candidate_models[str(variant)] = [m]
                else:
                    self._fit_final(
                        X, y,
                        core_only=(variant == "core"),
                        sparse_variant=(variant == "sparse"),
                        epi_variant=(variant == "epi"),
                        sparse_top=(int(sparse_top) if variant == "sparse" else 0),
                        epi_terms=(epi_terms if variant == "epi" else None),
                        _kernel_state=(epi_state if variant == "epi" else shared_state),
                    )
                    candidate_models[str(variant)] = (list(self.ensemble_models_)
                                                      if self.ensemble_models_
                                                      else [self.model_])
            for variant in diagnostic_variants:
                if variant in candidate_models:
                    continue
                self._fit_final(
                    X, y,
                    core_only=(variant == "core"),
                    sparse_variant=(variant == "sparse"),
                    epi_variant=(variant == "epi"),
                    sparse_top=(int(sparse_top) if variant == "sparse" else 0),
                    epi_terms=(epi_terms if variant == "epi" else None),
                    _kernel_state=(epi_state if variant == "epi" else shared_state),
                )
                candidate_models[str(variant)] = (list(self.ensemble_models_)
                                                  if self.ensemble_models_
                                                  else [self.model_])
        finally:
            if shared_state is not None:
                shared_state.clear_training_kernel_cache()
            if epi_state is not None:
                epi_state.clear_training_kernel_cache()
        master = None
        for name in ("full", "sparse", "core"):
            if name in candidate_models:
                master = candidate_models[name]
                break
        if master:
            self.model_ = master[0]
            self.ensemble_models_ = list(master)
        self.candidate_models_ = candidate_models
        return self

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """Fit the base model with calibration, gate and candidate stacking."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        reps = max(1, int(self.n_inner_repeats))
        sparse_top = self._resolved_sparse_top(X.shape[1])
        auto_kernel_spec = str(self.kernels).strip().lower() in {"auto", "default", ""}
        if auto_kernel_spec and self.auto_sparse_additive:
            if not self._auto_sparse_eligible(X.shape[0], X.shape[1], sparse_top):
                sparse_top = 0

        a_list, b_list = [], []
        full_oofs, core_oofs, sparse_oofs, epi_oofs = [], [], [], []
        # Auxiliary model calls are explicitly opt-in.  In the standard
        # algorithm `aux_candidates=()` so the final model is formed only from
        # native MSB-GP core/sparse/full candidates, with no internal GBLUP,
        # RR-BLUP or GMKL call.
        resolved_aux = (self._resolved_aux_candidates()
                        if self.candidate_stack and len(y) >= self.aux_candidate_min_samples
                        else ())
        aux_oofs = {name: [] for name in resolved_aux}
        fold_id_list = []
        core_dom_share_values = []
        for rep in range(reps):
            seed = int(self.seed) + rep * 1000
            (a, b, oof, oof_core, oof_sparse, oof_epi,
             core_dom_share, fold_ids) = self._fit_calibration_once(
                X, y, seed, sparse_top=sparse_top)
            if self.candidate_stack and aux_oofs:
                aux_rep = self._fit_aux_oof(X, y, seed)
                for name in aux_oofs:
                    if name in aux_rep:
                        aux_oofs[name].append(aux_rep[name])
            core_dom_share_values.append(core_dom_share)
            a_list.append(a)
            b_list.append(b)
            full_oofs.append(oof)
            if oof_core is not None:
                core_oofs.append(oof_core)
            if oof_sparse is not None:
                sparse_oofs.append(oof_sparse)
            if oof_epi is not None:
                epi_oofs.append(oof_epi)
            fold_id_list.append(fold_ids)

        self.higher_order_score_ = None
        self.gate_passed_ = True
        self.gate_dropped_kernels_ = []
        self.sparse_additive_selected_ = False
        self.sparse_additive_score_ = None
        self.epi_candidate_score_ = None
        self.epi_selected_ = False
        self.epi_sparse_terms_ = []
        self.sparse_additive_top_markers_used_ = int(sparse_top)
        self.core_dominance_share_ = None
        self.sparse_gate_threshold_used_ = float(self.sparse_gate_threshold)
        self.final_model_variant_ = "full"
        self.candidate_stack_enabled_ = False
        self.candidate_stack_names_ = []
        self.candidate_stack_weights_ = None
        self.candidate_stack_means_ = None
        self.candidate_stack_scales_ = None
        self.candidate_stack_score_ = None
        self.candidate_stack_best_single_score_ = None
        self.candidate_stack_gain_ = None
        self.candidate_stack_cv_oof_ = None
        self.candidate_stack_affine_ = {}
        self.candidate_stack_align_ = {}
        self.aux_candidate_names_ = list(aux_oofs.keys())
        self.aux_candidate_scores_ = {}
        self.aux_candidate_selected_ = False

        higher_order = None
        core_r = None
        full_r = None
        if self.adaptive_gate and core_oofs:
            full_r = float(np.mean([pearson_corr(o, y) for o in full_oofs]))
            core_r = float(np.mean([pearson_corr(c, y) for c in core_oofs]))
            best_r = full_r
            best_oofs = full_oofs
            full_gate_score = float(full_r - core_r)
            if epi_oofs:
                epi_r = float(np.mean([pearson_corr(o, y) for o in epi_oofs]))
                self.epi_candidate_score_ = float(epi_r - core_r)
                if epi_r > best_r:
                    best_r = epi_r
                    best_oofs = epi_oofs
                    self.epi_selected_ = True
            if sparse_oofs:
                sparse_r = float(np.mean([pearson_corr(o, y)
                                          for o in sparse_oofs]))
                self.sparse_additive_score_ = float(sparse_r - full_r)
                mean_core_dom = (float(np.mean(core_dom_share_values))
                                 if core_dom_share_values else 0.0)
                self.core_dominance_share_ = mean_core_dom
                sparse_threshold = (
                    self.sparse_gate_threshold_high
                    if mean_core_dom >= self.sparse_high_dominance_share
                    else self.sparse_gate_threshold
                )
                self.sparse_gate_threshold_used_ = float(sparse_threshold)
                if (full_gate_score <= self.sparse_full_gate_margin and
                        self.sparse_additive_score_ > sparse_threshold and
                        (sparse_r - core_r) > self.gate_threshold):
                    self.sparse_additive_selected_ = True
                    best_r = sparse_r
                    best_oofs = sparse_oofs
                    self.epi_selected_ = False
            higher_order = best_r - core_r
            self.higher_order_score_ = float(higher_order)
            self.gate_passed_ = bool(higher_order > self.gate_threshold)
            if not self.gate_passed_:
                core_a, core_b = [], []
                for c in core_oofs:
                    ca, cb = self._affine_coeffs(c, y)
                    core_a.append(ca)
                    core_b.append(cb)
                a_list, b_list = core_a, core_b
                self.gate_dropped_kernels_ = ["gaussian", "poly", "AA", "AD",
                                             "DD", "RSI", "additive_sparse"]
                self.sparse_additive_selected_ = False
            else:
                chosen_a, chosen_b = [], []
                for o in best_oofs:
                    ca, cb = self._affine_coeffs(o, y)
                    chosen_a.append(ca)
                    chosen_b.append(cb)
                a_list, b_list = chosen_a, chosen_b

        # Internal candidate stacking.  This remains MSB-GP: the stack is a
        # leakage-free ensemble of the model's own core/sparse/full candidates.
        # Each candidate is first mapped to phenotype scale using its own OOF
        # calibration; this avoids a candidate with a numerically inflated raw
        # kernel scale dominating the stack at prediction time.
        if self.candidate_stack and full_oofs and core_oofs:
            candidate_names, candidate_cols = [], []
            if sparse_oofs:
                candidate_names.append("sparse")
                candidate_cols.append(np.mean(np.vstack(sparse_oofs), axis=0))
            if epi_oofs:
                candidate_names.append("epi")
                candidate_cols.append(np.mean(np.vstack(epi_oofs), axis=0))
            candidate_names.append("core")
            candidate_cols.append(np.mean(np.vstack(core_oofs), axis=0))
            candidate_names.append("full")
            candidate_cols.append(np.mean(np.vstack(full_oofs), axis=0))
            for name, reps_oof in aux_oofs.items():
                if reps_oof:
                    candidate_names.append(name)
                    candidate_cols.append(np.mean(np.vstack(reps_oof), axis=0))

            candidate_affine = {}
            calibrated_cols = []
            for name, col in zip(candidate_names, candidate_cols):
                ca_j, cb_j = self._affine_coeffs(col, y)
                candidate_affine[name] = (float(ca_j), float(cb_j))
                calibrated_cols.append(ca_j * np.asarray(col, dtype=float) + cb_j)
            A = np.column_stack(calibrated_cols)
            fold_ids = (fold_id_list[0] if fold_id_list else
                        np.zeros(len(y), dtype=int))
            stack_oof, means, scales, weights = _cross_fit_stack_oof(
                A, y, fold_ids, ridge=self.stack_ridge)
            single_scores = [pearson_corr(A[:, j], y) for j in range(A.shape[1])]
            best_single_r = float(np.max(single_scores))
            stack_r = float(pearson_corr(stack_oof, y))
            gain = float(stack_r - best_single_r)
            self.candidate_stack_score_ = stack_r
            self.candidate_stack_best_single_score_ = best_single_r
            self.candidate_stack_gain_ = gain
            self.candidate_stack_cv_oof_ = stack_oof
            for name, score in zip(candidate_names, single_scores):
                if name in self.aux_candidate_names_:
                    self.aux_candidate_scores_[name] = float(score)
            best_single_idx = int(np.nanargmax(single_scores))
            best_single_name = candidate_names[best_single_idx]
            msb_names = {"sparse", "epi", "core", "full"}
            msb_scores = [single_scores[i] for i, name in enumerate(candidate_names)
                          if name in msb_names]
            best_msb_r = float(max(msb_scores)) if msb_scores else -np.inf
            force_aux_single = bool(
                best_single_name in self.aux_candidate_names_ and
                best_single_r >= self.aux_candidate_min_score and
                best_single_r >= best_msb_r + self.aux_candidate_min_gain
            )
            should_enable = bool(
                len(candidate_names) >= 2 and
                (gain >= self.stack_min_gain or force_aux_single)
            )
            if should_enable:
                if force_aux_single and gain < self.stack_min_gain:
                    weights = np.zeros(len(candidate_names), dtype=float)
                    weights[best_single_idx] = 1.0
                    means = np.mean(A, axis=0)
                    scales = _safe_scale_vec(A)
                    stack_oof = A[:, best_single_idx]
                    stack_r = float(single_scores[best_single_idx])
                    self.aux_candidate_selected_ = True
                active = np.where(np.asarray(weights, dtype=float) > 1e-10)[0]
                if active.size == 0:
                    active = np.asarray([best_single_idx], dtype=int)
                active = list(active)
                active_weights = np.asarray(weights, dtype=float)[active]
                active_weights = active_weights / max(float(active_weights.sum()), 1e-12)
                self.candidate_stack_enabled_ = True
                self.candidate_stack_names_ = [candidate_names[i] for i in active]
                self.candidate_stack_weights_ = active_weights
                self.candidate_stack_means_ = np.asarray(means, dtype=float)[active]
                self.candidate_stack_scales_ = np.asarray(scales, dtype=float)[active]
                self.candidate_stack_affine_ = {candidate_names[i]: candidate_affine[candidate_names[i]]
                                                for i in active}
                ca, cb = self._affine_coeffs(stack_oof, y)
                a_list, b_list = [ca], [cb]
                self.final_model_variant_ = "candidate_stack"
                self.gate_passed_ = True
                if core_r is not None:
                    self.higher_order_score_ = float(stack_r - core_r)

        self.a_ = float(np.mean(a_list))
        self.b_ = float(np.mean(b_list))
        self.calibration_a_values_ = a_list
        self.calibration_b_values_ = b_list

        if self.candidate_stack_enabled_:
            self._fit_final_candidates(X, y, self.candidate_stack_names_,
                                       sparse_top=sparse_top)
            # Final models are fitted on the full training set, whereas the
            # candidate stack weights were learned from inner-fold OOF columns.
            # Align each final candidate's training distribution to its own OOF
            # distribution before applying the OOF affine map.  This is a
            # leakage-free scale stabiliser and prevents heavy-tailed final
            # candidates from destabilising the convex stack.
            alignments = {}
            active_name_set = set(self.candidate_stack_names_)
            candidate_col_by_name = {name: col for name, col in zip(candidate_names, candidate_cols)}
            cross_cache_state = _common_model_kernel_state([
                self.candidate_models_.get(name, [])
                for name in self.candidate_stack_names_
            ])
            if cross_cache_state is not None:
                cross_cache_state.enable_cross_kernel_cache()
            try:
                for name in self.candidate_stack_names_:
                    raw_oof = candidate_col_by_name[name]
                    models = self.candidate_models_.get(name, [])
                    if not models:
                        alignments[name] = (1.0, 0.0)
                        continue
                    raw_train = np.mean([np.asarray(mm.predict(X)).ravel()
                                         for mm in models], axis=0)
                    alignments[name] = _match_robust_distribution(raw_train, raw_oof)
            finally:
                if cross_cache_state is not None:
                    cross_cache_state.clear_cross_kernel_cache()
            self.candidate_stack_align_ = alignments
        else:
            core_only = bool(self.adaptive_gate and not self.gate_passed_)
            sparse_variant = bool((not core_only) and self.sparse_additive_selected_)
            epi_variant = bool((not core_only) and (not sparse_variant) and
                               self.epi_selected_)
            self.final_model_variant_ = (
                "core" if core_only else
                "sparse_additive" if sparse_variant else
                "epi" if epi_variant else "full"
            )
            epi_terms = None
            epi_state = None
            if epi_variant and self.epi_sparse_residual:
                core_for_epi = self._new_base(core_only=True).fit(X, y)
                epi_terms = self._select_sparse_epi_terms(X, y, core_for_epi)
                self.epi_sparse_terms_ = [tuple(t) for t in epi_terms]
                epi_state = self._new_base(
                    epi_variant=True, epi_terms=epi_terms)._build_kernel_state(X, y)
                epi_state.set_epi_terms(epi_terms)
            self._fit_final(X, y, core_only=core_only,
                            sparse_variant=sparse_variant,
                            epi_variant=epi_variant,
                            sparse_top=sparse_top,
                            epi_terms=epi_terms,
                            _kernel_state=epi_state)
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X):
        """Return affine-calibrated MSB-GP predictions."""
        # Convert once so repeated candidate predictions can share cross-kernel
        # cache entries keyed by the identity of the same validation matrix.
        X = np.asarray(X, dtype=float)
        if getattr(self, "candidate_stack_enabled_", False):
            cols = []
            affine = getattr(self, "candidate_stack_affine_", {})
            align = getattr(self, "candidate_stack_align_", {})
            entries = []
            for name in self.candidate_stack_names_:
                models = getattr(self, "candidate_models_", {}).get(name, [])
                if models:
                    entries.append((name, models))

            cache_state = _common_model_kernel_state(
                [models for _, models in entries])
            if cache_state is not None:
                cache_state.enable_cross_kernel_cache()
            try:
                for name, models in entries:
                    raw = np.mean([m.predict(X) for m in models], axis=0)
                    alpha, beta = align.get(name, (1.0, 0.0))
                    raw = alpha * np.asarray(raw, dtype=float).ravel() + beta
                    ca_j, cb_j = affine.get(name, (1.0, 0.0))
                    cols.append(ca_j * raw + cb_j)
            finally:
                if cache_state is not None:
                    cache_state.clear_cross_kernel_cache()
            if cols:
                raw = _apply_nonnegative_stack(
                    np.column_stack(cols),
                    self.candidate_stack_means_,
                    self.candidate_stack_scales_,
                    self.candidate_stack_weights_,
                )
            else:
                raw = np.asarray(self.model_.predict(X)).ravel()
        elif getattr(self, "ensemble_models_", None):
            raw = np.mean([m.predict(X) for m in self.ensemble_models_], axis=0)
        else:
            raw = np.asarray(self.model_.predict(X)).ravel()
        return self.a_ * np.asarray(raw).ravel() + self.b_

    # ------------------------------------------------------------------ #
    def get_higher_order_score(self):
        """Return the out-of-fold higher-order score, or None if disabled."""
        return getattr(self, "higher_order_score_", None)

    def get_gate_diagnostics(self):
        """Return a small dict describing the higher-order gate decision."""
        return {
            "adaptive_gate": bool(self.adaptive_gate),
            "gate_threshold": float(self.gate_threshold),
            "higher_order_score": getattr(self, "higher_order_score_", None),
            "gate_passed": bool(getattr(self, "gate_passed_", True)),
            "dropped_kernels": list(getattr(self, "gate_dropped_kernels_", [])),
            "sparse_additive_enabled": bool(
                getattr(self, "auto_sparse_additive", False)),
            "sparse_additive_selected": bool(
                getattr(self, "sparse_additive_selected_", False)),
            "sparse_additive_score": getattr(
                self, "sparse_additive_score_", None),
            "sparse_additive_top_markers_used": int(
                getattr(self, "sparse_additive_top_markers_used_", 0)),
            "epi_candidate_score": getattr(self, "epi_candidate_score_", None),
            "epi_selected": bool(getattr(self, "epi_selected_", False)),
            "core_dominance_share": getattr(
                self, "core_dominance_share_", None),
            "sparse_gate_threshold_used": float(
                getattr(self, "sparse_gate_threshold_used_",
                        self.sparse_gate_threshold)),
            "sparse_full_gate_margin": float(
                getattr(self, "sparse_full_gate_margin", 0.0)),
            "candidate_stack_enabled": bool(
                getattr(self, "candidate_stack_enabled_", False)),
            "candidate_stack_names": list(
                getattr(self, "candidate_stack_names_", [])),
            "candidate_stack_weights": (None if getattr(self, "candidate_stack_weights_", None) is None
                                        else [float(v) for v in self.candidate_stack_weights_]),
            "candidate_stack_score": getattr(
                self, "candidate_stack_score_", None),
            "candidate_stack_best_single_score": getattr(
                self, "candidate_stack_best_single_score_", None),
            "candidate_stack_gain": getattr(
                self, "candidate_stack_gain_", None),
            "aux_candidate_names": list(getattr(self, "aux_candidate_names_", [])),
            "aux_candidate_scores": {
                str(k): float(v)
                for k, v in getattr(self, "aux_candidate_scores_", {}).items()
            },
            "aux_candidate_selected": bool(
                getattr(self, "aux_candidate_selected_", False)),
            "aux_candidate_min_score": float(
                getattr(self, "aux_candidate_min_score", 0.0)),
            "aux_candidate_min_samples": int(
                getattr(self, "aux_candidate_min_samples", 0)),
            "candidate_stack_affine": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in getattr(self, "candidate_stack_affine_", {}).items()
            },
            "candidate_stack_align": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in getattr(self, "candidate_stack_align_", {}).items()
            },
            "final_model_variant": getattr(
                self, "final_model_variant_", None),
            "sparse_additive_min_samples": int(
                getattr(self, "sparse_additive_min_samples", 0)),
            "sparse_additive_min_features": int(
                getattr(self, "sparse_additive_min_features", 0)),
        }

    # ------------------------------------------------------------------ #
    def get_sparse_epistasis_diagnostics(self):
        """Return the native residual sparse AA/AD/DD selection summary.

        The returned terms are learned from training-fold core residuals only.
        They are used by the native ``epi`` candidate when
        ``epi_sparse_residual=True``.  No external predictor is involved.
        """
        terms = [tuple(t) for t in getattr(self, "epi_sparse_terms_", [])]
        counts = {
            name: sum(1 for t in terms if t and str(t[0]).upper() == name)
            for name in ("AA", "AD", "DD")
        }
        return {
            "enabled": bool(getattr(self, "epi_sparse_residual", False)),
            "selected": bool(getattr(self, "epi_selected_", False)),
            "n_terms": int(len(terms)),
            "counts": counts,
            "terms": terms,
            "top_markers": int(getattr(self, "epi_sparse_top_markers", 0)),
            "interactions_per_type": int(
                getattr(self, "epi_sparse_interactions_per_type", 0)),
            "max_terms": int(getattr(self, "epi_sparse_max_terms", 0)),
        }

    def get_architecture_diagnostics(self):
        """Return all compact interpretability diagnostics for the fitted model.

        This keeps downstream benchmark/reporting code from having to know the
        individual accessor calls.  The sparse AA/AD/DD terms are included here
        alongside kernel weights, the higher-order gate, and the gate score.
        """
        diag = {
            "kernel_weights": self.get_kernel_weights(),
            "higher_order_score": self.get_higher_order_score(),
        }
        try:
            diag["gate"] = self.get_gate_diagnostics()
        except Exception as exc:
            diag["gate"] = {}
            diag["gate_error"] = str(exc)
        try:
            diag["sparse_epistasis"] = self.get_sparse_epistasis_diagnostics()
        except Exception as exc:
            diag["sparse_epistasis"] = {}
            diag["sparse_epi_error"] = str(exc)
        return diag

    def get_rsi_stability(self):
        """Return strict RSI stability diagnostics for the MSBCal workflow.

        For the homogeneous bagged ensemble, per-model diagnostics are
        retained and frequencies are averaged across base models.  A term is
        listed in ``selected_terms`` only when its mean frequency across the
        ensemble reaches the nominal threshold.
        """
        models = getattr(self, "ensemble_models_", None)
        if not models:
            return self.model_.get_rsi_stability()

        diagnostics = [m.get_rsi_stability() for m in models]
        thresholds = {d["nominal_threshold"] for d in diagnostics}
        effective = {d["effective_threshold"] for d in diagnostics}
        if len(thresholds) != 1 or len(effective) != 1:
            raise RuntimeError("Inconsistent RSI stability thresholds in ensemble")
        nominal = thresholds.pop()
        effective_threshold = effective.pop()
        repeats = int(models[0].rsi_stability_repeats)
        terms = sorted({term for d in diagnostics
                        for term in d["counts"]})
        frequencies = {}
        for term in terms:
            values = [float(d["frequencies"].get(term, 0.0))
                      for d in diagnostics]
            frequencies[term] = float(np.mean(values))
        selected = [term for term, freq in frequencies.items()
                    if repeats > 0 and freq >= nominal / repeats]
        return {
            "enabled": all(bool(d["enabled"]) for d in diagnostics),
            "nominal_threshold": nominal,
            "effective_threshold": effective_threshold,
            "selected_terms": selected,
            "counts": {},
            "frequencies": frequencies,
            "per_model": diagnostics,
        }

    def get_kernel_weights(self):
        """Mean kernel weights across the selected MSB-GP ensemble."""
        if getattr(self, "candidate_stack_enabled_", False):
            acc = {}
            total_weight = 0.0
            models_by_name = getattr(self, "candidate_models_", {})
            weights = getattr(self, "candidate_stack_weights_", None)
            if weights is not None:
                for name, w in zip(self.candidate_stack_names_, weights):
                    models = models_by_name.get(name, [])
                    if not models:
                        continue
                    w = float(max(0.0, w))
                    if w <= 0.0:
                        continue
                    total_weight += w
                    for m in models:
                        if not hasattr(m, "get_kernel_weights"):
                            continue
                        for k, v in m.get_kernel_weights().items():
                            acc[k] = acc.get(k, 0.0) + w * float(v) / len(models)
            if total_weight > 0.0 and acc:
                return {k: v / total_weight for k, v in acc.items()}
        if getattr(self, "ensemble_models_", None):
            acc = {}
            for m in self.ensemble_models_:
                if not hasattr(m, "get_kernel_weights"):
                    continue
                for k, v in m.get_kernel_weights().items():
                    acc[k] = acc.get(k, 0.0) + float(v)
            n = len(self.ensemble_models_)
            return {k: v / n for k, v in acc.items()}
        return self.model_.get_kernel_weights()
