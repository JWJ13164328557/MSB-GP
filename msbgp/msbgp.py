"""MSB-GP: Multi-kernel Sparse Genomic Prediction.

MSB-GP extends GBLUP by fitting an interpretable library of relationship
kernels with a marginal-likelihood-based adaptive kernel weighting.  The
implementation is deterministic and leakage-free: all kernel parameters are
estimated on the training set, variance components are estimated by restricted
maximum likelihood (REML), and the optional sparse interaction layer uses
repeated training-set subsampling.

Model
-----
    y = mu 1 + sum_k g_k + e
    g_k ~ N(0, s2_k K_k),  e ~ N(0, s2_e I)

The identifiable parameterisation is ``s2_k = v_total softmax(w)_k``.  This
constrains kernel shares to be non-negative and to sum to one.  Variance
components are estimated by maximising the REML log-likelihood with an
intercept as the only fixed effect.  The optimization vector is

    theta = [log(v_total), w_2..w_K, log(s2_e)].

Kernel contributions are adaptively estimated by REML under a constrained
variance-share parameterization;
it is not an explicit Bayesian ARD prior and is not a Bayesian MCMC sampler.
Similarly, ``get_marker_effects`` is a post-hoc ridge reconstruction for
exploration rather than a BayesA/BayesC-style sparse effect estimator.

Evidence-constrained nonlinear gate
-----------------------------------
Because the generic nonlinear kernels (gaussian, poly) can mimic the additive
kernel on finite samples, the unconstrained REML objective occasionally lets a
nonlinear kernel absorb the whole additive signal and drift to a degenerate
boundary solution.  To keep kernel weights identified we add a weak
marginal-likelihood penalty on the *total* mass assigned to the non-core
(non-additive, non-dominance) kernels,

    penalty = kappa * S_NL,   S_NL = sum_{k in NL} p_k,
    kappa   = kernel_prior_strength * n,

which shrinks nonlinear contributions toward the additive/dominance core
unless the data provide enough marginal-likelihood evidence to keep them.
Additive and dominance shares are never penalised, so genuine additive and
dominance signal is preserved.  Set ``kernel_prior="additive"`` (with
``kernel_prior_add_share``) for an explicit target-distribution shrinkage,
or ``kernel_prior=None`` to disable the gate.
"""

import warnings

import numpy as np

from .kernels import KernelState, _as_float_matrix, _to_dense
from .rsi import build_rsi_terms

__all__ = ["MSBGP"]

_DEF_JITTER_BASE = 1e-8
_DEF_JITTER_MAX = 1e-4
_DEF_JITTER_GROWTH = 10.0


class MSBGP:
    """Multi-kernel genomic predictor with REML kernel weighting.

    Parameters
    ----------
    kernels : list or "auto"
        Kernel names from {additive, dominance, gaussian, poly, AA, AD, DD}.
        "auto" uses additive, dominance, gaussian and poly.  The internal
        RSI kernel is opt-in (rsi=True) and is NOT in the default library.
    em_iters : int
        Maximum L-BFGS-B iterations.
    tol : float
        Relative objective tolerance.
    l2_kernels : bool
        Compatibility argument; retained but unused.
    jitter_base, jitter_max : float
        Diagonal jitter control.  A small non-zero floor is always enforced.
    rsi : bool
        Whether to enable the internal RSI sparse-interaction kernel (default False).
    rsi_stability_repeats : int
        Number of training-set subsamples used by RSI stability selection.
        The default is 20; set to 1 only for very quick tests.
    rsi_stability_fraction : float
        Nominal selection-frequency threshold.  A term must be selected in at
        least ``ceil(fraction * repeats)`` subsamples.  No fallback threshold
        is used; if no term passes, the RSI kernel is omitted.
    rsi_stability_min_terms : int
        Compatibility argument.  It is retained in the API but is
        ignored when strict stability selection is active.
    epi_kernels : bool, default True
        Add dedicated sparse AA/AD/DD epistasis kernels to the ``auto``
        library.  The kernels are formed from training-only marginal screening
        and remain part of the single MSB-GP model.
    epi_top_markers : int, default 0
        Number of training-selected markers used for each sparse epistasis
        relationship matrix.  The default 0 uses all markers, giving the
        efficient all-pair AA/AD/DD product kernels.
    residualize : bool or list
        Residualise selected kernels against the additive kernel.
    kernel_correlation_threshold : float or None
        Absolute Pearson correlation above which two kernels are flagged as
        redundant.  ``None`` disables the diagnostic.
    kernel_redundancy : {"warn", "drop", "none"}
        Behaviour when the threshold is exceeded.  Default is "warn".
    kernel_prior : {"nonlinear", "additive", None} or dict/array
        Evidence constraint on the learned kernel shares.  The default
        ``"nonlinear"`` applies a weak penalty ``kappa * S_NL`` to the total
        non-core (non-additive, non-dominance) kernel mass ``S_NL``, which
        keeps the kernel weights identified without suppressing real additive
        or dominance signal.  ``"additive"`` instead shrinks the whole share
        vector toward an additive-dominant target.  A dict/array gives explicit
        target shares (renormalised).  ``None`` disables the constraint.
    kernel_prior_strength : float
        Per-sample strength kappa of the constraint; the effective penalty is
        ``kernel_prior_strength * n``.  Default 0.02.  Set 0.0 (or
        ``kernel_prior=None``) to disable.
    kernel_prior_add_share : float
        Additive share of the target distribution when
        ``kernel_prior="additive"`` (the remainder is spread over the other
        kernels).  Default 0.5.
    """

    def __init__(self, kernels="auto", em_iters=500, tol=1e-7,
                 l2_kernels=True, verbose=False,
                 jitter_base=_DEF_JITTER_BASE,
                 jitter_max=_DEF_JITTER_MAX,
                 rsi=False, rsi_top_markers=150,
                 rsi_interactions_per_type=150, rsi_max_terms=500,
                 rsi_stability_repeats=20, rsi_stability_fraction=0.60,
                 rsi_stability_sample_fraction=0.80,
                 rsi_stability_min_terms=5, rsi_random_state=0,
                 rsi_gate=False, rsi_gate_permutations=8,
                 rsi_gate_alpha=0.05,
                 epi_kernels=True, epi_top_markers=0, epi_in_core=True,
                 epi_terms=None,
    sparse_additive_top_markers=0,
                 residualize=False,
                 kernel_prior="nonlinear", kernel_prior_strength=0.02,
                 kernel_floor=None, kernel_floor_strength=0.05,
                 kernel_prior_add_share=0.5,
                 kernel_correlation_threshold=0.98,
                 kernel_redundancy="warn",
                 trace_block_size=128,
                 reliability_block_size=512,
                 reml_init="single",
                 reml_multistart=1,
                 reml_max_eval=None,
                 reml_gtol=None):
        self.kernels = kernels
        self.em_iters = em_iters
        self.tol = tol
        self.l2_kernels = l2_kernels
        self.verbose = verbose
        self.jitter_base = jitter_base
        self.jitter_max = jitter_max
        self.rsi = rsi
        self.rsi_top_markers = rsi_top_markers
        self.rsi_interactions_per_type = rsi_interactions_per_type
        self.rsi_max_terms = rsi_max_terms
        self.rsi_stability_repeats = rsi_stability_repeats
        self.rsi_stability_fraction = rsi_stability_fraction
        self.rsi_stability_sample_fraction = rsi_stability_sample_fraction
        self.rsi_stability_min_terms = rsi_stability_min_terms
        self.rsi_random_state = rsi_random_state
        self.rsi_gate = rsi_gate
        self.rsi_gate_permutations = rsi_gate_permutations
        self.rsi_gate_alpha = rsi_gate_alpha
        self.epi_kernels = bool(epi_kernels)
        self.epi_top_markers = int(epi_top_markers)
        self.epi_in_core = bool(epi_in_core)
        self.epi_terms = None if epi_terms is None else list(epi_terms)
        self.sparse_additive_top_markers = int(sparse_additive_top_markers)
        self.residualize = residualize
        self.kernel_prior = kernel_prior
        self.kernel_prior_strength = float(kernel_prior_strength)
        self.kernel_floor = kernel_floor
        self.kernel_floor_strength = float(kernel_floor_strength)
        self.kernel_prior_add_share = float(kernel_prior_add_share)
        self.kernel_correlation_threshold = kernel_correlation_threshold
        self.kernel_redundancy = kernel_redundancy
        self.trace_block_size = int(trace_block_size)
        self.reliability_block_size = int(reliability_block_size)
        self.reml_init = str(reml_init).strip().lower()
        if self.reml_init not in {"single", "heuristic", "multi_start", "multistart", "auto"}:
            raise ValueError(
                "reml_init must be one of 'single', 'heuristic', or 'multi_start'"
            )
        self.reml_multistart = max(1, int(reml_multistart))
        self.reml_max_eval = None if reml_max_eval is None else max(1, int(reml_max_eval))
        self.reml_gtol = None if reml_gtol is None else float(reml_gtol)
        self.fitted_ = False

    def _stable_rsi_terms(self, X, y):
        """Select RSI terms by strict training-set stability selection.

        A candidate must appear in at least
        ``ceil(rsi_stability_fraction * rsi_stability_repeats)`` subsamples.
        No minimum-count fallback is used.  If
        no candidate reaches the nominal threshold, no RSI term is returned.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        reps = int(self.rsi_stability_repeats)
        frac = float(self.rsi_stability_sample_fraction)
        self.rsi_stability_counts_ = {}
        self.rsi_stability_nominal_threshold_ = None
        self.rsi_stability_effective_threshold_ = None
        self.rsi_stability_enabled_ = bool(reps > 1 and 0.0 < frac < 1.0)

        if not self.rsi_stability_enabled_:
            # This quick-test mode is not stability selection and is retained
            # only for small smoke tests.  It is documented as such.
            return build_rsi_terms(
                X, y,
                top_markers=self.rsi_top_markers,
                interactions_per_type=self.rsi_interactions_per_type,
                max_terms=self.rsi_max_terms,
                random_state=int(self.rsi_random_state),
            )

        selected, counts, threshold = self._stability_select(X, y)
        self.rsi_stability_counts_ = {term: int(count)
                                      for term, count in counts.items()}
        self.rsi_stability_nominal_threshold_ = threshold
        self.rsi_stability_effective_threshold_ = threshold
        return selected[:int(self.rsi_max_terms)]

    def _stability_select(self, X, y):
        """Return (selected_terms, counts, threshold) without side effects."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        reps = int(self.rsi_stability_repeats)
        frac = float(self.rsi_stability_sample_fraction)
        rng = np.random.default_rng(int(self.rsi_random_state))
        n_sub = max(10, int(round(n * frac)))
        counts = {}
        for rep in range(reps):
            idx = np.sort(rng.choice(n, size=n_sub, replace=False))
            terms = build_rsi_terms(
                X[idx], y[idx],
                top_markers=self.rsi_top_markers,
                interactions_per_type=self.rsi_interactions_per_type,
                max_terms=self.rsi_max_terms,
                random_state=int(self.rsi_random_state) + rep,
            )
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
        threshold = int(np.ceil(float(self.rsi_stability_fraction) * reps))
        selected = [term for term, count in counts.items()
                    if count >= threshold]
        selected.sort(key=lambda t: (-counts[t], t[0], t[1], t[2]))
        return selected, counts, threshold

    def _rsi_gate_pass(self, X, y, observed_terms):
        """Phenotype-permutation activation gate for the RSI kernel.

        The RSI kernel is enabled only when the number of strictly stable
        interactions selected on the real phenotype exceeds the upper
        ``1 - rsi_gate_alpha`` quantile of the same statistic evaluated on
        permuted phenotypes.  This prevents RSI from firing on chance
        structure.
        """
        if not getattr(self, "rsi_gate", False):
            return True
        obs_n = int(len(observed_terms))
        y = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(int(self.rsi_random_state) + 9173)
        null_counts = []
        for _ in range(int(self.rsi_gate_permutations)):
            y_perm = rng.permutation(y)
            sel, _counts, _thr = self._stability_select(X, y_perm)
            null_counts.append(int(len(sel)))
        self.rsi_gate_observed_ = obs_n
        self.rsi_gate_null_counts_ = list(null_counts)
        if null_counts:
            thr = float(np.quantile(null_counts, 1.0 - float(self.rsi_gate_alpha)))
        else:
            thr = 0.0
        self.rsi_gate_threshold_ = thr
        passed = bool(obs_n > thr and obs_n > 0)
        self.rsi_gate_passed_ = passed
        return passed

    @staticmethod
    def _kernel_correlations(Klist):
        """Pairwise Pearson correlations of upper-triangular kernel entries."""
        if len(Klist) < 2:
            return {}
        n = Klist[0][1].shape[0]
        idx = np.triu_indices(n, 1)
        if idx[0].size == 0:
            return {}
        values = []
        for _, K in Klist:
            z = np.asarray(K)[idx].astype(float)
            z = z[np.isfinite(z)]
            values.append(z)
        out = {}
        for i in range(len(Klist)):
            for j in range(i + 1, len(Klist)):
                a, b = values[i], values[j]
                if a.size < 2:
                    out[(Klist[i][0], Klist[j][0])] = 0.0
                    continue
                if a.std() < 1e-15 or b.std() < 1e-15:
                    out[(Klist[i][0], Klist[j][0])] = 0.0
                else:
                    out[(Klist[i][0], Klist[j][0])] = float(np.corrcoef(a, b)[0, 1])
        return out

    def _handle_redundant_kernels(self, Klist):
        correlations = self._kernel_correlations(Klist)
        threshold = self.kernel_correlation_threshold
        if threshold is None or not correlations:
            self.kernel_correlations_ = correlations
            self.dropped_kernels_ = []
            return Klist

        retained = []
        dropped = []
        for item in Klist:
            redundant = False
            for kept in retained:
                pair = (kept[0], item[0])
                reverse = (item[0], kept[0])
                value = correlations.get(pair, correlations.get(reverse, 0.0))
                if np.isfinite(value) and abs(value) > float(threshold):
                    redundant = True
                    break
            if redundant:
                dropped.append(item[0])
            else:
                retained.append(item)

        if dropped and self.kernel_redundancy == "warn":
            warnings.warn(
                "Highly correlated kernels detected; retaining them may make "
                f"kernel shares unstable: {', '.join(dropped)}",
                RuntimeWarning,
                stacklevel=2,
            )
        elif dropped and self.kernel_redundancy == "drop":
            warnings.warn(
                "Dropping highly correlated kernels to improve identifiability: "
                f"{', '.join(dropped)}",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.kernel_redundancy not in {"warn", "drop", "none"}:
            raise ValueError("kernel_redundancy must be warn, drop, or none")
        if self.kernel_redundancy == "drop" and dropped:
            Klist = retained
        self.kernel_correlations_ = correlations
        self.dropped_kernels_ = dropped
        return Klist

    def _resolve_residualize(self):
        """Normalise the ``residualize`` option into a list of kernel names."""
        r = self.residualize
        if r in (False, None) or (isinstance(r, str) and r.lower() in ("none", "")):
            return []
        if r is True or (isinstance(r, str) and r.lower() == "auto"):
            # Deflate the two generic nonlinear kernels against the additive
            # kernel by default: these are the ones most able to mimic a
            # linear/additive signal and squeeze it out of the additive kernel.
            return ["gaussian", "poly"]
        if isinstance(r, str):
            return [r]
        return [str(x) for x in r]

    def _make_kernels(self, X, y=None):
        """Build the training-only kernel library and internal RSI kernel."""
        is_auto = self.kernels == "auto"
        explicit = [] if is_auto else [str(k) for k in self.kernels]
        explicit_lower = {k.lower() for k in explicit}
        epi_names = [k for k in explicit if k.upper() in {"AA", "AD", "DD"}]
        if is_auto and bool(getattr(self, "epi_kernels", False)):
            epi_names = ["AA", "AD", "DD"]
        if self.epi_terms is not None and (epi_names or (is_auto and bool(getattr(self, "epi_kernels", False)))):
            self.kernel_state_.set_epi_terms(self.epi_terms)
        sparse_names = {"additive_sparse", "add_sparse", "sparse_additive"}
        Klist = self.kernel_state_.build_training_library(
            additive=is_auto or ("additive" in explicit_lower),
            dominance=is_auto or ("dominance" in explicit_lower),
            gaussian=is_auto or ("gaussian" in explicit_lower),
            polynomial=is_auto or ("poly" in explicit_lower),
            epistasis=epi_names or None,
            additive_sparse=bool(explicit_lower & sparse_names),
        )
        use_rsi = bool(self.rsi) and y is not None and (
            is_auto or ("rsi" in explicit_lower)
        )
        self.rsi_terms_ = []
        self.rsi_scale_ = 1.0
        self.rsi_stability_counts_ = {}
        self.rsi_stability_nominal_threshold_ = None
        self.rsi_stability_effective_threshold_ = None
        self.rsi_stability_enabled_ = False
        self.rsi_gate_passed_ = None
        if use_rsi:
            terms = self._stable_rsi_terms(X, y)
            if terms and not self._rsi_gate_pass(X, y, terms):
                warnings.warn(
                    "RSI activation gate rejected the RSI kernel "
                    f"(observed={getattr(self, 'rsi_gate_observed_', None)} "
                    f"<= null threshold "
                    f"{getattr(self, 'rsi_gate_threshold_', None)}); "
                    "the model falls back to the core kernel library.",
                    RuntimeWarning, stacklevel=2)
                terms = []
            if terms:
                self.kernel_state_.set_rsi(terms)
                self.rsi_terms_ = list(terms)
                self.rsi_scale_ = float(self.kernel_state_.rsi_scale_)
                Klist.append(("RSI", self.kernel_state_.training_kernel("RSI")))
        return Klist

    @staticmethod
    def _softmax(w):
        w = np.asarray(w, dtype=float)
        w = w - np.max(w)
        e = np.exp(w)
        return e / np.sum(e)

    @staticmethod
    def _s2_from_theta(theta, n_k):
        theta = np.asarray(theta, dtype=float)
        log_v = float(np.clip(theta[0], -30.0, 30.0))
        v_total = float(np.exp(log_v))
        s2e = float(np.exp(float(np.clip(theta[n_k], -30.0, 30.0))))
        w = np.zeros(n_k)
        w[1:] = theta[1:n_k]
        p = MSBGP._softmax(w)
        return v_total * p, s2e, v_total

    def _reml_initial_points(self, theta0, n_k, var_y):
        """Return deterministic REML starting points.

        The default ``reml_init='single'`` preserves the historical single-start
        path exactly.  ``heuristic`` starts from a genetic/residual variance
        split that is often better conditioned for strongly unbalanced kernels,
        and ``multi_start``/``reml_multistart>1`` add deterministic perturbations
        to escape shallow boundary optima when requested.
        """
        base = np.asarray(theta0, dtype=float).copy()
        points = []
        init = str(getattr(self, "reml_init", "single")).lower()
        n_starts = max(1, int(getattr(self, "reml_multistart", 1)))

        if init in ("heuristic", "auto"):
            alt = base.copy()
            alt[0] = np.log(max(0.90 * float(var_y), 1e-8))
            alt[n_k] = np.log(max(0.10 * float(var_y), 1e-8))
            points.append(("heuristic", alt))
        else:
            points.append(("single", base))

        if init in ("multi_start", "multistart", "auto") or n_starts > 1:
            points.append(("heuristic", base.copy()))
            alt = base.copy()
            alt[0] = np.log(max(0.90 * float(var_y), 1e-8))
            alt[n_k] = np.log(max(0.10 * float(var_y), 1e-8))
            points.append(("heuristic_variance_split", alt))

        rng = np.random.default_rng(20261010)
        while len(points) < n_starts:
            start = base.copy()
            if n_k > 1:
                start[1:n_k] += rng.normal(0.0, 0.75, size=n_k - 1)
            start[0] += rng.normal(0.0, 0.50)
            start[n_k] += rng.normal(0.0, 0.50)
            points.append((f"perturb_{len(points)}", start))

        if n_starts == 1 and init not in ("heuristic", "auto"):
            return points[:1]
        return points[:n_starts]

    def _run_reml_optimizer(self, objective, gradient, theta0, bounds, n_k, var_y):
        """Optimize REML with optional deterministic multi-start diagnostics."""
        from scipy.optimize import minimize

        options = {"maxiter": int(self.em_iters), "ftol": float(self.tol)}
        if getattr(self, "reml_max_eval", None) is not None:
            options["maxfun"] = int(self.reml_max_eval)
        if getattr(self, "reml_gtol", None) is not None:
            options["gtol"] = float(self.reml_gtol)

        starts = self._reml_initial_points(theta0, n_k, var_y)
        results = []
        best = None
        best_fun = np.inf
        for label, start in starts:
            res = minimize(objective, start, jac=gradient, method="L-BFGS-B",
                           bounds=bounds, options=dict(options))
            try:
                res.reml_start_label_ = label
            except Exception:
                pass
            results.append(res)
            fun = float(getattr(res, "fun", np.inf))
            if best is None or (np.isfinite(fun) and fun < best_fun):
                best = res
                best_fun = fun
        self.optim_multistart_results_ = results
        self.optim_n_starts_ = len(results)
        self.optim_start_label_ = str(getattr(best, "reml_start_label_", "unknown"))
        return best

    def _make_V(self, s2, s2e, Klist, n, jitter):
        # LAPACK expects Fortran-contiguous input for in-place Cholesky.
        # Building V in Fortran order avoids an extra n-by-n conversion inside
        # scipy.linalg.cho_factor for every REML objective evaluation.
        if not Klist:
            V = np.zeros((n, n), dtype=float, order="F")
            V.flat[::n + 1] = s2e + jitter
            return V
        V = np.empty((n, n), dtype=float, order="F")
        V[:, :] = Klist[0][1]
        V *= float(s2[0])
        for (_, K), si in zip(Klist[1:], s2[1:]):
            V += float(si) * K
        V.flat[::n + 1] += float(s2e) + float(jitter)
        return V

    def _adapted_jitter(self, V_diag_scale):
        floor = 1e-10
        base = max(float(self.jitter_base) * max(float(V_diag_scale), 1.0), floor)
        return min(base, max(float(self.jitter_max), floor))

    @staticmethod
    def _factor_pd(V, base_jitter=0.0, jitter_max=1e-4, growth=10.0,
                   floor=1e-10, overwrite=False, rebuild=None):
        """Factor a positive-definite matrix with bounded adaptive jitter.

        Escalates diagonal jitter until a Cholesky factor is obtained.  If the
        bounded jitter is still insufficient (e.g. a kernel Gram matrix that is
        numerically singular because its rank is smaller than n), it falls back
        to an eigenvalue-floored reconstruction whose eigenvalues are all >= a
        scale-relative pad, so the factor is guaranteed to exist.
        """
        from scipy.linalg import cho_factor
        V = np.asarray(V, dtype=float)
        if not np.isfinite(V).all():
            mean_d = float(np.mean(np.abs(np.diag(V)))) or 1.0
            V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(V, np.maximum(np.diag(V), mean_d))
        n = V.shape[0]
        jitter_max = max(float(jitter_max), float(floor))
        cur = max(float(base_jitter), float(floor))
        # When ``rebuild`` is supplied, the first attempt may safely consume
        # the caller's V in place.  If Cholesky fails, subsequent attempts
        # reconstruct a clean V from the original kernel components, avoiding
        # both a permanent n-by-n copy and any dependence on partially
        # overwritten LAPACK workspace.
        use_overwrite = bool(overwrite and rebuild is not None)
        pending = V if use_overwrite else None
        while True:
            if pending is not None:
                A = pending
                pending = None
            elif rebuild is not None:
                A = np.asarray(rebuild(), dtype=float)
            else:
                A = np.array(V, dtype=float, copy=True)
            A.flat[::n + 1] += cur
            try:
                L, lower = cho_factor(A, lower=True, overwrite_a=True,
                                      check_finite=False)
                return L, lower
            except (np.linalg.LinAlgError, ValueError):
                if cur >= jitter_max:
                    base = (np.asarray(rebuild(), dtype=float)
                            if rebuild is not None else np.array(V, dtype=float, copy=True))
                    ev, evec = np.linalg.eigh(base)
                    scale = float(np.max(np.abs(ev))) or 1.0
                    pad = max(float(floor), 1e-8 * scale)
                    ev = np.maximum(ev, pad)
                    Vc = (evec * ev) @ evec.T
                    Vc.flat[::n + 1] += pad
                    return cho_factor(Vc, lower=True, overwrite_a=True,
                                      check_finite=False)
                cur = min(cur * growth, jitter_max)

    def _resolve_kernel_floor(self, n_k):
        """Per-kernel minimum share (floor) for the evidence-constrained fit.

        Returns a length-``n_k`` vector of minimum kernel shares, or ``None``.
        A float applies to the additive kernel(s); a dict maps kernel names to
        minimum shares; a string selects a named default group.  The floor is
        enforced softly in ``_neg_reml`` so a data-supported kernel can never be
        driven to (near) zero and discard the biological baseline.
        """
        kf = getattr(self, "kernel_floor", None)
        if kf is None:
            return None
        names = [str(n) for n in getattr(self, "kernel_names_", [])]
        n_k = int(n_k)
        vec = np.zeros(n_k, dtype=float)
        add_aliases = ("additive", "additive_sparse", "add_sparse",
                       "sparse_additive")
        if isinstance(kf, dict):
            for i, nm in enumerate(names):
                vec[i] = max(0.0, float(kf.get(nm, 0.0)))
        elif isinstance(kf, str):
            key = kf.strip().lower()
            strength = float(np.clip(getattr(self, "kernel_floor_strength",
                                               0.0), 0.0, 1.0))
            if key in ("additive", "add", "a", "dense_additive"):
                for i, nm in enumerate(names):
                    if str(nm).lower() in ("additive",):
                        vec[i] = strength
            elif key in ("core", "biological"):
                for i, nm in enumerate(names):
                    if str(nm).lower() in add_aliases + ("dominance",):
                        vec[i] = strength
            else:
                raise ValueError(f"Unknown kernel_floor specifier {kf!r}")
        else:
            strength = max(0.0, float(kf))
            for i, nm in enumerate(names):
                if str(nm).lower() == "additive":
                    vec[i] = strength
        if float(np.sum(vec)) <= 0.0:
            return None
        # floors must leave room for at least a sliver of every other kernel
        return np.minimum(vec, 1.0 - 1e-6)

    def _kernel_prior_p(self, n_k):
        """Target kernel-share vector for the evidence-constrained prior.

        ``None`` (default) disables the prior.  A string ``"additive"`` builds
        an additive-dominant target whose additive share is
        ``kernel_prior_add_share``; a dict/array specifies explicit target
        shares (renormalised).  The returned vector sums to one.
        """
        kp = getattr(self, "kernel_prior", None)
        if kp is None:
            return None
        names = [str(n) for n in getattr(self, "kernel_names_", [])]
        if isinstance(kp, str):
            key = kp.strip().lower()
            if key in ("", "none"):
                return None
            if key in ("nonlinear", "non-linear", "core", "nongaussian"):
                return None  # handled by the nonlinear-mass gate in _neg_reml
            if key in ("additive", "add", "a"):
                n_k = int(n_k)
                share = float(np.clip(self.kernel_prior_add_share, 0.0, 1.0))
                if n_k <= 1:
                    return np.ones(n_k, dtype=float)
                p0 = np.full(n_k, (1.0 - share) / float(n_k - 1))
                try:
                    p0[names.index("additive")] = share
                except ValueError:
                    p0[0] = share
                return p0
            raise ValueError(f"Unknown kernel_prior specifier {kp!r}")
        if isinstance(kp, dict):
            arr = np.array([max(0.0, float(kp.get(nm, 0.0))) for nm in names],
                           dtype=float)
        else:
            arr = np.asarray(kp, dtype=float).ravel()
            if arr.size != int(n_k):
                return None
            arr = np.maximum(arr, 0.0)
        total = float(np.sum(arr))
        if total <= 0.0:
            return None
        return arr / total


    def _reml_ones(self, n):
        """Cached intercept vector for repeated REML evaluations."""
        ones = getattr(self, "_ones_scratch_", None)
        if ones is None or ones.shape[0] != n:
            ones = np.ones(n, dtype=float)
            self._ones_scratch_ = ones
        return ones

    def _reml_y1_solve(self, n, L, lower, y):
        """Solve V^-1 [y, 1] with a reusable RHS buffer."""
        from scipy.linalg import cho_solve
        rhs = getattr(self, "_reml_y1_rhs_scratch_", None)
        if rhs is None or rhs.shape != (n, 2):
            rhs = np.empty((n, 2), dtype=float, order="F")
            self._reml_y1_rhs_scratch_ = rhs
        rhs[:, 0] = y
        rhs[:, 1] = self._reml_ones(n)
        sol = cho_solve((L, lower), rhs, overwrite_b=True,
                        check_finite=False)
        return sol[:, 0], sol[:, 1]

    def _reml_trace_rhs(self, n, block):
        """Reusable RHS buffer for blockwise exact trace estimation."""
        rhs = getattr(self, "_reml_trace_rhs_scratch_", None)
        if rhs is None or rhs.shape != (n, block):
            rhs = np.zeros((n, block), dtype=float, order="F")
            self._reml_trace_rhs_scratch_ = rhs
        return rhs

    def _neg_reml(self, theta, Klist, y, jitter=None):
        """Negative REML log-likelihood and analytical gradient.

        With X as an intercept design matrix, the restricted criterion is

            0.5 [ log|V| + log|X'V^-1X| + y'Py ],

        where P = V^-1 - V^-1X(X'V^-1X)^-1X'V^-1.  Cholesky solves are used
        throughout; no explicit V^-1 is needed for the likelihood value.  A
        fixed jitter supplied by ``fit`` keeps the objective and analytical
        gradient exactly consistent during optimisation.
        """
        from scipy.linalg import cho_factor, cho_solve

        # Only the most recent theta is cached: objective and gradient are
        # usually evaluated consecutively at the same point, while retaining
        # every n-vector would add avoidable memory pressure on large data.
        cache_key = np.asarray(theta, dtype=float).tobytes()
        last_key, cached = getattr(self, "_last_reml_cache_", (None, None))
        if cached is not None and last_key == cache_key:
            return cached

        n = y.shape[0]
        n_k = len(Klist)
        s2, s2e, v_total = self._s2_from_theta(theta, n_k)
        if jitter is None:
            jitter = self._adapted_jitter(v_total + s2e)
        V = self._make_V(s2, s2e, Klist, n, jitter)
        if not np.isfinite(V).all():
            result = (1e20, np.zeros(n_k + 1), np.zeros(n))
            self._last_reml_cache_ = (cache_key, result)
            return result

        # Release any previous factor before creating the next one; otherwise
        # the one-step factor cache would transiently double Cholesky memory.
        self._last_factor_cache_ = (None, None, None)
        rebuild_V = lambda: self._make_V(s2, s2e, Klist, n, jitter)
        try:
            L, lower = self._factor_pd(V, 0.0, overwrite=True,
                                       rebuild=rebuild_V)
        except (np.linalg.LinAlgError, ValueError):
            # Leave the penalised objective instead of crashing; the optimiser
            # will move away from the offending variance configuration.
            result = (1e20, np.zeros(n_k + 1), np.zeros(n))
            self._last_reml_cache_ = (cache_key, result)
            return result
        # Keep the most recent successful Cholesky factor.  ``fit`` often
        # evaluates the final theta during optimisation and would otherwise
        # factor the same V a second time.
        self._last_factor_cache_ = (cache_key, L, lower)
        ones = self._reml_ones(n)
        Vinv_y, Vinv_one = self._reml_y1_solve(n, L, lower, y)
        gls_B = float(ones @ Vinv_one)
        if not np.isfinite(gls_B) or gls_B <= 0:
            result = (1e20, np.zeros(n_k + 1), np.zeros(n))
            self._last_reml_cache_ = (cache_key, result)
            return result
        invB = 1.0 / gls_B
        XVinv_y = float(ones @ Vinv_y)
        P_y = Vinv_y - Vinv_one * invB * XVinv_y

        logdet_V = 2.0 * np.sum(np.log(np.diag(L)))
        logdet_B = np.log(gls_B)
        quadratic = float(y @ P_y)
        if not (np.isfinite(logdet_V) and np.isfinite(logdet_B)
                and np.isfinite(quadratic)):
            result = (1e20, np.zeros(n_k + 1), np.zeros(n))
            self._last_reml_cache_ = (cache_key, result)
            return result
        nll = 0.5 * (logdet_V + logdet_B + quadratic)

        # Exact traces without materialising the full n-by-n inverse.  Solve
        # identity columns in blocks, accumulating tr(V^-1) and
        # tr(V^-1 K_k) in the same pass.  The result is numerically equivalent
        # to the previous Hadamard-product form, but avoids one persistent
        # n-by-n Vinv allocation on every REML evaluation.
        trace_k = np.zeros(n_k, dtype=float)
        trace_ep = 0.0
        trace_block = int(getattr(self, "trace_block_size", 128))
        trace_block = max(1, min(trace_block, n))
        rhs = self._reml_trace_rhs(n, trace_block)
        for start in range(0, n, trace_block):
            stop = min(n, start + trace_block)
            width = stop - start
            rhs_view = rhs[:, :width]
            rhs_view.fill(0.0)
            rhs_view[np.arange(start, stop), np.arange(width)] = 1.0
            Xinv = cho_solve((L, lower), rhs_view, overwrite_b=True,
                             check_finite=False)
            trace_ep += float(np.sum(Xinv[np.arange(start, stop), np.arange(width)]))
            for i, (_, K) in enumerate(Klist):
                # K is symmetric, so this equals sum_j (V^-1 K)_jj for the
                # current identity-column block.
                trace_k[i] += float(np.einsum("ij,ij->", Xinv, K[:, start:stop]))

        h = np.zeros(n_k)
        for i, (_, K) in enumerate(Klist):
            # trace(K P) = trace(K V^-1) - v'Kv / (1'V^-1 1), with v = V^-1 1.
            K_Py = K @ P_y
            K_one = K @ Vinv_one
            trace_kp = trace_k[i] - float(Vinv_one @ K_one) * invB
            quad_kp = float(P_y @ K_Py)
            h[i] = 0.5 * (trace_kp - quad_kp)
        trace_ep_corrected = trace_ep - float(Vinv_one @ Vinv_one) * invB
        quad_ep = float(P_y @ P_y)
        h_e = 0.5 * (trace_ep_corrected - quad_ep)

        w = np.zeros(n_k)
        w[1:] = theta[1:n_k]
        p = self._softmax(w)
        p_dot_h = float(p @ h)
        gradient = np.zeros(n_k + 1)
        gradient[0] = float(np.sum(s2 * h))
        for j in range(1, n_k):
            gradient[j] = float(s2[j] * (h[j] - p_dot_h))
        gradient[n_k] = float(h_e * s2e)

        # Evidence-constrained prior on the learned kernel shares.
        #
        #   mode == "nonlinear": penalise only the total mass given to the
        #       non-core (nonlinear / epistasis / RSI) kernels, kappa*S_NL
        #       with S_NL = sum_{k in NL} p_k.  Additive and dominance keep
        #       their data-driven balance, so real additive/dominance signal
        #       is never suppressed.  Gradient:
        #           dS_NL/dw_j = [j in NL] p_j - p_j * S_NL.
        #   mode == "target": kappa * KL(p0 || p), gradient kappa*(p_j-p0_j).
        #
        # kappa_eff = per-sample strength x n keeps the prior influence
        # comparable across sample sizes.
        kappa_eff = float(getattr(self, "_kappa_eff_", 0.0))
        mode = getattr(self, "_kernel_prior_mode_", None)
        if kappa_eff > 0.0 and mode == "nonlinear":
            idx = getattr(self, "_nonlinear_idx_", ())
            if idx:
                S_nl = float(np.sum(p[list(idx)]))
                nll = nll + kappa_eff * S_nl
                idx_set = set(int(i) for i in idx)
                for j in range(1, n_k):
                    dmask = p[j] if j in idx_set else 0.0
                    gradient[j] += kappa_eff * (dmask - p[j] * S_nl)
        elif kappa_eff > 0.0 and mode == "target":
            p0 = getattr(self, "_kernel_prior_p_cache_", None)
            if p0 is not None and p0.shape[0] == n_k:
                log_p = np.log(np.clip(p, 1e-300, None))
                nll = nll - kappa_eff * float(np.sum(p0 * log_p))
                for j in range(1, n_k):
                    gradient[j] += kappa_eff * (p[j] - p0[j])

        # Per-kernel floor: softly require p_j >= floor_j for the protected
        # (biological baseline) kernels so REML can never zero the dense
        # additive/dominance background.  penalty = kappa*sum_j relu(f_j-p_j).
        kappa_floor = float(getattr(self, "_kappa_floor_eff_", 0.0))
        floor_vec = getattr(self, "_kernel_floor_vec_", None)
        if kappa_floor > 0.0 and floor_vec is not None:
            a = floor_vec - p
            active = a > 0.0
            if np.any(active):
                nll = nll + kappa_floor * float(np.sum(a[active]))
                S = float(np.sum(p[active]))
                for j in range(1, n_k):
                    if active[j]:
                        gradient[j] += kappa_floor * p[j] * (S - 1.0)
                    else:
                        gradient[j] += kappa_floor * p[j] * S

        gradient[~np.isfinite(gradient)] = 0.0

        result = (float(nll), gradient, P_y)
        self._last_reml_cache_ = (cache_key, result)
        return result

    def _build_kernel_state(self, X_train, y_train):
        """Build the training-only kernel state for this model specification.

        This helper is separated from :meth:`fit` so calibration and candidate
        stacking can build one state per outer/inner split and reuse it across
        the full, core, and sparse-additive MSB-GP variants.  Those variants
        share the same training genotypes and, when possible, the same marker
        screening; avoiding duplicate copies is the main low-risk memory
        optimisation for large real datasets.
        """
        self._residual_kernels_ = self._resolve_residualize()
        epi_explicit = (
            self.kernels != "auto" and
            any(str(k).upper() in {"AA", "AD", "DD"} for k in self.kernels)
        )
        epi_top = (int(getattr(self, "epi_top_markers", 0))
                   if (bool(getattr(self, "epi_kernels", False)) or epi_explicit)
                   else 0)
        sparse_explicit = (
            self.kernels != "auto" and
            any(str(k).lower() in
                {"additive_sparse", "add_sparse", "sparse_additive"}
                for k in self.kernels)
        )
        sparse_top = int(getattr(self, "sparse_additive_top_markers", 0))
        if sparse_explicit and sparse_top <= 0:
            # default to a modest slice when the kernel is requested but no
            # subset size was provided
            sparse_top = min(X_train.shape[1], max(10, int(round(0.10 * X_train.shape[1]))))
        if not sparse_explicit:
            sparse_top = 0
        state = KernelState().fit(
            X_train, residualize=self._residual_kernels_, y=y_train,
            epi_top_markers=epi_top,
            sparse_additive_top_markers=sparse_top)
        if self.epi_terms is not None:
            state.set_epi_terms(self.epi_terms)
        return state

    def fit(self, X_train, y_train, *, _kernel_state=None):
        """Fit kernel weights and variance components by REML.

        ``_kernel_state`` is an internal reuse hook used by calibration and
        candidate stacking.  It lets multiple model variants fitted on the same
        training split share allele frequencies, kernel normalisers, marker
        screening, and large genotype-derived matrices.
        """
        from scipy.optimize import minimize
        X_train = _as_float_matrix(X_train)
        y_train = np.asarray(y_train, dtype=float).ravel()
        n = X_train.shape[0]

        if _kernel_state is None:
            self.kernel_state_ = self._build_kernel_state(X_train, y_train)
        else:
            self._residual_kernels_ = self._resolve_residualize()
            self.kernel_state_ = _kernel_state
        Klist = self._make_kernels(X_train, y_train)
        Klist = self._handle_redundant_kernels(Klist)
        self.kernel_names_ = [name for name, _ in Klist]
        n_k = len(Klist)
        if n_k == 0:
            raise ValueError("At least one kernel must be enabled")
        self._kernel_prior_p_cache_ = self._kernel_prior_p(n_k)
        self._kernel_floor_vec_ = self._resolve_kernel_floor(n_k)
        _core = {"additive", "dominance", "additive_sparse", "add_sparse",
                 "sparse_additive"}
        if bool(getattr(self, "epi_in_core", True)) and (
                bool(getattr(self, "epi_kernels", False)) or
                any(str(nm).lower() in {"aa", "ad", "dd"}
                    for nm in self.kernel_names_)):
            _core = _core | {"aa", "ad", "dd"}
        _kp = getattr(self, "kernel_prior", None)
        if isinstance(_kp, str) and _kp.strip().lower() in (
                "nonlinear", "non-linear", "core", "nongaussian"):
            self._kernel_prior_mode_ = "nonlinear"
            self._nonlinear_idx_ = tuple(
                i for i, nm in enumerate(self.kernel_names_)
                if str(nm).lower() not in _core)
        elif self._kernel_prior_p_cache_ is not None:
            self._kernel_prior_mode_ = "target"
            self._nonlinear_idx_ = ()
        else:
            self._kernel_prior_mode_ = None
            self._nonlinear_idx_ = ()
        self._kappa_eff_ = float(self.kernel_prior_strength) * float(n)
        self._kappa_floor_eff_ = float(self.kernel_floor_strength) * float(n)

        self.sample_mean_ = float(y_train.mean())
        self.mu_ = self.sample_mean_
        y = y_train - self.sample_mean_
        var_y = float(np.var(y)) + 1e-8
        theta0 = np.zeros(n_k + 1)
        theta0[0] = np.log(max(var_y / 2.0, 1e-8))
        theta0[n_k] = np.log(max(var_y / 2.0, 1e-8))

        # Fix the numerical jitter for the whole optimization.  The tiny
        # theta-dependent jitter is harmless numerically but
        # is not differentiated in the analytical gradient.
        fixed_jitter = self._adapted_jitter(var_y)
        self._fixed_reml_jitter_ = fixed_jitter
        self._last_reml_cache_ = (None, None)
        self._last_factor_cache_ = (None, None, None)

        def objective(theta):
            return self._neg_reml(theta, Klist, y, fixed_jitter)[0]

        def gradient(theta):
            return self._neg_reml(theta, Klist, y, fixed_jitter)[1]

        bounds = ([(np.log(1e-8), np.log(1e12))] +
                  [(-30.0, 30.0)] * (n_k - 1) +
                  [(np.log(1e-8), np.log(1e12))])
        result = self._run_reml_optimizer(
            objective, gradient, theta0, bounds, n_k, var_y)
        theta = result.x
        if not result.success:
            warnings.warn(
                "MSB-GP REML optimizer did not converge: "
                f"{getattr(result, 'message', 'unknown optimizer status')}",
                RuntimeWarning,
                stacklevel=2,
            )
        s2, s2e, v_total = self._s2_from_theta(theta, n_k)
        s2 = np.clip(np.nan_to_num(s2, nan=1e-12, posinf=1e12, neginf=1e-12),
                     1e-12, 1e12)
        s2e = float(min(max(np.nan_to_num(s2e, nan=1e-12, posinf=1e12,
                                           neginf=1e-12), 1e-12), 1e12))
        # Reuse the last successful REML factor when the final theta was
        # evaluated during optimisation; this avoids one redundant O(n^3)
        # Cholesky and one full V allocation without changing the model.
        final_key = np.asarray(theta, dtype=float).tobytes()
        last_key, last_L, last_lower = getattr(
            self, "_last_factor_cache_", (None, None, None))
        if last_L is not None and last_key == final_key:
            L, lower = last_L, last_lower
        else:
            V = self._make_V(s2, s2e, Klist, n, fixed_jitter)
            rebuild_V = lambda: self._make_V(s2, s2e, Klist, n, fixed_jitter)
            L, lower = self._factor_pd(V, 0.0, overwrite=True,
                                       rebuild=rebuild_V)

        # GLS fixed-effect estimate.  Centering by the arithmetic mean is only
        # an initial translation; the final intercept must be estimated with
        # the same V used by REML.
        from scipy.linalg import cho_solve
        ones = self._reml_ones(n)
        Vinv_y, Vinv_one = self._reml_y1_solve(n, L, lower, y)
        gls_B = float(ones @ Vinv_one)
        if not np.isfinite(gls_B) or gls_B <= 0:
            raise RuntimeError("Non-positive GLS information for the intercept")
        gls_intercept = float(ones @ Vinv_y) / gls_B
        Py = Vinv_y - Vinv_one * gls_intercept

        self.s2_ = s2
        self.s2e_ = s2e
        self.v_total_ = v_total
        # Do not retain every n-by-n training kernel after fit.  They are
        # reconstructible from the training-only kernel state, which reduces
        # persistent memory substantially for multi-kernel models.  A lazy
        # ``Klist_`` property preserves compatibility for diagnostics.
        self._Klist_lazy_cache_ = None
        self.n_ = n
        self.X_train_ = X_train
        self.y_ = y
        self.Py_ = np.asarray(Py, dtype=float)
        self.Vinv_one_ = np.asarray(Vinv_one, dtype=float)
        self.gls_B_ = gls_B
        self.gls_intercept_ = gls_intercept
        self.mu_ = self.sample_mean_ + gls_intercept
        self.optim_result_ = result
        self.optim_success_ = bool(getattr(result, "success", False))
        self.optim_message_ = str(getattr(result, "message", ""))
        self.optim_niter_ = int(getattr(result, "nit", -1))
        grad = getattr(result, "jac", None)
        self.optim_gradient_norm_ = (
            float(np.linalg.norm(np.asarray(grad, dtype=float)))
            if grad is not None else float("nan")
        )
        names = (["log_v_total"] +
                 [f"w_{j}" for j in range(1, n_k)] +
                 ["log_s2e"])
        bounds_arr = np.asarray(bounds, dtype=float)
        at_lower = np.isclose(theta, bounds_arr[:, 0], rtol=0.0, atol=1e-8)
        at_upper = np.isclose(theta, bounds_arr[:, 1], rtol=0.0, atol=1e-8)
        self.optim_boundary_parameters_ = {
            name: bool(lo or hi)
            for name, lo, hi in zip(names, at_lower, at_upper)
        }
        self.reml_loglik_ = -float(result.fun)
        self.trace_ = [float(result.fun)]
        self.cho_L_ = L
        self.cho_lower_ = lower
        self.fitted_ = True
        return self

    @property
    def Klist_(self):
        """Lazy compatibility view of training kernels.

        The fit stores kernel names and variance components only.  Accessing
        this property reconstructs the n-by-n kernels on demand; use
        ``predict`` for normal low-memory prediction.
        """
        if not getattr(self, "fitted_", False):
            return []
        if getattr(self, "_Klist_lazy_cache_", None) is None:
            self._Klist_lazy_cache_ = [
                (name, self.kernel_state_.training_kernel(name))
                for name in self.kernel_names_
            ]
        return self._Klist_lazy_cache_

    def predict(self, X_va):
        """Predict genetic values using training-only fitted kernel state."""
        if not self.fitted_:
            raise RuntimeError("fit() must be called before predict()")
        X_va = _as_float_matrix(X_va)
        K_cross = self.kernel_state_.weighted_cross_kernel(
            self.kernel_names_, self.s2_, X_va)
        pred = K_cross.T @ self.Py_
        return self.mu_ + pred

    def get_variance_components(self):
        """Return kernel variance components and residual variance."""
        self._check_fitted()
        return ({name: s for name, s in zip(self.kernel_names_, self.s2_)},
                self.s2e_)

    def get_kernel_weights(self):
        """Return adaptive kernel shares (non-negative and summing to one)."""
        self._check_fitted()
        total = float(np.sum(self.s2_))
        return {name: float(s / total) if total > 0 else 0.0
                for name, s in zip(self.kernel_names_, self.s2_)}

    def get_kernel_correlations(self):
        """Return pairwise training-kernel correlation diagnostics."""
        self._check_fitted()
        return dict(getattr(self, "kernel_correlations_", {}))

    def get_dropped_kernels(self):
        """Return kernels removed by redundancy handling, if any."""
        self._check_fitted()
        return list(getattr(self, "dropped_kernels_", []))

    def get_rsi_stability(self):
        """Return strict RSI stability-selection diagnostics.

        Returns
        -------
        dict
            Keys include ``enabled``, ``nominal_threshold``,
            ``effective_threshold``, ``selected_terms``, ``counts`` and
            ``frequencies``.  ``effective_threshold`` equals the nominal
            threshold because no fallback relaxation is used.
        """
        self._check_fitted()
        counts = dict(getattr(self, "rsi_stability_counts_", {}))
        threshold = getattr(self, "rsi_stability_nominal_threshold_", None)
        return {
            "enabled": bool(getattr(self, "rsi_stability_enabled_", False)),
            "nominal_threshold": threshold,
            "effective_threshold": getattr(
                self, "rsi_stability_effective_threshold_", threshold),
            "selected_terms": list(getattr(self, "rsi_terms_", [])),
            "counts": counts,
            "frequencies": {
                term: (count / int(self.rsi_stability_repeats)
                       if self.rsi_stability_repeats > 0 else 0.0)
                for term, count in counts.items()
            },
        }

    def get_marker_effects(self, X, lam=None):
        """Post-hoc marker-effect reconstruction by ridge regression.

        This is exploratory and is not a sparse Bayesian QTL-effect estimate.
        """
        self._check_fitted()
        X = _to_dense(X)
        gebv = self.predict(X) - self.mu_
        n, m = X.shape
        Xc = X - X.mean(axis=0)

        # Build the smaller Gram matrix exactly once.  For m > n this is the
        # dual n-by-n system; otherwise it is the primal m-by-m system.
        gram = Xc @ Xc.T if m > n else Xc.T @ Xc
        if lam is None:
            eigenvalues = np.linalg.eigvalsh(gram)
            lam = max(float(eigenvalues[-1]) * 0.01, 1e-6)
        if m > n:
            alpha = np.linalg.solve(gram + lam * np.eye(n), gebv)
            return Xc.T @ alpha
        A = gram + lam * np.eye(m)
        return np.linalg.solve(A, Xc.T @ gebv)

    def get_genetic_variance_ratio(self):
        """Return total fitted genetic variance / total phenotypic variance."""
        self._check_fitted()
        s2g = float(np.sum(self.s2_))
        total = s2g + float(self.s2e_)
        return s2g / total if total > 0 else 0.0

    def get_model_heritability(self):
        """Alias for the total genetic-variance ratio (not narrow-sense h2)."""
        return self.get_genetic_variance_ratio()

    def get_heritability(self):
        """Compatibility alias; see ``get_model_heritability``.

        The returned value is a model variance ratio because the fitted
        kernels may contain dominance, nonlinear and interaction components.
        It should not be interpreted as narrow-sense heritability.
        """
        warnings.warn(
            "get_heritability is a compatibility alias; use get_model_heritability. "
            "The value is a total genetic-variance ratio, not narrow-sense h2.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_model_heritability()

    def get_reliability(self):
        """Training-set PEV-based reliability for total genetic value.

        Rel_i = 1 - PEV_i / G_ii, where G is the fitted total genetic
        covariance and PEV = diag(G - G P G).  The projection
        P = V^-1 - V^-1X(X'V^-1X)^-1X'V^-1 keeps the fixed-effect estimate
        consistent with REML.

        The matrix products are evaluated in column blocks so the fitted
        n-by-n G and the full n-by-n solve V^-1G are never both materialised.
        """
        self._check_fitted()
        from scipy.linalg import cho_solve

        n = self.n_
        # Reconstruct the fitted total genetic covariance G = sum_k s2_k K_k
        # once, rather than retaining every kernel and every n-by-n solved
        # block.  This keeps the reliability diagnostic memory bounded by one
        # G matrix plus one temporary kernel.
        G = np.zeros((n, n), dtype=float)
        diag_G = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        for name, s2 in zip(self.kernel_names_, self.s2_):
            K = self.kernel_state_.training_kernel(name)
            diag_G += s2 * K.diagonal()
            b += s2 * (K @ self.Vinv_one_)
            G += s2 * K
            del K

        diag_GVinvG = np.zeros(n, dtype=float)
        block_value = getattr(self, "reliability_block_size", 512)
        if block_value is None:
            block_value = getattr(self, "trace_block_size", 128)
        block = int(block_value)
        block = max(1, min(block, n))
        rhs = np.empty((n, block), dtype=float)
        for start in range(0, n, block):
            stop = min(n, start + block)
            width = stop - start
            G_block = rhs[:, :width]
            G_block.fill(0.0)
            G_block += G[:, start:stop]
            X_block = cho_solve(
                (self.cho_L_, self.cho_lower_),
                G_block,
                overwrite_b=False,
                check_finite=False,
            )
            # For i in this row block, diag(G V^-1 G)_i is
            # sum_j G_ij (V^-1 G)_ji.  Since G is symmetric, the column of
            # V^-1G corresponding to i can be matched with G[:, i].
            diag_GVinvG[start:stop] += np.sum(X_block * G_block, axis=0)

        # With an intercept-only fixed effect, the diagonal correction is
        # b_i^2 / (1'V^-1 1), where b = G V^-1 1.
        pev_diag = diag_G - diag_GVinvG + (b * b) / self.gls_B_
        denominator = np.maximum(diag_G, 1e-15)
        reliability = 1.0 - np.maximum(pev_diag, 0.0) / denominator
        return np.clip(reliability, 0.0, 1.0)

    def summary(self):
        """Return a concise fitted-model summary."""
        self._check_fitted()
        vc, s2e = self.get_variance_components()
        total = sum(vc.values()) + s2e
        lines = ["MSB-GP fit summary (REML)",
                 f"  individuals (train): {self.n_}",
                 f"  kernels: {', '.join(self.kernel_names_)}"]
        if getattr(self, "dropped_kernels_", None):
            lines.append("  dropped for redundancy: "
                         f"{', '.join(self.dropped_kernels_)}")
        lines.append(f"  total genetic variance: {self.v_total_:.4f}")
        lines.append("  variance components (share of total):")
        for name, s in vc.items():
            lines.append(f"    {name:>9s}: {s:.4f} ({100*s/total:.1f}%)")
        lines.append(f"    residual: {s2e:.4f} ({100*s2e/total:.1f}%)")
        return "\n".join(lines)

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("fit() must be called first")







