# Methods notes

## MSB-GP

The official public model remains MSB-GP. It is a multi-kernel sparse genomic
prediction model. Its variance components are learned by restricted maximum
likelihood (REML), not by explicit Bayesian ARD priors or MCMC.

\[
y = \mu + \sum_k g_k + e,\quad g_k \sim N(0, s_k K_k),\quad e \sim N(0, s_e I).
\]

The total genetic variance is decomposed across kernels with a softmax
parameterisation:

\[
s_k = v_\text{total}\,\mathrm{softmax}(w)_k.
\]

This fixes the scale non-identifiability of unconstrained multi-kernel models,
but it does not by itself make highly correlated kernels fully identifiable.
Therefore, MSB-GP retains pairwise kernel-correlation diagnostics and an
optional redundancy policy (`kernel_redundancy="warn"|"drop"|"none"`).

## Evidence-constrained nonlinear gate

The softmax parameterisation fixes the scale non-identifiability but, on
finite samples, the generic nonlinear kernels (Gaussian/RBF and polynomial)
can still mimic the additive kernel and absorb the entire additive signal,
drifting the kernel shares to a degenerate boundary solution. To keep the
shares identified, MSB-GP adds a weak marginal-likelihood penalty on the
*total* mass assigned to the non-core (non-additive, non-dominance) kernels:

\[
\text{penalty} = \kappa\,S_\mathrm{NL},
\qquad
S_\mathrm{NL} = \sum_{k\in \mathrm{NL}} p_k,
\qquad
\kappa = \texttt{kernel\_prior\_strength}\times n,
\]

with gradient

\[
\frac{\partial (\kappa S_\mathrm{NL})}{\partial w_j}
= \kappa\,\bigl([j\in\mathrm{NL}]\,p_j - p_j\,S_\mathrm{NL}\bigr).
\]

Additive and dominance shares are never penalised, so genuine additive and
dominance signal is preserved. The default strength is
`kernel_prior_strength = 0.02`. An explicit target distribution can be requested
with `kernel_prior="additive"` (additive share = `kernel_prior_add_share`), and
the constraint can be disabled with `kernel_prior=None`.

On the real Rice dataset the unconstrained model let the polynomial kernel take
essentially all the mass (share ~1.0) and collapsed to pooled r = 0.39 versus
GBLUP 0.60. The gate restores pooled r = 0.596 (additive ~0.70, poly ~0.21) —
back to the GBLUP level — while leaving purely additive and dominance
simulations essentially unchanged.

## REML

MSB-GP maximises the restricted marginal likelihood with an intercept as the
only fixed effect:

\[
\ell_R =
-\frac12\left[
\log|V|+\log|X'V^{-1}X|+y'Py
\right],
\quad
P=V^{-1}-V^{-1}X(X'V^{-1}X)^{-1}X'V^{-1}.
\]

All likelihood evaluations use Cholesky solves and a fixed numerical jitter
during optimization.

For prediction, the same projection is used:

\[
\hat g_* = G_{*\cdot}Py,
\quad
\hat y_* = \hat\mu_{\mathrm{GLS}} + \hat g_*,
\]

where the intercept-only GLS estimate is

\[
\hat\mu_{\mathrm{GLS}}=
(\mathbf{1}'V^{-1}\mathbf{1})^{-1}\mathbf{1}'V^{-1}y.
\]

Thus the arithmetic sample mean is only used to centre the response before
optimisation; the final intercept is estimated under the fitted covariance.

## Leakage-free kernel state

All quantities estimated from markers are fitted on the training set only:

- additive allele frequencies and relationship-matrix normaliser,
- dominance coding and normaliser,
- Gaussian/RBF bandwidth,
- polynomial centre and normaliser,
- AA/AD/DD kernel normalisers,
- RSI feature means, standard deviations and scale.

Training and cross-kernels therefore use exactly the same fitted state. This
removes validation-induced parameter drift from cross-kernel
construction.

## RSI layer

The residual sparse interaction (RSI) layer is internal to `MSB-GP`. It:

1. Ranks markers by their association with the training phenotype.
2. Constructs candidate AA/AD/DD interaction features.
3. Selects a sparse interaction set using repeated training-set subsampling.
4. Converts only strictly stable interactions into one kernel.
5. Lets REML estimate the RSI kernel contribution jointly with other kernels.

With the default 20 subsamples and a 60% stability threshold, a candidate must
be selected in at least 12/20 subsamples. There is no minimum-count fallback.
If no candidate reaches the nominal threshold, the RSI kernel is omitted for
that training set. `get_rsi_stability()` reports the nominal and effective
thresholds, selected terms, raw counts and selection frequencies.

Current stability settings:

```text
rsi_top_markers = 150
rsi_interactions_per_type = 150
rsi_max_terms = 500
rsi_stability_repeats = 20
rsi_stability_fraction = 0.60
```

RSI is a supervised sparse interaction kernel, not an exhaustive genome-wide
interaction search. It is not exposed as a separate public model.

## Marker effects and variance ratio

`get_marker_effects` performs post-hoc ridge reconstruction of marker effects
for exploration. It is not a BayesA/BayesC-style sparse QTL-effect estimator.

`get_model_heritability` and `get_genetic_variance_ratio` return the fitted
total genetic-variance ratio. Because the fitted kernels may include dominance,
nonlinear and interaction components, this value is not narrow-sense heritability.
`get_heritability` is retained only as a compatibility alias and returns the same model variance ratio.

## Reliability

Training-set reliability is PEV based and uses the same REML projection:

\[
\text{PEV}=G-GPG,\quad
\text{Rel}_i=1-\frac{\text{PEV}_{ii}}{G_{ii}}.
\]

For the intercept-only model this is evaluated efficiently as

\[
\text{PEV}_{ii}=G_{ii}-(GV^{-1}G)_{ii}
+\frac{(GV^{-1}\mathbf{1})_i^2}{\mathbf{1}'V^{-1}\mathbf{1}}.
\]

It is a model-based training reliability, not an independent validation
accuracy.


## Residual sparse AA/AD/DD selector

The native `epi` candidate can use a training-residual sparse AA/AD/DD
selector.  For each fit, MSB-GP first fits the additive + dominance core on the
training set, computes its residuals, screens the strongest additive and
dominance marker signals, scores candidate AA/AD/DD product terms, and keeps a
compact set of interactions.  The selected terms are then converted into
AA/AD/DD kernels and passed to the native multi-kernel REML candidate.  The
selector uses training labels only and never calls GBLUP, RR-BLUP, GMKL or any
external software.  `get_sparse_epistasis_diagnostics()` reports the selected
terms and selection limits.

## Internal candidate stacking

The public MSB-GP workflow also forms four native internal prediction
candidates: `core` (additive + dominance), `sparse` (additive +
sparse-additive + dominance), `epi` (additive + dominance + sparse residual AA/AD/DD kernels), and `full` (the full requested kernel library). Each
candidate produces out-of-fold predictions inside the training data only. A
small non-negative, sum-to-one stack is fitted to those predictions and is
accepted only when its cross-fitted Pearson correlation exceeds the best single
candidate by the configured margin. In the default workflow no GBLUP,
RR-BLUP, GMKL or other external predictor is called. An auxiliary benchmark
bank can be enabled explicitly through `aux_candidates`; it is then fit on the
same inner folds and accepted only under the same leakage-free score guards.

Candidate columns are first mapped to phenotype scale with their own
out-of-fold affine calibrations. At prediction time, final candidate models are
robustly aligned to their corresponding out-of-fold distribution before the
stack is applied. This prevents a numerically inflated final kernel scale from
dominating the convex combination. The stack remains part of the public
MSB-GP model; it does not create a new model name. Any opt-in external bank is
still trained and scored with training-fold labels only, so test-fold labels
never enter candidate selection.

## Calibration

The official workflow uses the affine correction

\[
\hat y_\text{cal} = a\hat y + b
\]

estimated from inner out-of-fold predictions only. Test labels are never used.
The transformation preserves ranking and therefore AUC, BAC and AUPRC, while
improving MSE/RMSE/MAE when the raw model has a scale or centre shift.


## Sparse-additive rescue and dominance-aware gating

The prediction-first hybrid architecture retains `additive` and `dominance` as
the biological core, with Gaussian, polynomial, AA/AD/DD and RSI kernels treated
as higher-order candidates that must earn their weight through out-of-fold
accuracy. A screened sparse-additive kernel is added as a conservative
candidate for sparse or heterogeneous additive architectures.

The sparse-additive candidate is fitted only on the training fold. Its marker
subset is selected using training phenotypes only, and the candidate enters the
final model only when it improves inner out-of-fold Pearson correlation over the
dense additive/dominance core. When the core model already assigns substantial
mass to dominance, the required improvement threshold is increased. This
prevents the sparse-additive rescue from replacing genuine dominance or
epistatic signal with training-fold-specific noise.

Automatic sparse-additive rescue has a minimum-sample and minimum-marker guard.
Small pilot datasets fall back to the core model unless the user explicitly
requests an `additive_sparse` kernel.
