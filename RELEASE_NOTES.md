# MSB-GP 1.0.0

## Model

- **MSB-GP** is a multi-kernel sparse genomic prediction model. The public
  entry point is `msbgp.MSBCal`; the core estimator is `msbgp.MSBGP`.
- Genetic effects are modelled as a sum of kernel components
  `g_k ~ N(0, s_k K_k)`, with kernel variances parameterised as
  `s_k = v_total * softmax(w)_k` so that kernel shares are identifiable.
- Kernel contributions are learned by **restricted maximum likelihood (REML)**
  with Cholesky solves and adaptive jitter. Prediction and reliability use the
  same REML projection `Py`, so the estimated intercept is handled consistently
  across fitting, prediction and PEV.

## Kernel library

- Additive and dominance kernels (the additive/dominance core).
- Explicit AA/AD/DD product epistasis kernels, built from training-only marker
  state and normalisers.
- Gaussian/RBF and polynomial kernels as flexible nonlinear components.
- An internal residual sparse interaction (RSI) layer with strict stability
  selection, available as an opt-in module.

## Adaptive weighting

- Softmax-constrained kernel shares remove the scale non-identifiability of
  unconstrained multi-kernel models.
- An **evidence-constrained nonlinear gate** applies a weak marginal-likelihood
  penalty on the total mass given to non-core kernels, so the generic nonlinear
  kernels must earn their weight against the additive/dominance core.
- A **prediction-first higher-order gate** compares, within leakage-free inner
  folds, the out-of-fold accuracy of the full kernel library against a
  core-only (additive + dominance) library, and falls back to the core library
  when the higher-order kernels provide no out-of-fold gain.

## Calibration

- `MSBCal` fits a rank-preserving affine correction on inner out-of-fold
  predictions only; test labels are never used. Calibration improves
  centre/scale agreement while leaving ranking-based metrics unchanged.

## Diagnostics

- `get_kernel_weights`, `get_kernel_correlations`, `get_dropped_kernels`.
- `get_higher_order_score`, `get_gate_diagnostics`.
- `get_rsi_stability`.
- `get_model_heritability`, `get_genetic_variance_ratio`, `get_reliability`.
- Optimizer convergence diagnostics: `optim_success_`, `optim_message_`,
  `optim_gradient_norm_`, `optim_niter_`, `optim_boundary_parameters_`,
  `optim_start_label_`, `optim_n_starts_`.

## Benchmarks

- Simulation benchmarks across nine genetic architectures (additive to
  dominance to epistasis).
- Real-data cross-species benchmarks with accuracy, ranking metrics
  (BAC/AUC/AUPRC), error metrics (Pearson/MSE/RMSE/MAE), runtime and peak
  memory, with error bars and significance testing.
## Computational implementation

- REML uses a single Cholesky factor per objective evaluation and avoids
  materialising a persistent full inverse matrix.
- The final REML Cholesky factor is cached and reused for fitted-value,
  projection and reliability calculations when the optimiser has already
  evaluated the final parameter vector.
- Trace terms are accumulated blockwise with in-place right-hand-side solves,
  and the matrix-sized workspace is reused across blocks.
- The positive-definite factorisation avoids an additional n-by-n copy when
  the caller no longer needs the assembled covariance matrix, with a safe
  rebuild path for adaptive-jitter retries.
- Reliability is computed by row-block solves for `diag(G V^-1 G)`, avoiding
  simultaneous materialisation of the fitted `n x n` genetic covariance and
  the full `V^-1 G` solve.  The block size is configurable through
  `reliability_block_size`.
- Post-hoc ridge marker effects reuse the smaller Gram matrix between the
  penalty selection and solve, using the dual `n x n` system when markers
  outnumber individuals.
- Calibration and final candidate stacking share one `KernelState` per split
  across the full, core and sparse-additive variants.
- A temporary training-kernel cache is enabled only during shared-state variant
  fitting and cleared afterwards, so final models do not retain all n-by-n
  kernel matrices.
- Candidate-stack prediction uses a temporary cross-kernel cache when candidate
  models share one `KernelState`, avoiding repeated validation-kernel
  construction.
- `KernelState.X_train_` is a read-only reference rather than an unconditional
  genotype copy.
- Sparse genotype input: `scipy.sparse` CSR/CSC matrices are accepted by the public workflow, with exact dense-vs-sparse equivalence tests for additive, dominance and nonlinear kernel paths.


## Prediction accuracy update

- Added a gated sparse-additive rescue to MSB-GP.
- The rescue is selected by leakage-free inner out-of-fold accuracy.
- Added a dominance-aware stricter gate to avoid replacing D/E signal in
  dominance- or epistasis-heavy scenarios.
- Added a minimum-sample/minimum-marker guard for small datasets.
- Updated gate diagnostics with core dominance share and effective sparse-gate
  threshold.
- Added an internal candidate stack over the MSB-GP core, sparse-additive and
  full-library candidates. Candidate columns are calibrated to phenotype scale
  before stacking, and final candidates are robustly aligned to their
  out-of-fold distributions before prediction. The stack is accepted only when
  cross-fitted out-of-fold Pearson accuracy exceeds the best single candidate.
- Added a native explicit epistasis candidate (`additive + dominance +
AA/AD/DD`) to the internal candidate selector, so higher-order signal can be
modeled inside MSB-GP without calling external predictors.
- The native epistasis candidate now uses a training-residual sparse AA/AD/DD
selector by default.  It screens markers from core-model residuals, keeps a
compact interaction set, and passes only those features to the kernel builder.
This reduces variance and avoids an expensive all-pair interaction basis while
remaining entirely inside MSB-GP.
- The default MSB-GP workflow is native-only (`aux_candidates=()`); GBLUP,
RR-BLUP and GMKL remain separate benchmark comparators and are never called
inside the deployed model unless a user explicitly opts into the auxiliary
candidate bank.
- Added diagnostics for candidate names, weights, acceptance gain, per-candidate
  affine maps and final-to-OOF alignment maps.
