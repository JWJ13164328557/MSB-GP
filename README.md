# MSB-GP

**MSB-GP** (Multi-kernel Sparse genomic prediction) is a complete, reproducible
toolkit for genomic selection (GS) modelling and benchmarking. It provides a
multi-kernel genomic prediction model with adaptive kernel weighting, an
internal sparse epistasis layer, and leakage-free output calibration, together
with the classical baselines and the benchmarking pipeline used to evaluate it.

## Features

- **Public model:** `MSB-GP`, implemented as `MSBCal` in code (calibrated
  workflow) over the core estimator `MSBGP`.
- **Kernels:** additive, dominance, Gaussian/RBF, polynomial, explicit AA/AD/DD
  product epistasis kernels, a training-residual sparse AA/AD/DD selector for
  the native epistasis candidate, and an opt-in residual sparse interaction
  (RSI) kernel.
- **Large-cohort genotype I/O:** int8/float32 memory-mapped `.npy` or `.dat`+JSON genotype matrices are accepted directly, reducing raw genotype storage by up to 8x while preserving the exact kernel/REML numerical path.
- **Adaptive weighting:** softmax-constrained kernel shares learned by REML,
  with an evidence-constrained nonlinear gate and a prediction-first
  higher-order gate.
- **Leakage-free calibration:** rank-preserving affine correction fitted on
  inner out-of-fold predictions only.
- **Internal candidate stacking:** the public MSB-GP workflow compares
  four native candidates through inner out-of-fold predictions: additive +
  dominance (`core`), sparse additive (`sparse`), explicit AA/AD/DD
  epistasis (`epi`, using a training-residual sparse AA/AD/DD selector by
  default), and the full kernel library (`full`). It combines them
  only when a constrained non-negative stack improves estimated accuracy.
- **Self-contained default:** the deployed model does not internally call
  GBLUP, RR-BLUP or GMKL. External baselines are separate benchmark
  comparators; an optional auxiliary candidate bank exists only for explicit
  opt-in studies.
- **Interpretability:** learned kernel shares separate additive from
  non-additive signal, with kernel-correlation and reliability diagnostics.
- **Baselines:** `GBLUP`, `RR-BLUP`, `BayesA`, `BayesC`, `RandomForest`, `MLP`,
  `DNNGP`, and GMKL variants.
- **Benchmarks:** a shared cross-validation pipeline for Pearson/Spearman,
  BAC/AUC/AUPRC, MSE/RMSE/MAE, runtime and peak memory, with error bars and
  significance testing.

## Install

```bash
python -m pip install -r requirements.txt
```

For an editable install:

```bash
python -m pip install -e .
```

## Quick start

```python
from sklearn.model_selection import train_test_split
from msbgp import MSBCal, pearson_corr, simulate_gs_data

d = simulate_gs_data(n=180, m=400, architecture="epistatic", h2=0.55, seed=1)
X, y = d["X"], d["y"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

model = MSBCal(k_inner=3).fit(X_tr, y_tr)
pred = model.predict(X_te)
print(pearson_corr(y_te, pred))
print(model.a_, model.b_)
print(model.get_kernel_weights())
```

Run the bundled example:

```bash
python examples/quickstart.py
```

Run the smoke tests:

```bash
python tests/run_tests.py
```

## Method summary

MSB-GP fits a multi-kernel model

\[
y = \mu + \sum_k g_k + e,\quad g_k \sim N(0, s_k K_k),\quad e \sim N(0, s_e I),
\]

where the \(K_k\) are additive, dominance, Gaussian, polynomial, epistasis and
RSI kernels. Kernel variances are parameterised as

\[
s_k = v_\text{total}\,\mathrm{softmax}(w)_k,
\]

which makes the kernel shares identifiable, and are estimated by restricted
maximum likelihood (REML) with Cholesky solves and adaptive jitter. A weak
evidence-constrained nonlinear gate penalises the total mass assigned to
non-core (non-additive, non-dominance) kernels so the generic nonlinear kernels
must earn their weight against the additive/dominance core. A prediction-first
higher-order gate compares, within leakage-free inner folds, the out-of-fold
accuracy of the full library against the additive+dominance core, and falls
back to the core when the higher-order kernels add no out-of-fold gain. The
public `MSBCal` wrapper then fits an affine calibration on inner out-of-fold
predictions only.

## Diagnostics

```python
model.get_kernel_weights()          # learned softmax kernel shares
model.get_kernel_correlations()     # pairwise kernel-correlation diagnostics
model.get_higher_order_score()      # OOF accuracy gain from higher-order kernels
model.get_gate_diagnostics()        # prediction-first gate status
model.get_rsi_stability()           # opt-in RSI stability diagnostics
model.get_sparse_epistasis_diagnostics()  # native sparse AA/AD/DD terms
model.get_model_heritability()      # total genetic-variance ratio
model.get_reliability()             # PEV-based training reliability
model.optim_success_                # REML optimizer convergence flag
```

## Benchmarks

Relevant scripts are provided under `experiments/`:

```bash
python experiments/run_comparison.py
python experiments/run_real_data_full.py
python experiments/run_benchmark_metrics.py
python experiments/run_real_data_mse.py
python experiments/run_5x5.py
python experiments/run_simulation_benchmark.py
python experiments/run_simulation_epi_validation.py
python experiments/run_objective_validation.py
```

Real-data inputs are not redistributed. See `data/README.md` for the expected
genotype and phenotype schema.

## Computational implementation

- REML uses one Cholesky factor per objective evaluation and avoids persistent
  full inverse matrices.
- Calibration and final candidate stacking share one `KernelState` per split
  across full, core, sparse-additive and explicit-epistasis variants.
- A temporary training-kernel cache is used only while shared-state variants are
  fitted and is cleared immediately afterwards, so final models do not retain
  all n-by-n kernel matrices.
- Candidate-stack prediction reuses a temporary cross-kernel cache when models
  share one `KernelState`, avoiding repeated validation-kernel construction.
- `KernelState.X_train_` is a read-only reference rather than an unconditional
  genotype copy.
- Sparse genotype input: `scipy.sparse` CSR/CSC matrices are accepted directly by `MSBGP` and `MSBCal`; the sparse path is covered by a dense-vs-sparse equivalence test with tolerance `<1e-9`.
- Memory-mapped genotype input: `save_genotype_memmap` / `load_genotype_memmap` support int8 `.npy` and `.dat`+JSON stores; the memmap path is covered by a dense-vs-memmap equivalence test with tolerance `<1e-9>`. REML initialization can use `reml_init="heuristic"` or `reml_init="multi_start"` with `reml_multistart>1`; the selected start and convergence diagnostics are stored on the fitted model.

## Adding a new species

Prepare your data folder:

```text
NewSpecies/
  Genotypes.csv   rows = markers, cols = samples (index = marker ID)
  Phenotypes.csv  rows = records, cols = <ID-column>, Trait_1
```

Then run the one-command analysis:

```bash
python experiments/run_new_species.py --data-dir "H:/.../NewSpecies"
```

For manual control:

```bash
python experiments/run_new_species.py `
    --data-dir "H:/.../NewSpecies" `
    --id-col "Sample" --coding "nucleotide" `
    --max-markers 3000 --max-samples 500 `
    --methods GBLUP RR-BLUP MSB-GP RandomForest `
    --genotype-memmap "H:/.../NewSpecies/Genotypes.dat" `
    --sample-ids "H:/.../NewSpecies/sample_ids.txt" `
    --reml-init heuristic --reml-multistart 2 `
    --output-dir "H:/.../results/new_species"
```

Outputs include `results_summary.csv` (Pearson r, MSE, RMSE, MAE, runtime),
`fold_details.csv`, `kernel_weights.csv`, pooled predictions, and a
publication-ready figure (PNG + PDF, 600 dpi).

## Package layout

```text
msbgp/
  kernels.py         kernel library
  rsi.py             residual sparse epistasis module
  bases.py           GBLUP, RR-BLUP, BayesA, BayesC
  msbgp.py           MSB-GP core estimator
  calibration.py     leakage-free affine calibration (MSBCal)
  ml_baselines.py    RandomForest / MLP baselines
  dnngp.py           DNNGP baseline
  data.py            simulator and data helpers
  cv.py              cross-validation utilities
  metrics.py         prediction metrics
examples/
  quickstart.py
experiments/
  benchmark and analysis scripts
tests/
  smoke tests
docs/
  methods, API and reproduction notes
```

## Citation

If you use this software, please cite MSB-GP (`CITATION.cff`).

## License

MIT. See `LICENSE`.

## Accuracy safeguards

The current MSB-GP iteration adds prediction-first safeguards inside the
same public `MSBCal` workflow:

- **Sparse-additive rescue:** a training-only screened additive kernel is
  evaluated by inner out-of-fold prediction and used only when it improves
  accuracy over the dense additive/dominance core. This targets sparse or
  heterogeneous additive architectures without changing the model name.
- **Dominance-aware gate:** when the core model is already dominance-heavy, the
  sparse-additive rescue must exceed a stricter out-of-fold threshold, so true
  dominance or epistatic signal is not replaced by fold-specific marker noise.
- **Small-data guard:** automatic sparse-additive rescue is disabled for very
  small pilot datasets unless explicitly requested, reducing overfitting risk.
- **Native epistasis candidate:** a training-residual sparse AA/AD/DD kernel
  candidate is evaluated as part of the same MSB-GP selector, so higher-order
  signal can be retained without invoking any external predictor.
- **Residual sparse AA/AD/DD selector:** top markers are screened from core
  residuals, a compact set of AA/AD/DD interaction features is selected, and
  only those selected features are passed to the native epistasis candidate.
  The selector uses training labels only and is disabled as a public model name;
  it is an internal layer of MSB-GP.
