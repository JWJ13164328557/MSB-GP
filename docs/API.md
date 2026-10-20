# API summary

## Official model

```python
from msbgp import MSBCal
model = MSBCal(k_inner=5, seed=0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

`MSBCal` is the calibrated implementation of the official MSB-GP workflow.
By default it is self-contained: all candidate predictors are native MSB-GP
multi-kernel models (`core`, `sparse`, `epi`, `full`), and GBLUP/RR-BLUP/GMKL
are not called internally. The optional `aux_candidates` argument exists only
for explicit benchmark studies.

## Genotype I/O

```python
import numpy as np
from msbgp import save_genotype_memmap, load_genotype_memmap, genotype_memmap_to_csr
X_mem = save_genotype_memmap(X, "Genotypes.dat", dtype=np.int8)
model = MSBCal(k_inner=3).fit(X_mem, y)
X_csr = genotype_memmap_to_csr(X_mem, chunk_size=2048)
```

`save_genotype_memmap` writes an int8 `.npy` file or a raw `.dat` file plus JSON sidecar. The model consumes memmaps directly and follows the same exact dense kernel/REML equations; this is a storage and peak-memory optimisation, not an approximation.

## Kernel weights

```python
model.get_kernel_weights()
```

Returns the learned softmax kernel shares.

## Calibration coefficients

```python
model.a_
model.b_
```

## Uncalibrated ablation

```python
from msbgp import MSBGP
model = MSBGP()
```

`MSBGP` is useful for diagnostics and ablation, but it is not the official
calibrated output model.

## Additive-only ablation

```python
model = MSBCal(kernels=["additive"])
```

Use this only as an ablation baseline.

## Diagnostics

```python
from msbgp import MSBGP
model = MSBGP().fit(X_train, y_train)

model.get_kernel_correlations()      # pairwise kernel-correlation diagnostics
model.get_dropped_kernels()          # kernels removed when redundancy="drop"
model.get_model_heritability()       # total genetic-variance ratio
model.get_genetic_variance_ratio()   # same model-level ratio
model.get_reliability()              # PEV-based training reliability using GPG
model.get_rsi_stability()            # strict RSI frequency diagnostics
model.optim_success_                 # REML optimizer convergence flag
model.optim_message_                 # optimizer message
model.optim_gradient_norm_           # final gradient norm
model.optim_niter_                   # optimizer iterations
model.optim_boundary_parameters_     # parameters at bounds
model.optim_start_label_             # selected REML start label
model.optim_n_starts_                # number of REML starts evaluated
```

Optional kernel redundancy control:

```python
model = MSBGP(
    kernel_correlation_threshold=0.98,
    kernel_redundancy="drop",       # or "warn" (default) / "none"
)
```

## Compatibility

```python
model.get_heritability()             # compatibility alias; not narrow-sense h2
```

## REML initialization and convergence diagnostics

```python
model = MSBCal(
    reml_init="heuristic",   # "single" (default), "heuristic", or "multi_start"
    reml_multistart=3,       # used when reml_init="multi_start"
    reml_max_eval=250,
    reml_gtol=1e-5,
)
```

Multi-start REML is deterministic. The optimizer stores `optim_success_`, `optim_message_`, `optim_gradient_norm_`, `optim_niter_`, `optim_boundary_parameters_`, `optim_start_label_`, and `optim_n_starts_` on the fitted model. These options change only the optimisation path and diagnostics, not the model formula.

## Accuracy controls

```python
model = MSBCal(
    sparse_additive_top_markers="auto",  # or int / None
    auto_sparse_additive=True,
    sparse_gate_threshold=0.04,
    sparse_gate_threshold_high=0.12,
    sparse_high_dominance_share=0.30,
    sparse_full_gate_margin=0.005,
)
model.fit(X_train, y_train)
model.get_gate_diagnostics()
```

`sparse_additive_top_markers="auto"` selects a training-only screened
sparse-additive candidate. It is accepted only when it improves leakage-free
inner out-of-fold Pearson correlation. The stricter high-dominance threshold is
used automatically when the core model is dominance-heavy.

The default selector also evaluates a native `epi` candidate built from
`additive + dominance + AA/AD/DD` kernels when `kernels="auto"`.



