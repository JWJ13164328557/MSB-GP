# Variance-reduction options

The public method is **MSB-GP**. The variance-reduction features described here
are internal options of the official `MSBCal` workflow and are not exposed as
separate models.

## Defaults

By default `MSBCal` uses a **single** inner CV partition for calibration and a
**single** model fit:

```python
from msbgp import MSBCal

model = MSBCal(kernels="auto", k_inner=5, seed=0)
```

## Optional: repeated leakage-free calibration

Set `n_inner_repeats > 1` to estimate the affine correction from several
independent inner CV partitions and average the coefficients. This reduces
dependence on one arbitrary inner split:

```python
model = MSBCal(k_inner=5, seed=0, n_inner_repeats=3)
```

## Optional: homogeneous bagged ensemble

Set `n_ensemble > 1` to train several MSB-GP fits on subsampled training sets
(each drawing `ensemble_sample_fraction` of the training samples) and average
their predictions. This preserves the public method name while reducing
prediction variance:

```python
model = MSBCal(k_inner=5, seed=0, n_ensemble=3, ensemble_sample_fraction=0.80)
```

Both options can be enabled together:

```python
model = MSBCal(
    kernels="auto",
    k_inner=5,
    seed=0,
    n_inner_repeats=3,
    n_ensemble=3,
    ensemble_sample_fraction=0.80,
)
```

## Notes

- Turning these options on increases runtime roughly in proportion to
  `n_inner_repeats * n_ensemble`; they are optional variance-reduction controls,
  not part of the default workflow.
- The residual sparse interaction (RSI) layer is opt-in and disabled by
  default (`rsi=False`).
- Leakage-free calibration is always applied: the affine correction is fitted on
  inner out-of-fold predictions only, and test labels are never used.
