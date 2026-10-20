# Reproducing results

## Environment

Use Python 3.9 or later.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Smoke test

```bash
python tests/run_tests.py
```

## Simulation benchmark

```bash
python experiments/run_simulation_benchmark.py
```

Outputs include:

- `simulation_results.csv`
- `best_methods.csv`
- `summary_with_vs_without_rsi.csv`
- `fig_simulation.png`
- `fig_simulation.pdf`
- `simulation.log`

## Real-data benchmark

Prepare one folder per species:

```text
Species/
  Genotypes.csv
  Phenotypes.csv
```

Then run:

```bash
python experiments/run_real_data_full.py
python experiments/run_benchmark_metrics.py
python experiments/run_real_data_mse.py
```

For a new species:

```bash
python experiments/run_new_species.py --data-dir "H:/.../NewSpecies" --output-dir "H:/.../results/new_species"
```

## Reproducibility checklist

1. Use the same random seed.
2. Confirm `python -c "import msbgp; print(msbgp.__version__)"` returns the MSB-GP version.
3. Report the number of folds, repeats, marker count and sample count.
4. Keep `simulation.log` and all CSV outputs.
5. For paired significance, report the fold-level paired test.


