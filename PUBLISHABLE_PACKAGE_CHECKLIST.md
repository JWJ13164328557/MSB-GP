# Publishable package checklist

## Package identity
- Distribution name: `MSB-GP`
- Version: `1.0.0`
- Import package: `msbgp`
- Public model: `MSBCal` (the calibrated MSB-GP workflow)
- Core estimator: `MSBGP`
- Python: >=3.9
- License: MIT

## Included
- Core package under `msbgp/`
- Benchmark and analysis scripts under `experiments/`
- Example workflow under `examples/`
- Tests under `tests/`
- Methods, API and reproduction documentation under `docs/`
- Data schema description under `data/`
- Packaging metadata: `pyproject.toml`, `MANIFEST.in`, `requirements.txt`
- Citation metadata: `CITATION.cff`
- Release notes: `RELEASE_NOTES.md`

## Verification
- `python -m compileall -q .`
- `python tests/run_tests.py`
- `python examples/quickstart.py`

## Recommended release command
```bash
python -m pip install -e .
python -m pytest -q
python examples/quickstart.py
```

## Publication-ready outputs
Benchmark outputs used in the paper are stored separately under:

```text
H:/现代农业研究院李博生老师课题组/AI育种大赛/results
```

The package itself does not redistribute protected real-data files.
