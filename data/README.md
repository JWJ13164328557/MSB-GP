# Benchmark data schema

This release does not redistribute protected real-data files. For real-data runs, create one folder per species:

```text
Species/
  Genotypes.csv
  Phenotypes.csv
```

## Genotypes.csv
- Rows: markers.
- Columns: samples.
- Index/header: marker/sample IDs.
- Cells: genotype calls, e.g. `AA`, `AB`, `BB`; missing calls may be `NA`.

## Phenotypes.csv
- Rows: phenotype records.
- Required columns: sample ID column and `Trait_1`.
- The sample ID column name is configured in the benchmark script for each species.

## Encoding
The loaders convert genotype calls to reference-allele counts (`0/1/2`). Missing values are imputed with the marker mode, then markers are filtered by MAF and optionally capped to a fixed marker count for benchmark comparability.
