"""msbgp: A toolkit for developing novel Genomic Selection algorithms.

This package integrates classical BLUP/Bayesian methods, kernel methods and
adaptive model-selection ideas into a set of reusable components, and provides
a new multi-kernel sparse genomic prediction algorithm (MSB-GP) with marginal-likelihood adaptive kernel weighting and
built-in leakage-free output calibration (MSBCal).
"""

from .kernels import (
    additive_kernel,
    dominance_kernel,
    gaussian_kernel,
    polynomial_kernel,
    build_kernel_library,
)
from .bases import GBLUP, RRBLUP, BayesA, BayesC
from .msbgp import MSBGP
from .gmkls import MeanGMKL, FHGMKL, CKAGMKL
from .calibration import MSBCal
from .ml_baselines import RandomForest, MLPRegressor
from .dnngp import DNNGP
from .metrics import (
    pearson_corr,
    prediction_accuracy,
    genetic_accuracy,
    neg_predictive_val,
)
from .data import (simulate_gs_data, save_genotype_memmap, load_genotype_memmap, validate_genotype_memmap, genotype_memmap_to_csr)
from .cv import heldout_split, random_kfold, run_methods

__all__ = [
    "additive_kernel",
    "dominance_kernel",
    "gaussian_kernel",
    "polynomial_kernel",
    "build_kernel_library",
    "GBLUP",
    "RRBLUP",
    "BayesA",
    "BayesC",
    "MSBGP",
    "MeanGMKL",
    "FHGMKL",
    "CKAGMKL",
    "MSBCal",
    "RandomForest",
    "MLPRegressor",
    "DNNGP",
    "pearson_corr",
    "prediction_accuracy",
    "genetic_accuracy",
    "neg_predictive_val",
    "simulate_gs_data",
    "save_genotype_memmap",
    "load_genotype_memmap",
    "validate_genotype_memmap",
    "genotype_memmap_to_csr",
    "heldout_split",
    "random_kfold",
    "run_methods",
]

__version__ = "1.0.0"




