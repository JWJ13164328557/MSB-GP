"""Prediction-quality metrics for Genomic Selection evaluation.

The single most important metric in GS is *prediction accuracy*: the
correlation between predicted and observed breeding values in a validation
set, Pearson-corrected by the square root of heritability for an
unbiased estimate of true accuracy.
"""

import numpy as np

__all__ = ["pearson_corr", "prediction_accuracy", "neg_predictive_val",
           "genetic_accuracy"]


def pearson_corr(y_true, y_pred):
    """Pearson correlation between observed and predicted values."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yt = yt - yt.mean()
    yp = yp - yp.mean()
    denom = np.sqrt((yt @ yt) * (yp @ yp))
    if denom == 0:
        return 0.0
    return float((yt @ yp) / denom)


def prediction_accuracy(y_true, y_pred):
    """Correlation-based prediction accuracy (== pearson correlation here;
    kept as a named alias used throughout the GS literature)."""
    return pearson_corr(y_true, y_pred)


def genetic_accuracy(y_true, y_pred, heritability=None):
    """True genetic accuracy: Pearson correlation divided by sqrt(h2).

    When heritability is unknown (e.g. validation has observed phenotypes),
    heritability can be estimated from the training set.  To avoid artefacts
    this helper only divides by sqrt(h2) when h2 is explicitly provided.
    """
    r = pearson_corr(y_true, y_pred)
    if heritability:
        return r / np.sqrt(heritability)
    return r


def neg_predictive_val(y_true, y_pred):
    """Negative predictive value analogue for continuous outcomes not used
    directly; kept for interface symmetry."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
