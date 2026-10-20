"""Cross-validation harness for Genomic Selection benchmarking.

The standard GS evaluation scheme is *random* k-fold CV (also 'genomic
prediction' uses a training/test split where test individuals have their
phenotypes masked).  This module provides a uniform interface so every method
(GS classic, ML, and the new MSB-GP) is evaluated identically.
"""

import numpy as np

from .metrics import prediction_accuracy

__all__ = ["random_kfold", "heldout_split", "run_methods"]


def _make_folds(n, k, seed, groups=None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def heldout_split(n, test_frac=0.2, seed=0):
    """Single random training/test split (train -> fit, test -> predict)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test = idx[:n_test]
    train = idx[n_test:]
    return train, test


def random_kfold(X, y, method, k=5, seed=0, verbose=False, on_fold=None,
                 **fit_kwargs):
    """Run random k-fold CV with a single model instance.

    Parameters
    ----------
    X, y : data (n, m) and (n,)
    method : an object with .fit(X_tr, y_tr) and .predict(X_va)
    k : number of folds
    seed : random seed
    on_fold : optional callable(fold_index, method) invoked after each fold's
        fit() so callers can inspect the freshly-fitted model (e.g. MSB-GP
        kernel weights) before it is re-fit on the next fold.

    Returns
    -------
    dict with 'accuracy' (list per fold), 'mean', 'std', and the pooled
    predictions.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    folds = _make_folds(n, k, seed)
    accs = []
    pooled_pred = np.zeros(n)
    pooled_true = np.zeros(n)
    for fi, order in enumerate(folds):
        test = np.sort(order)
        train = np.setdiff1d(np.arange(n), test)
        method.fit(X[train], y[train], **fit_kwargs)
        if on_fold is not None:
            on_fold(fi, method)
        yhat = np.asarray(method.predict(X[test])).ravel()
        pooled_pred[test] = yhat
        pooled_true[test] = y[test]
        r = prediction_accuracy(y[test], yhat)
        accs.append(r)
        if verbose:
            print(f"  fold {fi+1}/{k}: accuracy = {r:.4f}")
    return {
        "accuracy": accs,
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "pooled_pred": pooled_pred,
        "pooled_true": pooled_true,
    }


def run_methods(X, y, methods, k=5, seed=0, verbose=False):
    """Run several methods through the same CV and collect results."""
    results = {}
    for name, method in methods.items():
        if verbose:
            print(f"--- {name} ---")
        results[name] = random_kfold(X, y, method, k=k, seed=seed,
                                     verbose=verbose)
    return results
