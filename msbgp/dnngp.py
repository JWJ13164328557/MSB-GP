# -*- coding: utf-8 -*-
"""DNNGP baseline: deep neural network for genomic prediction (pure NumPy).

Reference
---------
Wang et al. (2022) "DNNGP: a deep neural network for genomic prediction",
Frontiers in Plant Science: standardised SNPs -> two hidden layers
(200, 40 units) with ReLU, batch normalisation and dropout (0.4) -> linear
output; Adam optimiser with early stopping on a validation split.

This implementation uses NumPy only (no TensorFlow dependency) with
single-threaded BLAS, matching the deterministic, single-thread convention
of the other msbgp models.

Interface matches the other msbgp models: fit(X, y) / predict(X).
"""
import numpy as np

__all__ = ["DNNGP"]


def _bn_forward(z, gamma, beta, rm, rv, training, momentum=0.9, eps=1e-5):
    if training:
        mu = z.mean(axis=0)
        var = z.var(axis=0)
        rm[:] = momentum * rm + (1 - momentum) * mu
        rv[:] = momentum * rv + (1 - momentum) * var
    else:
        mu, var = rm, rv
    zhat = (z - mu) / np.sqrt(var + eps)
    return zhat * gamma + beta, (zhat, mu, var)


def _bn_backward(dout, cache, gamma, beta):
    zhat, mu, var = cache
    n = dout.shape[0]
    dgamma = (dout * zhat).sum(axis=0)
    dbeta = dout.sum(axis=0)
    dzhat = dout * gamma
    inv_std = 1.0 / np.sqrt(var + 1e-5)
    dz = (1.0 / n) * inv_std * (
        n * dzhat - dzhat.sum(axis=0) - zhat * (dzhat * zhat).sum(axis=0))
    return dz, dgamma, dbeta


class DNNGP:
    """DNNGP with the reference architecture (200 -> 40 hidden units)."""

    def __init__(self, hidden=(200, 40), dropout=0.4, lr=1e-3,
                 epochs=300, batch_size=64, patience=20, seed=0, l2=0.0):
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.l2 = l2
        self.W_ = None

    # ------------------------------------------------------------------ #
    def _init_params(self, m):
        rng = np.random.default_rng(self.seed)
        sizes = [m] + list(self.hidden) + [1]
        W, b, gamma, beta, rm, rv = [], [], [], [], [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            W.append(rng.normal(0, np.sqrt(2.0 / fan_in),
                                size=(sizes[i], sizes[i + 1])))
            b.append(np.zeros(sizes[i + 1]))
            if i < len(sizes) - 2:          # hidden layers only
                gamma.append(np.ones(sizes[i + 1]))
                beta.append(np.zeros(sizes[i + 1]))
                rm.append(np.zeros(sizes[i + 1]))
                rv.append(np.ones(sizes[i + 1]))
        return {"W": W, "b": b, "gamma": gamma, "beta": beta,
                "rm": rm, "rv": rv}

    def _forward(self, X, p, training, rng=None):
        W, b = p["W"], p["b"]
        L = len(W)
        caches = []
        a = X
        for i in range(L):
            z = a @ W[i] + b[i]
            if i < L - 1:                                   # hidden layer
                bn_cache = (zhat, mu, var) = None, None, None  # placeholder
                out, c = _bn_forward(z, p["gamma"][i], p["beta"][i],
                                     p["rm"][i], p["rv"][i], training)
                h = np.maximum(out, 0)                       # ReLU
                if training and self.dropout > 0:
                    mask = (rng.random(h.shape) >= self.dropout) / \
                           max(1.0 - self.dropout, 1e-8)
                else:
                    mask = None
                a_next = h * mask if mask is not None else h
                caches.append(("hidden", a, z, c, out, mask))
                a = a_next
            else:
                caches.append(("out", a, z, None, None, None))
                a = z.ravel()
        return a, caches

    def _backward(self, dy, caches, p):
        W = p["W"]
        grads = {"W": [None] * len(W), "b": [None] * len(W),
                 "gamma": [None] * len(p["gamma"]),
                 "beta": [None] * len(p["beta"])}
        d = dy.astype(np.float64).reshape(-1, 1)             # (n, 1)
        for i in range(len(W) - 1, -1, -1):
            kind, a_in, z, c, out, mask = caches[i]
            if kind == "out":
                grads["W"][i] = a_in.T @ d + self.l2 * W[i]
                grads["b"][i] = d.sum(axis=0)
                da = d @ W[i].T
            else:
                dh = d * mask if mask is not None else d
                dout = dh * (out > 0)                        # ReLU grad
                dz, dgamma, dbeta = _bn_backward(dout, c, p["gamma"][i],
                                                 p["beta"][i])
                grads["gamma"][i] = dgamma
                grads["beta"][i] = dbeta
                grads["W"][i] = a_in.T @ dz + self.l2 * W[i]
                grads["b"][i] = dz.sum(axis=0)
                da = dz @ W[i].T
            d = da
        return grads

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        Xs = (X - self.x_mean_) / self.x_std_
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        ys = (y - self.y_mean_) / self.y_std_

        n = Xs.shape[0]
        n_val = max(int(0.1 * n), 1)
        idx = np.random.default_rng(self.seed).permutation(n)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr, ytr = Xs[tr_idx], ys[tr_idx]
        Xva, yva = Xs[val_idx], ys[val_idx]

        p = self._init_params(Xs.shape[1])
        # Adam state
        m = {k: [np.zeros_like(v) for v in p[k]] for k in ("W", "b", "gamma", "beta")}
        v = {k: [np.zeros_like(x) for x in p[k]] for k in ("W", "b", "gamma", "beta")}
        t = 0
        best_val = np.inf
        best_params = None
        bad = 0
        rng = np.random.default_rng(self.seed + 1)

        for ep in range(self.epochs):
            order = rng.permutation(len(ytr))
            for s in range(0, len(order), self.batch_size):
                bi = order[s:s + self.batch_size]
                xb, yb = Xtr[bi], ytr[bi]
                pred, caches = self._forward(xb, p, training=True, rng=rng)
                d = 2.0 * (pred - yb) / len(bi)
                grads = self._backward(d, caches, p)
                t += 1
                for k in ("W", "b", "gamma", "beta"):
                    for i in range(len(p[k])):
                        g = grads[k][i]
                        m[k][i] = 0.9 * m[k][i] + 0.1 * g
                        v[k][i] = 0.999 * v[k][i] + 0.001 * g * g
                        mh = m[k][i] / (1 - 0.9 ** t)
                        vh = v[k][i] / (1 - 0.999 ** t)
                        p[k][i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            pv, _ = self._forward(Xva, p, training=False)
            val_mse = float(np.mean((pv - yva) ** 2))
            if val_mse < best_val - 1e-6:
                best_val = val_mse
                best_params = {k: [x.copy() for x in p[k]]
                               for k in ("W", "b", "gamma", "beta", "rm", "rv")}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_params is not None:
            p = best_params
        self.W_ = p
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.x_mean_) / self.x_std_
        out, _ = self._forward(Xs, self.W_, training=False)
        return out * self.y_std_ + self.y_mean_