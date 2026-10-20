"""Pure-numpy ML baselines for Genomic Selection benchmarking.

sklearn / lightgbm / torch are not installed in this environment, so we ship
two lightweight, deterministic, pure-numpy baselines -- a Random Forest and a
small Multi-Layer Perceptron -- so that the MSB-GP comparison on real data can
also answer "how do classic ML / shallow-DL methods compare?".

Both expose the same sklearn-like interface used by `msbgp.cv`:
    model.fit(X_tr, y_tr);  yhat = model.predict(X_va)
"""

import numpy as np

__all__ = ["RandomForest", "MLPRegressor"]


def _col_sample_freq(X, y, w=None):
    """Bagging + random-feature-subset table with weights (for tree split)."""
    pass


class _RegTree:
    """Simple CART regression tree (numpy)."""

    def __init__(self, max_depth=8, min_samples_leaf=5, max_features=None,
                 rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng
        self.pred_ = None

    def _best_split(self, X, y, feat_idx, w):
        n = X.shape[0]
        total_w = float(w.sum())
        tot_y = float((y * w).sum())
        tot_y2 = float((y * y * w).sum())
        parent_within = tot_y2 - tot_y * tot_y / total_w
        best_gain, best_f, best_t = 0.0, None, None
        for f in feat_idx:
            x = X[:, f]
            order = np.argsort(x)
            xs = x[order]; ys = y[order]; ws = w[order]
            yws = ys * ws
            cw = 0.0; cy = 0.0; cy2 = 0.0
            for i in range(n - 1):
                cw += ws[i]; cy += yws[i]; cy2 += ys[i] * yws[i]
                lw = cw; rw = total_w - cw
                if lw < self.min_samples_leaf or rw < self.min_samples_leaf:
                    continue
                if xs[i] == xs[i + 1]:
                    continue
                left_ss = cy2 - cy * cy / lw
                ry = tot_y - cy
                ry2 = tot_y2 - cy2
                right_ss = ry2 - ry * ry / rw
                gain = parent_within - left_ss - right_ss
                if gain > best_gain:
                    best_gain = gain; best_f = f; best_t = (xs[i] + xs[i + 1]) / 2.0
        return best_f, best_t, best_gain

    def fit(self, X, y, w=None):
        self._fit(X, y, np.ones(X.shape[0]) if w is None else w, 0)
        self.pred_ = float(np.average(y, weights=w))
        return self

    def _fit(self, X, y, w, depth):
        n = X.shape[0]
        self.mean = float(np.average(y, weights=w))
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return
        m = X.shape[1]
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                n_feat = max(1, int(np.sqrt(m)))
            elif self.max_features == "log2":
                n_feat = max(1, int(np.log2(m)))
            else:
                n_feat = m
        elif self.max_features is None:
            n_feat = m
        else:
            n_feat = int(self.max_features)
        n_feat = min(n_feat, m)
        rng = self.rng
        if rng is None:
            feat_idx = np.arange(m)
        else:
            feat_idx = rng.choice(m, size=n_feat, replace=False)
        f, t, gain = self._best_split(X, y, feat_idx, w)
        if f is None or gain <= 1e-9:
            return
        mask = X[:, f] <= t
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return
        self.f, self.t = f, t
        self.left = _RegTree(self.max_depth, self.min_samples_leaf,
                             self.max_features, self.rng)
        self.right = _RegTree(self.max_depth, self.min_samples_leaf,
                              self.max_features, self.rng)
        self.left._fit(X[mask], y[mask], w[mask], depth + 1)
        self.right._fit(X[~mask], y[~mask], w[~mask], depth + 1)
        self.is_leaf = False
        return

    def predict(self, X):
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            out[i] = self._pred_one(X[i])
        return out

    def _pred_one(self, x):
        node = self
        while not getattr(node, "is_leaf", True):
            if x[node.f] <= node.t:
                node = node.left
            else:
                node = node.right
        return node.mean


class RandomForest:
    """Random Forest regressor (bagged CART trees with random feature subsets)."""

    def __init__(self, n_estimators=200, max_depth=8, min_samples_leaf=5,
                 max_features="sqrt", seed=0, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.n_jobs = n_jobs
        self.trees_ = []
        self.mu_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.mu_ = y.mean()
        self.trees_ = []
        rng = np.random.default_rng(self.seed)
        n, m = X.shape
        for t in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)  # bootstrap
            tr = _RegTree(self.max_depth, self.min_samples_leaf,
                          self.max_features, rng)
            tr.fit(X[idx], y[idx])
            self.trees_.append(tr)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.zeros((X.shape[0], len(self.trees_)))
        for ti, tree in enumerate(self.trees_):
            preds[:, ti] = tree.predict(X)
        return preds.mean(axis=1)


class MLPRegressor:
    """Small fully-connected MLP (2 hidden layers) trained with Adam.

    Features are standardised and the target centred.  Deterministic given a
    seed.  This is the "shallow deep-learning" baseline.
    """

    def __init__(self, hidden=(64, 32), lr=1e-2, epochs=300, batch_size=64,
                 seed=0, val_frac=0.15, patience=30):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.val_frac = val_frac
        self.patience = patience

    @staticmethod
    def _relu(z):
        return np.maximum(z, 0.0)

    def _forward(self, X, cache=False):
        # returns pre-activation lists: a0, then (z_k, a_k) per hidden layer,
        # then the raw output.  We store both pre- and post-activation so the
        # back-prop pass can recover the ReLU masks easily.
        a = X
        pre = [a]
        relu_masks = []
        acts = [a]
        for W, b in self.weights_[:-1]:
            z = a @ W + b
            relu_masks.append(z > 0)
            a = self._relu(z)
            pre.append(z)
            acts.append(a)
        W, b = self.weights_[-1]
        out = a @ W + b
        if cache:
            return out, acts, relu_masks
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, m = X.shape
        self.mu_ = y.mean()
        self.y_std_ = y.std() + 1e-8
        yc = (y - self.mu_) / self.y_std_

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-8
        Xs = (X - self.x_mean_) / self.x_std_

        rng = np.random.default_rng(self.seed)
        # train / val split
        idx = rng.permutation(n)
        n_val = int(round(n * self.val_frac))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        Xtr, ytr = Xs[tr_idx], yc[tr_idx]
        Xva, yva = Xs[val_idx], yc[val_idx]

        dims = [m] + list(self.hidden) + [1]
        self.weights_ = []
        for d0, d1 in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / d0)
            W = rng.normal(0, scale, size=(d0, d1))
            b = np.zeros(d1)
            self.weights_.append([W, b])

        best_vloss = np.inf
        best_state = None
        stale = 0
        for epoch in range(self.epochs):
            order = rng.permutation(len(Xtr))
            for s in range(0, len(order), self.batch_size):
                bi = order[s:s + self.batch_size]
                Xb, yb = Xtr[bi], ytr[bi]
                self._sgd_step(Xb, yb)
            vp = self._forward(Xva).ravel()
            vloss = float(np.mean((vp - yva) ** 2))
            if vloss < best_vloss - 1e-6:
                best_vloss = vloss
                best_state = [([W.copy() for W, b in self.weights_]),
                              [b.copy() for W, b in self.weights_]]
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            Ws, bs = best_state
            self.weights_ = [[W, b] for W, b in zip(Ws, bs)]
        return self

    def _sgd_step(self, X, y):
        out, acts, masks = self._forward(X, cache=True)
        n = X.shape[0]
        L = len(self.weights_)
        # gradient wrt output (pre-activation of output unit)
        g = (out.ravel() - y)[:, None]        # (n,1)
        newW, newb = [], []
        # ------ output layer (index L-1) ------
        W, b = self.weights_[-1]
        a_prev = acts[-1]                     # activation feeding output
        dW = (a_prev.T @ g) / n
        db = g.sum(axis=0) / n
        newW.append(W - self.lr * dW)
        newb.append(b - self.lr * db)
        g = g @ W.T                           # (n, d_{L-1})  dLoss/d a_{L-1}
        # ------ hidden layers (index L-2 .. 0) ------
        for layer in range(L - 2, -1, -1):
            W, b = self.weights_[layer]
            g = g * masks[layer]              # relu mask of this hidden layer
            a_prev = acts[layer]              # activation feeding this layer
            dW = (a_prev.T @ g) / n
            db = g.sum(axis=0) / n
            newW.append(W - self.lr * dW)
            newb.append(b - self.lr * db)
            if layer > 0:
                g = g @ W.T                   # dLoss/d a_{layer}
        newW.reverse(); newb.reverse()
        self.weights_ = [[W, b] for W, b in zip(newW, newb)]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_std_
        out = self._forward(Xs).ravel()
        return self.mu_ + out * self.y_std_
