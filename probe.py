from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ATTN_DIM = 6 * 14 * 8
HIDDEN_DIM = 896 + 5
TOTAL_DIM = ATTN_DIM + HIDDEN_DIM
PCA_K = 64


class _BootstrapLR:
    def __init__(self, n_seeds=30, C=0.003, seed=123):
        self.n_seeds = n_seeds; self.C = C; self.seed = seed
        self.models: list[LogisticRegression] = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); n = X.shape[0]; self.models = []
        for _ in range(self.n_seeds):
            idx = rng.choice(n, size=n, replace=True)
            m = LogisticRegression(C=self.C, penalty="l2", max_iter=2000, n_jobs=1)
            m.fit(X[idx], y[idx]); self.models.append(m)
        return self

    def predict_proba_pos(self, X):
        return np.stack([m.predict_proba(X)[:, 1] for m in self.models], axis=0).mean(axis=0)


class _MetaBLR:
    PARENT_SEEDS = (42, 43, 44, 45, 46)

    def __init__(self, n_seeds=30, C=0.003):
        self.n_seeds = n_seeds; self.C = C
        self.bags: list[_BootstrapLR] = []

    def fit(self, X, y):
        self.bags = [_BootstrapLR(self.n_seeds, self.C, ps).fit(X, y) for ps in self.PARENT_SEEDS]
        return self

    def predict_proba_pos(self, X):
        return np.stack([b.predict_proba_pos(X) for b in self.bags], axis=0).mean(axis=0)


class HallucinationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler_h = StandardScaler()
        self._pca: PCA | None = None
        self._scaler = StandardScaler()
        self._bag: _MetaBLR | None = None
        self._threshold = 0.5

    def _build_network(self, input_dim):
        self._net = nn.Sequential(nn.Linear(input_dim, 1))

    def forward(self, x):
        if self._net is None:
            raise RuntimeError("Call fit() first.")
        return self._net(x).squeeze(-1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int32)
        if X.shape[1] != TOTAL_DIM:
            raise ValueError(f"Expected feature dim {TOTAL_DIM}, got {X.shape[1]}.")
        self._build_network(X.shape[1])
        attn = X[:, :ATTN_DIM]
        hid = X[:, ATTN_DIM:]
        self._scaler_h = StandardScaler().fit(hid)
        hid_std = self._scaler_h.transform(hid)
        self._pca = PCA(n_components=PCA_K, random_state=0).fit(hid_std)
        hid_pca = self._pca.transform(hid_std)
        Xc = np.hstack([attn, hid_pca]).astype(np.float32)
        self._scaler = StandardScaler().fit(Xc)
        self._bag = _MetaBLR(n_seeds=30, C=0.003).fit(self._scaler.transform(Xc), y)
        return self

    def fit_hyperparameters(self, X_val, y_val):
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X):
        if self._bag is None:
            raise RuntimeError("Probe not fitted.")
        X = np.asarray(X, dtype=np.float32)
        attn = X[:, :ATTN_DIM]
        hid = X[:, ATTN_DIM:]
        hid_std = self._scaler_h.transform(hid)
        hid_pca = self._pca.transform(hid_std)
        Xc = np.hstack([attn, hid_pca]).astype(np.float32)
        p = self._bag.predict_proba_pos(self._scaler.transform(Xc))
        return np.stack([1.0 - p, p], axis=1)
