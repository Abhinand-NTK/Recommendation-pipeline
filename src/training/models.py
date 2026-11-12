"""Recommender models.

All models share one interface so training, evaluation and serving are generic:

    fit(train_df, feature_store)
    score(user_ids)            -> (n_users, n_items) ranking scores over self.item_ids
    predict_rating(u, i)       -> explicit rating predictions for (user, item) pairs
    recommend(user_ids, k, seen)

Models
    PopularityRecommender  - non-personalised baseline (most-rated items)
    MatrixFactorization    - biased explicit MF (ALS) - predicts ratings (RMSE/MAE)
    ImplicitALS            - implicit-feedback MF (Hu, Koren & Volinsky 2008) - ranking
    HybridRecommender      - implicit-ALS score + content (genre affinity from the
                             feature store) + popularity prior, blended on z-scores
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.feature_store import FeatureStore


class BaseRecommender:
    name = "base"

    def __init__(self, **params):
        self.params = params
        self.item_ids: np.ndarray | None = None
        self.user_ids: np.ndarray | None = None
        self.global_mean = 3.5

    # -- index helpers
    def _index(self, train: pd.DataFrame):
        self.user_ids = np.sort(train["user_id"].unique())
        self.item_ids = np.sort(train["item_id"].unique())
        self.u_pos = pd.Series(np.arange(len(self.user_ids)), index=self.user_ids)
        self.i_pos = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)
        self.global_mean = float(train["rating"].mean())

    def fit(self, train: pd.DataFrame, fs: FeatureStore):
        raise NotImplementedError

    def score(self, user_ids) -> np.ndarray:
        raise NotImplementedError

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_ids, k: int = 10, seen: dict | None = None) -> np.ndarray:
        """Top-k item ids per user, excluding items already seen in training."""
        s = self.score(user_ids).astype(np.float64, copy=True)
        if seen:
            for row, u in enumerate(user_ids):
                idx = seen.get(u)
                if idx is not None and len(idx):
                    s[row, idx] = -np.inf
        top = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]
        order = np.take_along_axis(s, top, axis=1).argsort(axis=1)[:, ::-1]
        return self.item_ids[np.take_along_axis(top, order, axis=1)]

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / "model.pkl"
        state = {k: v for k, v in self.__dict__.items() if k != "fs"}
        with open(p, "wb") as fh:
            pickle.dump({"cls": type(self).__name__, "state": state}, fh)
        return p

    @staticmethod
    def load(directory: Path, fs: FeatureStore | None = None) -> "BaseRecommender":
        with open(Path(directory) / "model.pkl", "rb") as fh:
            blob = pickle.load(fh)
        obj = MODEL_REGISTRY[blob["cls"]].__new__(MODEL_REGISTRY[blob["cls"]])
        obj.__dict__.update(blob["state"])
        if hasattr(obj, "_needs_fs"):
            obj.fs = fs or FeatureStore()
        return obj


# --------------------------------------------------------------------------- baseline
class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def fit(self, train, fs):
        self._index(train)
        itf = fs.get_item_features(self.item_ids)
        self.pop = itf["item_popularity_count"].to_numpy(np.float64)
        self.rating = itf["item_bayesian_avg_rating"].fillna(self.global_mean).to_numpy(np.float64)
        return self

    def score(self, user_ids):
        return np.tile(self.pop, (len(user_ids), 1))

    def predict_rating(self, user_ids, item_ids):
        pos = self.i_pos.reindex(item_ids).to_numpy()
        out = np.full(len(item_ids), self.global_mean)
        ok = ~np.isnan(pos)
        out[ok] = self.rating[pos[ok].astype(int)]
        return out


# --------------------------------------------------------------------------- MF (ALS)
class MatrixFactorization(BaseRecommender):
    """Biased MF:  r_ui ~ mu + b_u + b_i + p_u . q_i  (explicit ALS)."""
    name = "matrix_factorization"

    def fit(self, train, fs):
        self._index(train)
        k = self.params.get("factors", 64)
        iters = self.params.get("iterations", 15)
        reg = self.params.get("reg", 0.08)
        breg = self.params.get("bias_reg", 10.0)
        rng = np.random.default_rng(self.params.get("seed", 42))

        u = self.u_pos[train["user_id"]].to_numpy()
        i = self.i_pos[train["item_id"]].to_numpy()
        r = train["rating"].to_numpy(np.float64)
        nu, ni = len(self.user_ids), len(self.item_ids)
        mu = r.mean()

        # 1) regularised biases (a few alternating passes)
        bu, bi = np.zeros(nu), np.zeros(ni)
        cnt_u, cnt_i = np.bincount(u, minlength=nu), np.bincount(i, minlength=ni)
        for _ in range(10):
            bi = np.bincount(i, weights=r - mu - bu[u], minlength=ni) / (breg + cnt_i)
            bu = np.bincount(u, weights=r - mu - bi[i], minlength=nu) / (breg + cnt_u)
        res = r - mu - bu[u] - bi[i]

        # 2) ALS on the residuals
        R = sp.csr_matrix((res, (u, i)), shape=(nu, ni))
        Rt = R.T.tocsr()
        P = rng.normal(0, 0.1, (nu, k))
        Q = rng.normal(0, 0.1, (ni, k))
        eye = np.eye(k)

        def solve(M, fixed, out):
            for row in range(M.shape[0]):
                s, e = M.indptr[row], M.indptr[row + 1]
                if s == e:
                    continue
                cols, vals = M.indices[s:e], M.data[s:e]
                F = fixed[cols]
                out[row] = np.linalg.solve(F.T @ F + reg * (e - s) * eye, F.T @ vals)

        self.history = []
        for it in range(iters):
            solve(R, Q, P)
            solve(Rt, P, Q)
            pred = (P[u] * Q[i]).sum(1)
            self.history.append(float(np.sqrt(np.mean((res - pred) ** 2))))
        self.mu, self.bu, self.bi, self.P, self.Q = mu, bu, bi, P, Q
        return self

    def score(self, user_ids):
        pos = self.u_pos.reindex(user_ids).to_numpy()
        out = np.tile(self.mu + self.bi, (len(user_ids), 1))
        ok = ~np.isnan(pos)
        p = pos[ok].astype(int)
        out[ok] += self.bu[p][:, None] + self.P[p] @ self.Q.T
        return out

    def predict_rating(self, user_ids, item_ids):
        up = self.u_pos.reindex(user_ids).to_numpy()
        ip = self.i_pos.reindex(item_ids).to_numpy()
        out = np.full(len(user_ids), self.mu)
        ku, ki = ~np.isnan(up), ~np.isnan(ip)
        out[ku] += self.bu[up[ku].astype(int)]
        out[ki] += self.bi[ip[ki].astype(int)]
        both = ku & ki
        a, b = up[both].astype(int), ip[both].astype(int)
        out[both] += (self.P[a] * self.Q[b]).sum(1)
        return np.clip(out, 1.0, 5.0)


# --------------------------------------------------------------------------- implicit ALS
class ImplicitALS(BaseRecommender):
    """Weighted MF on implicit feedback: every rating is a positive 'watched' signal
    with confidence c_ui = 1 + alpha * (1 or rating)."""
    name = "implicit_als"

    def fit(self, train, fs):
        self._index(train)
        k = self.params.get("factors", 128)
        reg = self.params.get("reg", 100.0)
        alpha = self.params.get("alpha", 1.0)
        iters = self.params.get("iterations", 10)
        rng = np.random.default_rng(self.params.get("seed", 42))
        u = self.u_pos[train["user_id"]].to_numpy()
        i = self.i_pos[train["item_id"]].to_numpy()
        signal = np.ones(len(u)) if self.params.get("binary", True) else train["rating"].to_numpy(np.float64)
        C = sp.csr_matrix((alpha * signal, (u, i)), shape=(len(self.user_ids), len(self.item_ids)))
        Ct = C.T.tocsr()
        X = rng.normal(0, 0.01, (C.shape[0], k))
        Y = rng.normal(0, 0.01, (C.shape[1], k))
        eye = np.eye(k)

        def solve(M, F, out):
            gram = F.T @ F + reg * eye
            for row in range(M.shape[0]):
                s, e = M.indptr[row], M.indptr[row + 1]
                if s == e:
                    continue
                cols, conf = M.indices[s:e], M.data[s:e]
                Fi = F[cols]
                out[row] = np.linalg.solve(gram + (Fi.T * conf) @ Fi, Fi.T @ (1.0 + conf))

        for _ in range(iters):
            solve(C, Y, X)
            solve(Ct, X, Y)
        self.X, self.Y = X, Y
        # cold users (not in train) fall back to the mean user vector
        self.x_mean = X.mean(axis=0)
        return self

    def score(self, user_ids):
        pos = self.u_pos.reindex(user_ids).to_numpy()
        U = np.tile(self.x_mean, (len(user_ids), 1))
        ok = ~np.isnan(pos)
        U[ok] = self.X[pos[ok].astype(int)]
        return U @ self.Y.T

    def predict_rating(self, user_ids, item_ids):
        return None  # implicit model: produces preference scores, not ratings


# --------------------------------------------------------------------------- hybrid
class HybridRecommender(BaseRecommender):
    """z(implicit ALS) * w_cf + z(genre affinity) * w_genre + z(popularity) * w_popularity."""
    name = "hybrid"
    _needs_fs = True

    def fit(self, train, fs, base: BaseRecommender | None = None):
        self._index(train)
        self.fs = fs
        self.base = base if base is not None else ImplicitALS(**self.params.get("cf_params", {})).fit(train, fs)
        pop = fs.get_item_features(self.item_ids)["item_popularity_score"].fillna(0).to_numpy()
        self.pop_z = (pop - pop.mean()) / (pop.std() + 1e-9)
        return self

    def set_weights(self, **w):
        self.params.update(w)
        return self

    @staticmethod
    def _z(x):
        return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)

    def score(self, user_ids):
        w_cf = self.params.get("w_cf", 1.0)
        w_g = self.params.get("w_genre", 0.3)
        w_p = self.params.get("w_popularity", 0.5)
        out = w_cf * self._z(self.base.score(user_ids))
        if w_g:
            out += w_g * self._z(self.fs.genre_affinity(user_ids, self.item_ids))
        if w_p:
            out += w_p * self.pop_z[None, :]
        return out

    def predict_rating(self, user_ids, item_ids):
        return None


MODEL_REGISTRY = {c.__name__: c for c in [PopularityRecommender, MatrixFactorization,
                                          ImplicitALS, HybridRecommender]}
