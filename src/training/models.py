"""
Stage 6: Recommender Models
1. PopularityRecommender (Baseline non-personalized)
2. MatrixFactorizationSVD (Personalized regularized Funk-SVD with biases)
"""

import numpy as np
import pandas as pd


class PopularityRecommender:
    """Baseline non-personalized recommender ranking items by rating volume and mean score."""
    def __init__(self):
        self.item_scores = {}
        self.ranked_items = []
        self.user_history = {}
        self.global_mean = 3.5

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        for uid, grp in train_df.groupby("user_id"):
            self.user_history[int(uid)] = set(grp["item_id"].astype(int))

        stats = train_df.groupby("item_id").agg(count=("rating", "count"), mean=("rating", "mean"))
        # Score = count * mean
        scores = {int(iid): row["count"] * row["mean"] for iid, row in stats.iterrows()}
        self.item_scores = scores
        self.ranked_items = [iid for iid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        return float(self.global_mean)

    def recommend(self, user_id: int, top_k: int = 10) -> list:
        history = self.user_history.get(user_id, set())
        return [iid for iid in self.ranked_items if iid not in history][:top_k]


class MatrixFactorizationSVD:
    """Personalized Latent Factor Collaborative Filtering (Funk-SVD with user & item biases)."""
    def __init__(self, n_factors=32, lr=0.005, reg=0.02, n_epochs=15, random_state=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.global_mean = 3.5
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_history = {}
        self.bu = None
        self.bi = None
        self.P = None
        self.Q = None

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame):
        rng = np.random.default_rng(self.random_state)
        self.global_mean = float(train_df["rating"].mean())

        for uid, grp in train_df.groupby("user_id"):
            self.user_history[int(uid)] = set(grp["item_id"].astype(int))

        users = sorted(train_df["user_id"].unique())
        items = sorted(items_df["item_id"].unique())
        self.user_to_idx = {uid: idx for idx, uid in enumerate(users)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(items)}
        self.idx_to_item = {idx: iid for idx, iid in enumerate(items)}

        n_u = len(users)
        n_i = len(items)

        self.bu = np.zeros(n_u, dtype=np.float32)
        self.bi = np.zeros(n_i, dtype=np.float32)
        self.P = rng.normal(0, 0.1, (n_u, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_i, self.n_factors)).astype(np.float32)

        u_arr = np.array([self.user_to_idx[uid] for uid in train_df["user_id"]], dtype=int)
        i_arr = np.array([self.item_to_idx[iid] for iid in train_df["item_id"]], dtype=int)
        r_arr = train_df["rating"].values.astype(np.float32)

        for _ in range(self.n_epochs):
            for u, i, r in zip(u_arr, i_arr, r_arr):
                pred = self.global_mean + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))
                err = r - pred
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])
        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if user_id in self.user_to_idx and item_id in self.item_to_idx:
            u = self.user_to_idx[user_id]
            i = self.item_to_idx[item_id]
            pred = self.global_mean + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))
        else:
            pred = self.global_mean
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, top_k: int = 10) -> list:
        if user_id not in self.user_to_idx:
            return list(self.idx_to_item.values())[:top_k]
        u = self.user_to_idx[user_id]
        history = self.user_history.get(user_id, set())

        scores = self.global_mean + self.bu[u] + self.bi + np.dot(self.Q, self.P[u])
        candidates = [(self.idx_to_item[i], score) for i, score in enumerate(scores) if self.idx_to_item[i] not in history]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in candidates[:top_k]]
