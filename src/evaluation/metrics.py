"""Ranking and rating metrics.

All ranking metrics use binary relevance: a test item is relevant if its rating
is >= the relevance threshold (default 4).
"""
from __future__ import annotations

import numpy as np


def precision_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: list, relevant: set, k: int) -> float:
    gains = [1.0 / np.log2(i + 2) for i, it in enumerate(recs[:k]) if it in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return sum(gains) / ideal if ideal else 0.0


def hit_rate_at_k(recs: list, relevant: set, k: int) -> float:
    return float(bool(set(recs[:k]) & relevant))


def catalog_coverage(all_recs: np.ndarray, n_catalog: int) -> float:
    """Share of the catalog that appears in at least one user's top-k list."""
    return len(np.unique(all_recs)) / n_catalog


def intra_list_diversity(all_recs: np.ndarray, item_vectors: dict) -> float:
    """Mean pairwise (1 - cosine similarity) of genre vectors inside each list."""
    scores = []
    for recs in all_recs:
        V = np.array([item_vectors[i] for i in recs], dtype=np.float64)
        V /= np.linalg.norm(V, axis=1, keepdims=True).clip(min=1e-9)
        sim = V @ V.T
        k = len(recs)
        scores.append((k * k - sim.sum()) / (k * (k - 1)))  # off-diagonal mean of 1-sim
    return float(np.mean(scores))


def novelty(all_recs: np.ndarray, item_pop_share: dict) -> float:
    """Mean self-information -log2(p(item)) of recommended items (higher = less mainstream)."""
    vals = [-np.log2(max(item_pop_share.get(i, 0.0), 1e-9)) for i in all_recs.ravel()]
    return float(np.mean(vals))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))
