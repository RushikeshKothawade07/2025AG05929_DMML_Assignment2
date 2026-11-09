"""
Stage 7: Evaluation Metrics
Precision@K, Recall@K, NDCG@K, Coverage, Diversity, RMSE, and MAE.
"""

import numpy as np


def compute_rmse(actuals, preds):
    return float(np.sqrt(np.mean((actuals - preds) ** 2)))


def compute_mae(actuals, preds):
    return float(np.mean(np.abs(actuals - preds)))


def precision_at_k(recommended, relevant, k=10):
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / float(k)


def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / float(len(relevant))


def ndcg_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(idx + 2) for idx, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(k, len(relevant))))
    return (dcg / idcg) if idcg > 0 else 0.0


def catalog_coverage(all_recs, total_items):
    unique = set(item for recs in all_recs for item in recs)
    return round((len(unique) / float(total_items)) * 100.0, 2)


def intra_list_diversity(recommended, genres_map):
    k = len(recommended)
    if k <= 1:
        return 0.0
    dissims = []
    for i in range(k):
        for j in range(i + 1, k):
            g1 = genres_map.get(recommended[i], set())
            g2 = genres_map.get(recommended[j], set())
            union = len(g1 | g2)
            sim = (len(g1 & g2) / float(union)) if union > 0 else 1.0
            dissims.append(1.0 - sim)
    return float(np.mean(dissims)) if dissims else 0.0
