"""
Unit tests for Recommender Models and Metrics.
"""

import numpy as np
import pandas as pd
from src.training.models import PopularityRecommender, MatrixFactorizationSVD
from src.evaluation.metrics import compute_rmse, compute_mae, precision_at_k, ndcg_at_k


def test_popularity_recommender():
    train_df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "item_id": [10, 20, 20],
        "rating": [5.0, 4.0, 3.0]
    })
    items_df = pd.DataFrame({"item_id": [10, 20, 30]})

    model = PopularityRecommender().fit(train_df, items_df)
    recs = model.recommend(user_id=1, top_k=2)
    assert 10 not in recs  # already seen
    assert 20 not in recs  # already seen


def test_matrix_factorization_svd():
    train_df = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 20, 30],
        "rating": [5.0, 4.0, 3.0, 5.0]
    })
    items_df = pd.DataFrame({"item_id": [10, 20, 30]})

    model = MatrixFactorizationSVD(n_factors=4, n_epochs=5).fit(train_df, items_df)
    pred = model.predict_rating(user_id=1, item_id=30)
    assert 1.0 <= pred <= 5.0


def test_metrics():
    actuals = np.array([4.0, 5.0])
    preds = np.array([3.5, 4.5])
    assert compute_rmse(actuals, preds) > 0
    assert compute_mae(actuals, preds) > 0

    assert precision_at_k([1, 2, 3], {2, 3}, k=3) == 2 / 3
    assert 0.0 <= ndcg_at_k([1, 2, 3], {2, 3}, k=3) <= 1.0
