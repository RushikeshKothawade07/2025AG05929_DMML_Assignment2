"""
Stage 4: Data Transformation
Creates derived datasets:
1. User Activity Summary
2. Item Popularity Summary (IMDb Bayesian rating)
3. User-Item Interaction Matrix (Sparse CSR)
"""

import os
import sys
import json
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logger, load_config

logger = setup_logger("transformation")


def transform_data(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    processed_dir = config["paths"]["processed_dir"]

    users = pd.read_parquet(os.path.join(processed_dir, "cleaned_users.parquet"))
    items = pd.read_parquet(os.path.join(processed_dir, "cleaned_items.parquet"))
    train_df = pd.read_parquet(os.path.join(processed_dir, "train_interactions.parquet"))

    # 1. User Activity Summary
    user_stats = train_df.groupby("user_id").agg(
        interaction_count=("rating", "count"),
        avg_rating=("rating", "mean"),
        rating_std=("rating", "std")
    ).reset_index()
    user_summary = pd.merge(users, user_stats, on="user_id", how="left")
    user_summary["interaction_count"] = user_summary["interaction_count"].fillna(0).astype(int)
    user_summary["avg_rating"] = user_summary["avg_rating"].fillna(train_df["rating"].mean()).round(3)
    user_summary.to_parquet(os.path.join(processed_dir, "user_activity_summary.parquet"), index=False)

    # 2. Item Popularity (IMDb Bayesian formula: WR = (v/(v+m))*R + (m/(v+m))*C)
    global_mean = train_df["rating"].mean()
    m = 10  # damping factor
    item_stats = train_df.groupby("item_id").agg(
        rating_count=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()
    v = item_stats["rating_count"]
    r = item_stats["avg_rating"]
    item_stats["bayesian_rating"] = ((v / (v + m)) * r + (m / (v + m)) * global_mean).round(3)

    item_summary = pd.merge(items, item_stats, on="item_id", how="left")
    item_summary["rating_count"] = item_summary["rating_count"].fillna(0).astype(int)
    item_summary["bayesian_rating"] = item_summary["bayesian_rating"].fillna(global_mean).round(3)
    item_summary.to_parquet(os.path.join(processed_dir, "item_popularity_summary.parquet"), index=False)

    # 3. User-Item Interaction Matrix (Sparse CSR)
    user_to_idx = {uid: idx for idx, uid in enumerate(sorted(users["user_id"].unique()))}
    item_to_idx = {iid: idx for idx, iid in enumerate(sorted(items["item_id"].unique()))}

    rows = train_df["user_id"].map(user_to_idx).dropna().astype(int)
    cols = train_df["item_id"].map(item_to_idx).dropna().astype(int)
    data = train_df.loc[rows.index, "rating"].values

    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_to_idx), len(item_to_idx)), dtype=float)
    save_npz(os.path.join(processed_dir, "user_item_matrix.npz"), matrix)

    sparsity = 100.0 * (1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1])))
    logger.info(f"Derived tables saved. Sparse Matrix shape: {matrix.shape}, Sparsity: {sparsity:.2f}%.")
    return {"matrix_shape": list(matrix.shape), "sparsity": round(sparsity, 2)}


if __name__ == "__main__":
    transform_data()
