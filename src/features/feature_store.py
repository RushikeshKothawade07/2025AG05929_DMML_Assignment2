"""
Stage 5: Feature Management
Computes reusable user, item, and interaction features and exports the Feature Catalog.
"""

import os
import sys
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logger, load_config

logger = setup_logger("feature_store")


def generate_features(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    processed_dir = config["paths"]["processed_dir"]
    features_dir = config["paths"]["features_dir"]
    reports_dir = config["paths"]["reports_dir"]

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    users = pd.read_parquet(os.path.join(processed_dir, "cleaned_users.parquet"))
    items = pd.read_parquet(os.path.join(processed_dir, "cleaned_items.parquet"))
    train_df = pd.read_parquet(os.path.join(processed_dir, "train_interactions.parquet"))

    global_mean = float(train_df["rating"].mean())
    max_ts = int(train_df["timestamp"].max())

    # 1. User Features
    u_stats = train_df.groupby("user_id").agg(
        user_rating_count=("rating", "count"),
        user_avg_rating=("rating", "mean"),
        user_last_ts=("timestamp", "max")
    ).reset_index()

    # Preferred genre per user
    merged = pd.merge(train_df, items[["item_id", "genres"]], on="item_id", how="inner")
    pref_genres = {}
    for uid, grp in merged.groupby("user_id"):
        genres = [g.strip() for g_str in grp["genres"] for g in str(g_str).split("|") if g.strip()]
        pref_genres[uid] = Counter(genres).most_common(1)[0][0] if genres else "Drama"

    u_stats["user_preferred_genre"] = u_stats["user_id"].map(pref_genres).fillna("Drama")
    u_stats["days_since_last_interaction"] = ((max_ts - u_stats["user_last_ts"]) / 86400.0).round(1)

    user_features = pd.merge(users, u_stats, on="user_id", how="left")
    user_features["user_rating_count"] = user_features["user_rating_count"].fillna(0).astype(int)
    user_features["user_avg_rating"] = user_features["user_avg_rating"].fillna(global_mean).round(3)
    user_features.drop(columns=["user_last_ts"], inplace=True)
    user_features.to_parquet(os.path.join(features_dir, "user_features.parquet"), index=False)

    # 2. Item Features
    i_stats = train_df.groupby("item_id").agg(
        item_rating_count=("rating", "count"),
        item_avg_rating=("rating", "mean")
    ).reset_index()
    i_stats["item_popularity_score"] = (i_stats["item_rating_count"] / i_stats["item_rating_count"].max()).round(3)

    item_features = pd.merge(items, i_stats, on="item_id", how="left")
    item_features["item_rating_count"] = item_features["item_rating_count"].fillna(0).astype(int)
    item_features["item_avg_rating"] = item_features["item_avg_rating"].fillna(global_mean).round(3)
    item_features["item_popularity_score"] = item_features["item_popularity_score"].fillna(0.0)
    item_features.to_parquet(os.path.join(features_dir, "item_features.parquet"), index=False)

    # 3. Interaction Features
    inter = train_df.copy()
    inter["interaction_hour"] = pd.to_datetime(inter["timestamp"], unit="s").dt.hour
    inter = pd.merge(inter, user_features[["user_id", "user_preferred_genre"]], on="user_id", how="left")
    inter = pd.merge(inter, items[["item_id", "genres"]], on="item_id", how="left")
    inter["user_item_genre_match"] = inter.apply(
        lambda r: 1 if str(r["user_preferred_genre"]) in str(r["genres"]) else 0, axis=1
    )
    inter.drop(columns=["user_preferred_genre", "genres"], inplace=True)
    inter.to_parquet(os.path.join(features_dir, "interaction_features.parquet"), index=False)

    # 4. Feature Catalog Markdown Table
    catalog_md = """# Reusable Feature Catalog

| Feature Name | Entity | Data Type | Formula / Logic | Description |
| :--- | :--- | :--- | :--- | :--- |
| `user_rating_count` | User | INT | COUNT(ratings in train) | Total interactions by user |
| `user_avg_rating` | User | FLOAT | MEAN(ratings in train) | User's rating leniency baseline |
| `user_preferred_genre` | User | STRING | MODE(genres of rated movies) | User's top preferred genre |
| `days_since_last_interaction` | User | FLOAT | (max_snapshot_ts - last_ts) / 86400 | Recency & churn indicator |
| `item_rating_count` | Item | INT | COUNT(ratings in train) | Movie popularity volume |
| `item_avg_rating` | Item | FLOAT | MEAN(ratings in train) | Perceived quality of movie |
| `item_popularity_score` | Item | FLOAT | count / max(count) | Normalized popularity index [0, 1] |
| `interaction_hour` | Interaction | INT | EXTRACT(HOUR from timestamp) | Time-of-day contextual consumption |
| `user_item_genre_match` | Interaction | INT | 1 if preferred_genre in item_genres else 0 | Thematic alignment match |
"""
    with open(os.path.join(reports_dir, "feature_catalog.md"), "w", encoding="utf-8") as f:
        f.write(catalog_md)

    logger.info(f"Feature Store built. Catalog exported to {reports_dir}/feature_catalog.md.")
    return {"user_features": len(user_features), "item_features": len(item_features)}


if __name__ == "__main__":
    generate_features()
