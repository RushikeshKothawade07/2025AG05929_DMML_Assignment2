"""
Stage 3: Data Preparation
Standardizes item/user formats and performs strict chronological 80/20 train/test split.
"""

import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logger, load_config

logger = setup_logger("preparation")


def prepare_data(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    processed_dir = config["paths"]["processed_dir"]
    train_ratio = config["preparation"]["train_split_ratio"]

    users = pd.read_parquet(os.path.join(processed_dir, "validated_users.parquet"))
    items = pd.read_parquet(os.path.join(processed_dir, "validated_items.parquet"))
    interactions = pd.read_parquet(os.path.join(processed_dir, "validated_interactions.parquet"))

    # Standardize Item metadata (extract release year)
    def get_year(title):
        m = re.search(r"\((\d{4})\)", str(title))
        return int(m.group(1)) if m else 1995

    items["release_year"] = items["title"].apply(get_year)

    # Chronological Time-based Split (prevents data leakage)
    sorted_df = interactions.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
    split_idx = int(len(sorted_df) * train_ratio)

    train_df = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)

    users.to_parquet(os.path.join(processed_dir, "cleaned_users.parquet"), index=False)
    items.to_parquet(os.path.join(processed_dir, "cleaned_items.parquet"), index=False)
    train_df.to_parquet(os.path.join(processed_dir, "train_interactions.parquet"), index=False)
    test_df.to_parquet(os.path.join(processed_dir, "test_interactions.parquet"), index=False)

    logger.info(f"Split interactions: {len(train_df)} train (80%), {len(test_df)} test (20%).")
    return {"train_count": len(train_df), "test_count": len(test_df)}


if __name__ == "__main__":
    prepare_data()
