"""
Unit tests for Feature Store generation.
"""

import pandas as pd
from src.features.feature_store import generate_features


def test_feature_generation():
    # Verifies feature files can be generated from existing processed parquet data
    res = generate_features("configs/config.yaml")
    assert "user_features" in res
    assert "item_features" in res
    assert res["user_features"] > 0
