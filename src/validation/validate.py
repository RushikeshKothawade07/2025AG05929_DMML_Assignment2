"""
Stage 2: Data Quality Validation
Performs automated checks:
- Missing values & duplicates
- Value bounds (ratings in [1.0, 5.0], realistic timestamps)
- Referential integrity (foreign key checks on user_id and item_id)
Separates invalid records into data/errors/ and writes a validation report.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logger, load_config

logger = setup_logger("validation")


def validate_data(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    raw_dir = config["paths"]["raw_dir"]
    processed_dir = config["paths"]["processed_dir"]
    errors_dir = config["paths"]["errors_dir"]
    reports_dir = config["paths"]["reports_dir"]

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    users = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    items = pd.read_csv(os.path.join(raw_dir, "items.csv"))
    interactions = pd.read_csv(os.path.join(raw_dir, "interactions.csv"))

    logger.info(f"Validating {len(interactions)} interaction records...")

    valid_user_ids = set(users["user_id"].dropna().astype(int))
    valid_item_ids = set(items["item_id"].dropna().astype(int))

    errors = []
    for idx, row in interactions.iterrows():
        reasons = []
        # 1. Missing values
        if pd.isna(row["user_id"]) or pd.isna(row["item_id"]) or pd.isna(row["rating"]) or pd.isna(row["timestamp"]):
            reasons.append("MISSING_VALUE")
        # 2. Rating bounds [1.0, 5.0]
        if pd.notna(row["rating"]) and (row["rating"] < 1.0 or row["rating"] > 5.0):
            reasons.append("INVALID_RATING_BOUNDS")
        # 3. Timestamp bounds
        if pd.notna(row["timestamp"]) and (row["timestamp"] <= 0 or row["timestamp"] > 2000000000):
            reasons.append("INVALID_TIMESTAMP")
        # 4. Referential integrity (FK)
        if pd.notna(row["user_id"]) and int(row["user_id"]) not in valid_user_ids:
            reasons.append("UNKNOWN_USER_FK")
        if pd.notna(row["item_id"]) and int(row["item_id"]) not in valid_item_ids:
            reasons.append("UNKNOWN_ITEM_FK")

        if reasons:
            errors.append((idx, "; ".join(reasons)))

    # Deduplication check
    dup_mask = interactions.duplicated(subset=["user_id", "item_id", "timestamp"], keep="first")
    for idx in np.where(dup_mask)[0]:
        errors.append((idx, "DUPLICATE_RECORD"))

    # Separate errors into error dataset
    error_indices = set(idx for idx, _ in errors)
    error_dict = {idx: reason for idx, reason in errors}

    invalid_df = interactions.loc[list(error_indices)].copy()
    invalid_df["error_reason"] = invalid_df.index.map(error_dict)
    clean_df = interactions.drop(index=list(error_indices)).copy()

    # Save outputs
    invalid_path = os.path.join(errors_dir, "invalid_interactions.csv")
    invalid_df.to_csv(invalid_path, index=False)

    users.to_parquet(os.path.join(processed_dir, "validated_users.parquet"), index=False)
    items.to_parquet(os.path.join(processed_dir, "validated_items.parquet"), index=False)
    clean_df.to_parquet(os.path.join(processed_dir, "validated_interactions.parquet"), index=False)

    # Generate Markdown Report
    report_path = os.path.join(reports_dir, "data_validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Data Quality & Validation Report\n\n")
        f.write(f"- **Total Records Evaluated:** {len(interactions):,}\n")
        f.write(f"- **Clean Valid Records:** {len(clean_df):,}\n")
        f.write(f"- **Quarantined Invalid Records:** {len(invalid_df):,}\n")
        f.write(f"- **Quarantine Location:** `{invalid_path}`\n\n")
        f.write("## Validation Rules Summary\n\n")
        f.write("| Rule Checked | Criteria | Status |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write("| Nulls / Missing Values | Critical fields not null | ENFORCED |\n")
        f.write("| Rating Domain | 1.0 <= rating <= 5.0 | ENFORCED |\n")
        f.write("| Timestamp Range | Epoch > 0 & < 2030 | ENFORCED |\n")
        f.write("| Duplicate Rows | Unique (user, item, timestamp) | ENFORCED |\n")
        f.write("| Referential Integrity | User & Item must exist in DB | ENFORCED |\n")

    logger.info(f"Validation finished: {len(clean_df)} valid, {len(invalid_df)} quarantined in {invalid_path}.")
    return {"valid": len(clean_df), "quarantined": len(invalid_df)}


if __name__ == "__main__":
    validate_data()
