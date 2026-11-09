"""
Stage 7: Model Evaluation
Evaluates Popularity vs SVD Recommender across Accuracy, Ranking, Coverage, and Diversity.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import mlflow

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from src.utils import setup_logger, load_config
from src.evaluation.metrics import (
    compute_rmse, compute_mae, precision_at_k, recall_at_k,
    ndcg_at_k, catalog_coverage, intra_list_diversity
)

logger = setup_logger("evaluation")


def evaluate_and_compare_models(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    reports_dir = config["paths"]["reports_dir"]
    threshold = config["evaluation"]["relevance_threshold"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    test_df = pd.read_parquet(os.path.join(processed_dir, "test_interactions.parquet"))
    items_df = pd.read_parquet(os.path.join(processed_dir, "cleaned_items.parquet"))
    total_items = len(items_df["item_id"].unique())

    # Build ground-truth relevant items per test user (rating >= 3.5)
    relevant_items = {}
    for uid, grp in test_df.groupby("user_id"):
        pos = set(grp[grp["rating"] >= threshold]["item_id"].astype(int))
        if pos:
            relevant_items[int(uid)] = pos

    # Genre map for diversity
    genres_map = {
        int(r["item_id"]): {g.strip() for g in str(r["genres"]).split("|") if g.strip()}
        for _, r in items_df.iterrows()
    }

    models_to_test = [
        ("Baseline_Popularity", os.path.join(models_dir, "popularity_model.pkl")),
        ("Personalized_SVD", os.path.join(models_dir, "svd_model.pkl"))
    ]

    results = []

    for name, path in models_to_test:
        with open(path, "rb") as f:
            model = pickle.load(f)

        # 1. Rating prediction metrics
        preds = [model.predict_rating(int(r["user_id"]), int(r["item_id"])) for _, r in test_df.iterrows()]
        rmse = round(compute_rmse(test_df["rating"].values, np.array(preds)), 4)
        mae = round(compute_mae(test_df["rating"].values, np.array(preds)), 4)

        # 2. Top-K Ranking metrics
        p5, r5, n5 = [], [], []
        p10, r10, n10, div10 = [], [], [], []
        all_recs_10 = []

        for uid, pos_set in relevant_items.items():
            recs = model.recommend(uid, top_k=10)
            all_recs_10.append(recs)

            p5.append(precision_at_k(recs, pos_set, k=5))
            r5.append(recall_at_k(recs, pos_set, k=5))
            n5.append(ndcg_at_k(recs, pos_set, k=5))

            p10.append(precision_at_k(recs, pos_set, k=10))
            r10.append(recall_at_k(recs, pos_set, k=10))
            n10.append(ndcg_at_k(recs, pos_set, k=10))
            div10.append(intra_list_diversity(recs, genres_map))

        cov10 = catalog_coverage(all_recs_10, total_items)

        metrics = {
            "model_name": name,
            "rmse": rmse,
            "mae": mae,
            "precision_at_5": round(float(np.mean(p5)), 4),
            "recall_at_5": round(float(np.mean(r5)), 4),
            "ndcg_at_5": round(float(np.mean(n5)), 4),
            "precision_at_10": round(float(np.mean(p10)), 4),
            "recall_at_10": round(float(np.mean(r10)), 4),
            "ndcg_at_10": round(float(np.mean(n10)), 4),
            "coverage_at_10_pct": cov10,
            "diversity_at_10": round(float(np.mean(div10)), 4)
        }
        results.append(metrics)

        # Log to MLflow
        try:
            with mlflow.start_run(run_name=f"eval_{name}"):
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model_name"})
        except Exception:
            pass

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)

    # Save Markdown Table
    md = "# Model Evaluation & Comparison Table\n\n"
    md += "| Model | RMSE (↓) | MAE (↓) | Precision@5 | NDCG@5 | Precision@10 | NDCG@10 | Coverage@10 | Diversity@10 |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for r in results:
        md += f"| **{r['model_name']}** | {r['rmse']} | {r['mae']} | {r['precision_at_5']} | {r['ndcg_at_5']} | {r['precision_at_10']} | {r['ndcg_at_10']} | {r['coverage_at_10_pct']}% | {r['diversity_at_10']} |\n"

    with open(os.path.join(reports_dir, "evaluation_results.md"), "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("Evaluation completed. Results saved in reports/evaluation_results.md.")
    return results


if __name__ == "__main__":
    evaluate_and_compare_models()
