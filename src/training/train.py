"""
Stage 6: Model Training
Trains Popularity Baseline & Matrix Factorization SVD, tracking runs in MLflow.
"""

import os
import sys
import pickle
import pandas as pd
import mlflow

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from src.utils import setup_logger, load_config
from src.training.models import PopularityRecommender, MatrixFactorizationSVD

logger = setup_logger("training")


def train_and_track_models(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    train_df = pd.read_parquet(os.path.join(processed_dir, "train_interactions.parquet"))
    items_df = pd.read_parquet(os.path.join(processed_dir, "cleaned_items.parquet"))

    # 1. Baseline Popularity Model
    logger.info("Training Model 1: Baseline Popularity...")
    pop_model = PopularityRecommender().fit(train_df, items_df)
    pop_path = os.path.join(models_dir, "popularity_model.pkl")
    with open(pop_path, "wb") as f:
        pickle.dump(pop_model, f)

    # 2. Personalized SVD Model
    logger.info("Training Model 2: Personalized Matrix Factorization SVD...")
    svd_cfg = config["models"]["matrix_factorization_svd"]
    svd_model = MatrixFactorizationSVD(
        n_factors=svd_cfg["n_factors"],
        lr=svd_cfg["learning_rate"],
        reg=svd_cfg["reg"],
        n_epochs=svd_cfg["n_epochs"],
        random_state=svd_cfg["random_state"]
    ).fit(train_df, items_df)
    svd_path = os.path.join(models_dir, "svd_model.pkl")
    with open(svd_path, "wb") as f:
        pickle.dump(svd_model, f)

    # MLflow Tracking
    try:
        with mlflow.start_run(run_name="train_Baseline_Popularity"):
            mlflow.log_param("model_type", "PopularityRecommender")
            mlflow.log_artifact(pop_path)

        with mlflow.start_run(run_name="train_Personalized_SVD"):
            mlflow.log_params(svd_cfg)
            mlflow.log_artifact(svd_path)
        logger.info("Models and parameters logged to MLflow.")
    except Exception as e:
        logger.warning(f"MLflow note: {e}")

    return {"popularity_model": pop_path, "svd_model": svd_path}


if __name__ == "__main__":
    train_and_track_models()
