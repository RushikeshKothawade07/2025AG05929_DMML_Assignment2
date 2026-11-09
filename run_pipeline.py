"""
Master Pipeline Orchestration Runner
Executes: Ingestion -> Validation -> Preparation -> Transformation -> Features -> Training -> Evaluation
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from src.ingestion.ingest import ingest_data
from src.validation.validate import validate_data
from src.preparation.prepare import prepare_data
from src.transformation.transform import transform_data
from src.features.feature_store import generate_features
from src.training.train import train_and_track_models
from src.evaluation.evaluate import evaluate_and_compare_models
from src.utils import setup_logger

logger = setup_logger("master_pipeline")


def run(stage="all", config_path="configs/config.yaml"):
    stages = [
        ("ingest", "1. Ingest Data", ingest_data),
        ("validate", "2. Validate Quality", validate_data),
        ("prepare", "3. Prepare & Split", prepare_data),
        ("transform", "4. Transform Data", transform_data),
        ("features", "5. Feature Store", generate_features),
        ("train", "6. Train Models", train_and_track_models),
        ("evaluate", "7. Evaluate Models", evaluate_and_compare_models),
    ]

    t_start = time.time()
    logger.info(f"=== Starting Recommendation Pipeline (Stage: {stage.upper()}) ===")

    for key, label, fn in stages:
        if stage in ["all", key]:
            t0 = time.time()
            logger.info(f"--- Running: {label} ---")
            fn(config_path)
            logger.info(f"Completed {label} in {round(time.time() - t0, 2)}s.")

    total_time = round(time.time() - t_start, 2)
    logger.info(f"=== Pipeline Completed Successfully in {total_time}s! ===")

    # Write simple dashboard
    with open("reports/dag_run_status.md", "w", encoding="utf-8") as f:
        f.write("# Pipeline Execution Status\n\n")
        f.write(f"- **Status:** 🟢 SUCCESS\n")
        f.write(f"- **Total Time:** {total_time}s\n")
        f.write(f"- **Stages Executed:** Ingestion ➔ Validation ➔ Preparation ➔ Transformation ➔ Features ➔ Training ➔ Evaluation\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Recommendation Pipeline")
    parser.add_argument("--stage", default="all", choices=["all", "ingest", "validate", "prepare", "transform", "features", "train", "evaluate"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(stage=args.stage, config_path=args.config)
