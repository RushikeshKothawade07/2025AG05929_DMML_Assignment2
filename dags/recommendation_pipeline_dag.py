"""
Apache Airflow DAG for Recommendation Pipeline
Coordinates: Ingestion >> Validation >> Preparation >> Transformation >> Features >> Training >> Evaluation
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

try:
    from src.ingestion.ingest import ingest_data
    from src.validation.validate import validate_data
    from src.preparation.prepare import prepare_data
    from src.transformation.transform import transform_data
    from src.features.feature_store import generate_features
    from src.training.train import train_and_track_models
    from src.evaluation.evaluate import evaluate_and_compare_models
except ImportError:
    pass

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="recommendation_pipeline_dag",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    t1 = PythonOperator(task_id="ingest", python_callable=ingest_data)
    t2 = PythonOperator(task_id="validate", python_callable=validate_data)
    t3 = PythonOperator(task_id="prepare", python_callable=prepare_data)
    t4 = PythonOperator(task_id="transform", python_callable=transform_data)
    t5 = PythonOperator(task_id="features", python_callable=generate_features)
    t6 = PythonOperator(task_id="train", python_callable=train_and_track_models)
    t7 = PythonOperator(task_id="evaluate", python_callable=evaluate_and_compare_models)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
