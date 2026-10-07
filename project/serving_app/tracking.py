"""학습과 서빙이 동일한 MLflow 저장소를 사용하도록 설정합니다."""

import os

import mlflow

from serving_app.config import ARTIFACT_DIR, DEFAULT_TRACKING_URI, RUNTIME_DIR


def configure_tracking() -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    if uri == DEFAULT_TRACKING_URI:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)


def configure_experiment() -> None:
    configure_tracking()
    name = "JejuSolar"
    if mlflow.get_experiment_by_name(name) is None:
        if mlflow.get_tracking_uri() == DEFAULT_TRACKING_URI:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(name, artifact_location=ARTIFACT_DIR.as_uri())
        else:
            mlflow.create_experiment(name)
    mlflow.set_experiment(name)
