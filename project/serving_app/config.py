"""실행 위치와 관계없이 사용하는 HAIC 프로젝트 경로."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "serving_app" / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
DEFAULT_TRACKING_URI = f"sqlite:///{RUNTIME_DIR / 'mlflow.db'}"
ARTIFACT_DIR = RUNTIME_DIR / "mlartifacts"
