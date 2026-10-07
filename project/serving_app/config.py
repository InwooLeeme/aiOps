"""제주 시간별 태양광 발전량 예측의 공용 설정."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "serving_app" / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
DEFAULT_TRACKING_URI = f"sqlite:///{RUNTIME_DIR / 'mlflow.db'}"
ARTIFACT_DIR = RUNTIME_DIR / "mlartifacts"
SEED = 42
MODEL_NAME = "JejuSolarPredictor"
BASE_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4
