"""
모델 로딩 — MODEL_SOURCE=mlflow(기본) 이면 Registry의 Production 모델, local 이면 로컬 .keras 파일.

· 스케일러는 모델 소스와 무관하게 항상 로컬 scaler.pkl (학습 때 fit 한 것을 고정 사용)
· LOADING_MODE=eager 면 서버 시작 시, lazy(기본) 면 첫 요청 시 로드
· 재학습으로 새 Production 이 승격되면 reload() 로 캐시를 갈아끼운다
"""
import os
import time

import numpy as np

from data.features import SCALER_PATH, SolarScaler

LOCAL_MODEL_PATH = "serving_app/models/solar_local.keras"
MODEL_NAME = "Solar_Predictor"
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/Production"

_model_cache = None


class LoadedModel:
    def __init__(self, keras_model, scaler: SolarScaler, version: str):
        self._m = keras_model
        self.scaler = scaler
        self.version = version

    def predict_cf(self, x: np.ndarray) -> float:
        """x: (N_STEPS, N_FEATURES) 원 단위 입력 → 다음날 이용률 cf (0~1)."""
        p = self._m.predict(self.scaler.transform(x[None]), verbose=0)[0][0]
        return float(np.clip(p, 0.0, 1.0))


def _load_from_local() -> LoadedModel:
    from tensorflow import keras

    return LoadedModel(keras.models.load_model(LOCAL_MODEL_PATH), SolarScaler.load(SCALER_PATH), "local")


def _load_from_mlflow() -> LoadedModel:
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient

    keras_model = mlflow.tensorflow.load_model(MLFLOW_MODEL_URI)
    client = MlflowClient()
    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
    return LoadedModel(keras_model, SolarScaler.load(SCALER_PATH), f"v{prod.version}")


def _load_model() -> LoadedModel:
    return _load_from_mlflow() if os.getenv("MODEL_SOURCE", "mlflow") == "mlflow" else _load_from_local()


def load_eager() -> LoadedModel:
    global _model_cache
    start = time.time()
    _model_cache = _load_model()
    print(f"[eager] model {_model_cache.version} loaded in {time.time() - start:.2f}s")
    return _model_cache


def get_model() -> LoadedModel:
    global _model_cache
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model {_model_cache.version} loaded in {time.time() - start:.2f}s on first request")
    return _model_cache


def reload() -> LoadedModel:
    """새 Production 승격 직후 호출: 캐시를 비우고 다시 로드."""
    global _model_cache
    _model_cache = None
    return load_eager()
