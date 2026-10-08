"""모델·스케일러·검증 메타데이터를 동일 학습 버전으로 함께 불러옵니다."""

import hashlib
import json
import math
import os
from pathlib import Path
from threading import RLock

from data.daily_features import FEATURE_COLUMNS, SEQ_LEN, SolarScaler

from serving_app.config import MODEL_DIR, MODEL_NAME

LOCAL_BUNDLE_DIR = MODEL_DIR / "solar-daily"
_model_cache = None
_cache_lock = RLock()


def artifact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bundle(directory: Path):
    paths = {
        name: directory / name
        for name in ("model.keras", "scaler.json", "metadata.json")
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(
            "제주 태양광 모델 번들이 없습니다. "
            "python project/scripts/train_baseline_v1.py --csv <제주 CSV>를 실행하세요"
        )
    metadata = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
    if (
        metadata.get("feature_columns") != FEATURE_COLUMNS
        or metadata.get("seq_len") != SEQ_LEN
    ):
        raise ValueError("모델의 feature_columns/seq_len이 현재 태양광 입력과 다릅니다")
    if (
        metadata.get("granularity") != "daily"
        or metadata.get("target") != "next_day_generation_mwh"
    ):
        raise ValueError("일별 총발전량 모델 번들이 필요합니다")
    if metadata.get("unit") != "MWh":
        raise ValueError("모델 발전량 단위가 MWh가 아닙니다")
    for name in ("model.keras", "scaler.json"):
        if metadata.get("artifact_sha256", {}).get(name) != artifact_hash(paths[name]):
            raise ValueError(f"모델 artifact 버전이 일치하지 않습니다: {name}")
    return SolarScaler.load(paths["scaler.json"]), metadata


class LoadedModel:
    def __init__(
        self,
        keras_model,
        scaler: SolarScaler,
        version: str,
        registry_version: str | None = None,
        metadata: dict | None = None,
    ):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version
        self.registry_version = registry_version
        self.metadata = metadata or {}

    def predict_one(self, sequence: list[dict]) -> float:
        import numpy as np

        if len(sequence) != SEQ_LEN:
            raise ValueError(f"정확히 {SEQ_LEN}일의 입력이 필요합니다")
        x = np.asarray(
            [[self.scaler.transform_point(row) for row in sequence]], dtype="float32"
        )
        predicted = float(
            np.asarray(self._keras_model(x, training=False)).reshape(-1)[0]
        )
        value = float(self.scaler.inverse_target(predicted))
        if not math.isfinite(value):
            raise ValueError("모델이 유한한 발전량을 반환하지 않았습니다")
        return max(0.0, value)


def _load_from_local() -> LoadedModel:
    from tensorflow import keras

    scaler, metadata = read_bundle(LOCAL_BUNDLE_DIR)
    model = keras.models.load_model(LOCAL_BUNDLE_DIR / "model.keras")
    return LoadedModel(
        model,
        scaler,
        f"solar-local-{metadata['artifact_sha256']['model.keras'][:12]}",
        metadata=metadata,
    )


def _load_from_mlflow(model_name=MODEL_NAME, *, version=None) -> LoadedModel:
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient

    from serving_app.tracking import configure_tracking

    configure_tracking()
    client = MlflowClient()
    if version is None:
        candidates = client.get_latest_versions(model_name, ["Production"])
        production = max(
            (v for v in candidates if v.current_stage == "Production"),
            key=lambda v: int(v.version),
            default=None,
        )
        if production is None:
            raise FileNotFoundError(f"{model_name}의 Production 모델이 없습니다")
        version = str(production.version)
    version = str(version)
    pinned = client.get_model_version(model_name, version)
    directory = Path(client.download_artifacts(pinned.run_id, "bundle"))
    scaler, metadata = read_bundle(directory)
    model = mlflow.tensorflow.load_model(f"models:/{model_name}/{version}")
    return LoadedModel(
        model, scaler, "production", registry_version=version, metadata=metadata
    )


def promote_candidate(version, incumbent, sequence) -> LoadedModel:
    """후보 번들 로딩·실제 추론을 완료한 뒤 Production과 캐시를 교체한다."""
    from mlflow.tracking import MlflowClient

    global _model_cache
    replacement = _load_from_mlflow(version=str(version))
    if not replacement.metadata.get("gate_passed"):
        raise ValueError("검증 게이트를 통과한 후보만 승격할 수 있습니다")
    if str(replacement.metadata.get("parent_version")) != str(
        incumbent.registry_version
    ):
        raise ValueError("후보의 기준 모델이 현재 운영 모델과 다릅니다")
    value = replacement.predict_one(sequence)
    if not math.isfinite(value):
        raise ValueError("후보 모델이 유한한 발전량을 반환하지 않았습니다")
    with _cache_lock:
        if _model_cache is not incumbent:
            raise ValueError("재학습 중 서빙 모델이 변경되었습니다")
        client = MlflowClient()
        production = client.get_latest_versions(MODEL_NAME, ["Production"])
        versions = {
            str(v.version) for v in production if v.current_stage == "Production"
        }
        if versions != {str(incumbent.registry_version)}:
            raise ValueError("재학습 중 Production 모델이 변경되었습니다")
        client.transition_model_version_stage(
            MODEL_NAME, str(version), "Production", archive_existing_versions=True
        )
        _model_cache = replacement
    return replacement


def _load_model() -> LoadedModel:
    source = os.getenv("MODEL_SOURCE", "local").lower()
    if source == "mlflow":
        return _load_from_mlflow()
    if source != "local":
        raise ValueError("MODEL_SOURCE는 local 또는 mlflow여야 합니다")
    return _load_from_local()


def reload_model() -> LoadedModel:
    global _model_cache
    with _cache_lock:
        replacement = _load_model()
        _model_cache = replacement
        return replacement


def load_eager() -> LoadedModel:
    return reload_model()


def get_model() -> LoadedModel:
    with _cache_lock:
        if _model_cache is None:
            return reload_model()
        return _model_cache
