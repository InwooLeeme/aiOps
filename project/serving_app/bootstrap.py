"""빈 실행 환경만 준비한다. 기존 Production과 업로드는 덮어쓰지 않는다."""

import logging
import os
from pathlib import Path

from data import storage
from data.daily_features import load_rows, sample_windows
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from serving_app import model_loader
from serving_app.config import MODEL_NAME, RUNTIME_DIR
from serving_app.tracking import configure_tracking
from serving_app.train_and_register import train_and_register, train_local

logger = logging.getLogger("solar_aiops")


def prepare_service():
    source = os.getenv("MODEL_SOURCE", "local").lower()
    if source == "local":
        if model_loader.LOCAL_BUNDLE_DIR.exists() and any(
            model_loader.LOCAL_BUNDLE_DIR.iterdir()
        ):
            model_loader.read_bundle(model_loader.LOCAL_BUNDLE_DIR)
            return {"status": "existing", "source": "local"}
        # Docker의 읽기 전용 모델 마운트에 쓰지 않는다.
        directory = RUNTIME_DIR / "bootstrap-solar-daily"
        if not directory.exists() or not any(directory.iterdir()):
            train_local(storage.latest_upload(), directory=directory)
        model_loader.read_bundle(directory)
        model_loader.LOCAL_BUNDLE_DIR = directory
        return {"status": "prepared", "source": "local"}
    if source != "mlflow":
        raise ValueError("MODEL_SOURCE는 local 또는 mlflow여야 합니다")
    configure_tracking()
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_NAME, ["Production"])
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        versions = []
    if any(v.current_stage == "Production" for v in versions):
        logger.info("[초기 준비] 기존 Production 모델 재사용")
        return {"status": "existing", "source": "mlflow"}
    # 같은 스냅샷으로 학습 및 후보 로딩 검증을 수행한다.
    path = Path(storage.latest_upload())
    rows = load_rows(path)
    logger.info("[초기 준비] %s로 태양광 모델 학습·검증 시작", path.name)
    result = train_and_register(rows=rows, promote=False)
    if not result.get("gate_passed"):
        raise ValueError(
            "초기 모델 검증 게이트 미통과: Production을 생성하지 않았습니다"
        )
    candidate = model_loader._load_from_mlflow(version=result["version"])
    if not candidate.metadata.get("gate_passed"):
        raise ValueError("초기 후보의 검증 게이트 메타데이터가 유효하지 않습니다")
    window, _ = next(sample_windows(rows))
    candidate.predict_one(window)
    # 단일 프로세스 구성에서도 외부 등록이 있었다면 기존 운영 모델을 보존한다.
    if client.get_latest_versions(MODEL_NAME, ["Production"]):
        raise ValueError(
            "초기 준비 중 Production 모델이 생성되었습니다. 다시 시작하세요"
        )
    client.transition_model_version_stage(MODEL_NAME, result["version"], "Production")
    logger.info("[초기 준비] Production v%s 준비 완료", result["version"])
    return {**result, "status": "prepared", "promoted": True, "source": "mlflow"}
