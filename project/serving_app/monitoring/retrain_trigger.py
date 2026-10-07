"""성능 감시와 명시적 재학습. 데이터 재생 자체는 모델을 변경하지 않는다."""

import logging
from datetime import timedelta

from data.features import parse_timestamp

from serving_app.monitoring.drift_detector import assess_drift

logger = logging.getLogger("solar_aiops")


def check_and_trigger(records: list[dict], threshold: float | None = None) -> dict:
    result = assess_drift(records, threshold)
    if result["status"] == "performance_degraded":
        logger.warning(
            "태양광 예측 성능 저하: RMSE=%.3f MWh, 기준=%.3f MWh. "
            "데이터 품질 확인 후 재학습 검토",
            result["rmse"],
            threshold,
        )
    return result


def retrain_at(rows: list[dict], cutoff: str) -> dict:
    from serving_app import model_loader
    from serving_app.train_and_register import fine_tune

    stamp = parse_timestamp(cutoff)
    start = stamp - timedelta(days=90)
    history = [r for r in rows if start <= parse_timestamp(r["timestamp"]) <= stamp]
    result = fine_tune(history)
    if result.get("promoted"):
        model_loader.reload_model()
        logger.info(
            "태양광 모델 재학습 검증 통과 및 새 버전 로딩 완료: %s",
            result.get("version"),
        )
    return result
