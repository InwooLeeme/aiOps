"""관측 배치 성능 저하 → 재학습 → 검증된 후보의 운영 모델 교체."""

import logging
import os
from datetime import timedelta

from data.features import parse_timestamp, sample_windows

from serving_app.monitoring.drift_detector import assess_drift

logger = logging.getLogger("solar_aiops")


def check_and_trigger(
    records: list[dict],
    threshold: float | None = None,
    *,
    rows=None,
    incumbent=None,
    metadata_extra=None,
) -> dict:
    result = assess_drift(records, threshold)
    result["retraining"] = None
    if result["status"] != "performance_degraded":
        return result
    logger.warning(
        "[WARN] 태양광 예측 성능 저하: RMSE=%.3f MWh, 기준=%.3f MWh",
        result["rmse"],
        threshold,
    )
    if os.getenv("MODEL_SOURCE", "local").lower() != "mlflow":
        result["retraining"] = {
            "status": "blocked",
            "reason": "자동 재학습은 MODEL_SOURCE=mlflow에서 실행하세요",
        }
    elif rows is None or incumbent is None:
        result["retraining"] = {
            "status": "blocked",
            "reason": "평가에 사용한 데이터와 운영 모델이 필요합니다",
        }
    else:
        cutoff = max(r["timestamp"] for r in records)
        result["retraining"] = retrain_at(
            rows, cutoff, incumbent=incumbent, metadata_extra=metadata_extra
        )
    return result


def retrain_at(
    rows: list[dict], cutoff: str, *, incumbent=None, metadata_extra=None
) -> dict:
    from serving_app import model_loader
    from serving_app.train_and_register import fine_tune

    if os.getenv("MODEL_SOURCE", "local").lower() != "mlflow":
        return {
            "status": "blocked",
            "reason": "재학습은 MODEL_SOURCE=mlflow에서 실행하세요",
        }
    stamp = parse_timestamp(cutoff)
    start = stamp - timedelta(days=90) + timedelta(hours=1)
    history = [r for r in rows if start <= parse_timestamp(r["timestamp"]) <= stamp]
    incumbent = incumbent if incumbent is not None else model_loader.get_model()
    logger.info("[INFO] 자동 재학습 시작: 관측 종료=%s, 최근 90일", cutoff)
    try:
        result = fine_tune(
            history, incumbent=incumbent, promote=False, metadata_extra=metadata_extra
        )
    except ValueError as exc:
        logger.warning("재학습 차단: %s", exc)
        return {"status": "blocked", "reason": str(exc), "promoted": False}
    except Exception as exc:
        logger.exception("재학습 실패: 기존 운영 모델 유지")
        return {"status": "failed", "reason": str(exc), "promoted": False}
    result["cutoff_timestamp"] = cutoff
    if not result.get("gate_passed"):
        logger.info("[INFO] 검증 게이트 미통과: 기존 운영 모델 유지")
        return {**result, "status": "gate_rejected"}
    try:
        # 결측된 target 기상값 대신 마지막 유효 평가 입력으로 로딩·추론 검증.
        window, _ = list(sample_windows(history))[-1]
        loaded = model_loader.promote_candidate(result["version"], incumbent, window)
    except Exception as exc:
        logger.exception("후보 활성화 실패: 기존 서빙 모델 유지")
        return {
            **result,
            "status": "activation_failed",
            "reason": str(exc),
            "promoted": False,
        }
    logger.info(
        "[OK] 검증 통과 → Production 승격 및 서빙 교체 완료: v%s",
        loaded.registry_version,
    )
    return {
        **result,
        "status": "promoted",
        "promoted": True,
        "served_model_version": loaded.registry_version,
    }
