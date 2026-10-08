"""원본 CSV 사본으로 운영 모델의 감시·재학습 파이프라인을 실행한다."""

import hashlib
import json
import logging

from data.daily_features import parse_timestamp

from serving_app.config import MODEL_NAME

DEFAULT_START = "2024-07-21T00:00:00"
logger = logging.getLogger("solar_aiops")


def inject_curtailment(rows):
    """3일마다 발전량을 10%로 제한한 별도 사본."""
    changed = []
    for source in rows:
        row = dict(source)
        if (
            row["generation_mwh"] is not None
            and parse_timestamp(row["timestamp"]).toordinal() % 3 == 0
        ):
            row["generation_mwh"] *= 0.1
        changed.append(row)
    return changed


def run_simulation(
    rows, incumbent, scenario, start_timestamp=DEFAULT_START, *, dataset_hash=None
):
    from serving_app.monitoring.drift_detector import WINDOW_SIZE
    from serving_app.routers.predict import evaluate_batch, model_identity

    if scenario not in {"normal", "drift"}:
        raise ValueError("시나리오는 normal 또는 drift여야 합니다")
    history = inject_curtailment(rows) if scenario == "drift" else rows
    source_hash = (
        dataset_hash
        or hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    )
    # 원본 관측과 합성 관측의 감시 창을 섞지 않는다.
    context_hash = (
        source_hash
        if scenario == "normal"
        else hashlib.sha256(
            (source_hash + ":daily-curtailment-v2-every-3-days").encode()
        ).hexdigest()
    )
    base_version = model_identity(incumbent)
    metadata = {"simulation": True, "scenario": scenario, "dataset_sha256": source_hash}
    batch = evaluate_batch(
        history,
        incumbent,
        start_timestamp,
        WINDOW_SIZE,
        context_hash,
        strict=True,
        metadata_extra=metadata,
    )
    check = batch.drift_check
    trained = check.get("retraining")
    served = (
        trained.get("served_model_version")
        if trained and trained.get("promoted")
        else base_version
    )
    logger.info(
        "[시뮬레이션] %s, 평가 모델 v%s, RMSE=%.3f, 서빙 모델 v%s",
        scenario,
        base_version,
        check["rmse"],
        served,
    )
    return {
        "simulation": True,
        "scenario": scenario,
        "description": "원본 데이터"
        if scenario == "normal"
        else "3일마다 일별 총발전량을 10%로 제한한 합성 시나리오",
        "start_timestamp": start_timestamp,
        "cutoff_timestamp": batch.records[-1]["timestamp"],
        "base_model_version": base_version,
        "model_name": MODEL_NAME,
        "served_model_version": served,
        "records": batch.records,
        "drift_check": check,
        "retraining": trained,
    }
