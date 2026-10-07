"""원본 관측과 운영 모델을 보존하는 태양광 드리프트 시연."""

import logging
from datetime import timedelta

from data.features import parse_timestamp, sample_windows

from serving_app.model_loader import _load_from_mlflow
from serving_app.monitoring.drift_detector import assess_drift
from serving_app.train_and_register import fine_tune

SIMULATION_MODEL_NAME = "JejuSolarSimulator"
DEFAULT_START = "2024-05-23T17:00:00"
logger = logging.getLogger("solar_aiops")


def inject_curtailment(rows):
    """08~18시의 짝수 시간에 발전량을 10%로 제한한 별도 사본."""
    changed = []
    for source in rows:
        row = dict(source)
        hour = parse_timestamp(row["timestamp"]).hour
        if row["generation_mwh"] is not None and 8 <= hour <= 18 and hour % 2 == 0:
            row["generation_mwh"] *= 0.1
        changed.append(row)
    return changed


def run_simulation(rows, incumbent, scenario, start_timestamp=DEFAULT_START):
    start = parse_timestamp(start_timestamp)
    cutoff = start + timedelta(hours=167)
    selection_end = max(
        incumbent.metadata.get("training_end", ""),
        incumbent.metadata.get("validation_end", ""),
    )
    if not selection_end or start <= parse_timestamp(selection_end):
        raise ValueError("시뮬레이션 구간은 운영 모델 학습·검증 종료 이후여야 합니다")
    if scenario not in {"normal", "drift"}:
        raise ValueError("시나리오는 normal 또는 drift여야 합니다")
    # 변형한 과거 입력과 정답을 평가·재학습에 동일하게 사용한다.
    history = [
        dict(row)
        for row in rows
        if cutoff - timedelta(days=90) <= parse_timestamp(row["timestamp"]) <= cutoff
    ]
    if scenario == "drift":
        history = inject_curtailment(history)
    evaluation = [
        (window, target)
        for window, target in sample_windows(history)
        if parse_timestamp(target["timestamp"]) >= start
    ]
    records = [
        {
            "timestamp": target["timestamp"],
            "region": "제주",
            "predicted": incumbent.predict_one(window),
            "actual": target["generation_mwh"],
            "simulation": True,
        }
        for window, target in evaluation
    ]
    check = assess_drift(records, incumbent.metadata.get("drift_threshold_mwh"))
    if len(records) != 168 or not check["ready"]:
        raise ValueError(
            "시뮬레이션에는 임계값과 결측 없는 연속 168시간 평가 구간이 필요합니다"
        )
    result = {
        "simulation": True,
        "scenario": scenario,
        "description": "원본 데이터"
        if scenario == "normal"
        else "08~18시 짝수 시간 발전량을 10%로 제한한 합성 시나리오",
        "start_timestamp": start.isoformat(),
        "cutoff_timestamp": cutoff.isoformat(),
        "base_model_version": str(incumbent.registry_version or incumbent.version),
        "model_name": SIMULATION_MODEL_NAME,
        "simulation_model_version": None,
        "reload_verified": False,
        "records": records,
        "drift_check": check,
        "retraining": None,
    }
    logger.info(
        "[시뮬레이션] %s 배치 평가: RMSE=%.3f MWh, 상태=%s",
        scenario,
        check["rmse"],
        check["status"],
    )
    if check["status"] == "performance_degraded":
        logger.warning(
            "[시뮬레이션] 성능 저하 감지 → 자동 재학습 시작 (%s)", SIMULATION_MODEL_NAME
        )
        try:
            trained = fine_tune(
                history,
                incumbent=incumbent,
                model_name=SIMULATION_MODEL_NAME,
                metadata_extra={
                    "simulation": True,
                    "scenario": scenario,
                    "source_model_version": result["base_model_version"],
                },
            )
        except ValueError as exc:
            trained = {"status": "blocked", "reason": str(exc)}
        result["retraining"] = trained
        if trained.get("promoted"):
            # 운영 모델 캐시는 건드리지 않고 별도 모델을 실제 로딩·추론한다.
            result["simulation_model_version"] = trained["version"]
            try:
                simulated = _load_from_mlflow(SIMULATION_MODEL_NAME)
                if simulated.registry_version != trained["version"]:
                    raise ValueError("승격한 버전과 로딩된 시뮬레이션 버전이 다릅니다")
                result["reloaded_prediction_mwh"] = simulated.predict_one(
                    evaluation[-1][0]
                )
                result["reloaded_target_timestamp"] = evaluation[-1][1]["timestamp"]
                result["reload_verified"] = True
                logger.info(
                    "[시뮬레이션] %s v%s 승격 및 로딩 완료",
                    SIMULATION_MODEL_NAME,
                    simulated.registry_version,
                )
            except Exception as exc:
                # 등록은 이미 완료되었으므로 그 사실을 지우지 않고 후속 실패를 표시한다.
                result["reload_error"] = str(exc)
                logger.exception("[시뮬레이션] 승격 완료 후 로딩·예측 검증 실패")
        else:
            logger.info(
                "[시뮬레이션] 승격 없음: %s",
                trained.get("reason", "검증 게이트 미통과"),
            )
    return result
