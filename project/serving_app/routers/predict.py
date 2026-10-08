"""태양광 다음 날짜 총발전량 예측과 드리프트 시뮬레이션·자동 재학습."""

import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from data.daily_features import decode_csv, parse_timestamp, sample_windows
from data.storage import latest_upload
from fastapi import APIRouter, HTTPException, Query, Request

from serving_app import forecasts, model_loader
from serving_app.config import MODEL_NAME, RUNTIME_DIR
from serving_app.evaluation import wape_pct
from serving_app.monitoring.drift_detector import WINDOW_SIZE, assess_drift
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.schemas import (
    BatchTestResponse,
    PredictRequest,
    PredictResponse,
    SimulationRequest,
)

router = APIRouter()
recent_predictions: list[dict] = []
REPLAY_LOG = RUNTIME_DIR / "solar_daily_replay.jsonl"
SIMULATION_LOG = RUNTIME_DIR / "solar_daily_simulation.jsonl"
_replay_lock = threading.Lock()
_last_replay: dict = {}
_batch_contexts: dict = {}


def current_model():
    try:
        return model_loader.get_model()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, f"태양광 모델을 먼저 학습·준비하세요: {exc}") from exc


def model_identity(model):
    return str(model.registry_version or model.version)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    model = current_model()
    sequence = [p.model_dump() for p in req.sequence]
    context = forecasts.forecast_context(sequence[-1]["timestamp"])
    if context["forecast_context"] == "future_input":
        raise HTTPException(422, "아직 관측할 수 없는 미래 시각의 입력입니다")
    try:
        value = model.predict_one(sequence)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not math.isfinite(value):
        raise HTTPException(503, "모델이 유효한 발전량을 반환하지 않았습니다")
    context = forecasts.forecast_context(sequence[-1]["timestamp"])
    record = forecasts.save_prediction(
        forecasts.db_path(request), context, round(value, 5), model
    )
    return PredictResponse(
        predicted_generation_mwh=record["predicted"],
        target_timestamp=record["target_timestamp"],
        region="제주",
        model_version=model_identity(model),
        input_end_timestamp=record["input_end_timestamp"],
        issued_at=record["issued_at"],
        prediction_id=record["prediction_id"],
        forecast_context=record["forecast_context"],
        monitoring_eligible=bool(record["monitoring_eligible"]),
        exclusion_reason=record["exclusion_reason"],
    )


@router.get("/predictions/recent")
def prediction_history(request: Request):
    return forecasts.summary(forecasts.db_path(request), model_loader._model_cache)


def batch_history(dataset_hash, identity):
    """재시작 후에도 같은 관측으로 재학습하지 않도록 누적 로그를 읽는다."""
    if not REPLAY_LOG.is_file():
        return {}
    records = {}
    with REPLAY_LOG.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if (
                record.get("dataset_sha256") == dataset_hash
                and record.get("model_version") == identity
            ):
                records[record["timestamp"]] = record
    recent = sorted(records.values(), key=lambda r: r["timestamp"])[-WINDOW_SIZE:]
    return {"cutoff": recent[-1]["timestamp"], "records": recent} if recent else {}


def evaluate_batch(
    rows,
    model,
    start_timestamp,
    limit,
    dataset_hash,
    *,
    strict=False,
    metadata_extra=None,
):
    """호출자가 공용 잠금을 보유한 상태에서 평가·누적·재학습한다."""
    evaluated_until = max(
        model.metadata.get("validation_end") or "",
        model.metadata.get("training_end") or "",
    )
    if not evaluated_until or parse_timestamp(start_timestamp) <= parse_timestamp(
        evaluated_until
    ):
        raise HTTPException(
            422, "모델 학습·검증 종료 시각 이후의 이력만 평가할 수 있습니다"
        )
    records = []
    identity = model_identity(model)
    context_key = (dataset_hash, identity)
    previous = _batch_contexts.get(context_key)
    if previous is None:
        previous = batch_history(dataset_hash, identity)
    observed_until = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
    for window, target in sample_windows(rows):
        if target["timestamp"] < start_timestamp:
            continue
        if parse_timestamp(target["timestamp"]) + timedelta(days=1) > observed_until:
            raise HTTPException(
                422, "미래 시각의 관측값은 과거 평가에 사용할 수 없습니다"
            )
        records.append(
            {
                "timestamp": target["timestamp"],
                "region": "제주",
                "predicted": model.predict_one(window),
                "actual": target["generation_mwh"],
                "model_version": model_identity(model),
            }
        )
        if len(records) >= limit:
            break
    if not records:
        raise HTTPException(
            422, "해당 기간에 결측 없는 14일 입력과 다음 날짜 총발전량 정답이 없습니다"
        )
    if strict and (
        len(records) != WINDOW_SIZE
        or records[0]["timestamp"] != start_timestamp
        or not assess_drift(records, model.metadata.get("drift_threshold_mwh"))["ready"]
    ):
        raise HTTPException(
            422, "시뮬레이션에는 결측 없는 연속 14일 평가 구간이 필요합니다"
        )
    threshold = model.metadata.get("drift_threshold_mwh")
    if previous and records[-1]["timestamp"] <= previous["cutoff"]:
        # 선택한 구간만 다시 평가하며 감시 상태·누적 로그·재학습은 건드리지 않는다.
        return BatchTestResponse(
            dataset_sha256=dataset_hash,
            predictions=[r["predicted"] for r in records],
            records=records,
            drift_check={
                **assess_drift(records, threshold),
                "evaluation_only": True,
                "new_count": 0,
                "model_version": identity,
                "retraining": None,
            },
        )
    previous_cutoff = previous.get("cutoff", "")
    fresh = [r for r in records if r["timestamp"] > previous_cutoff]
    accumulated = list(previous.get("records", [])) + fresh
    accumulated = accumulated[-WINDOW_SIZE:]
    replay_id = uuid.uuid4().hex
    REPLAY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REPLAY_LOG.open("a", encoding="utf-8") as stream:
        for record in fresh:
            stream.write(
                json.dumps(
                    {
                        **record,
                        "replay_id": replay_id,
                        "dataset_sha256": dataset_hash,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    recent_predictions[:] = accumulated
    _last_replay.clear()
    _last_replay.update(previous)
    if fresh:
        recent_predictions[:] = accumulated
        _last_replay.clear()
        _last_replay.update(
            cutoff=records[-1]["timestamp"],
            dataset_hash=dataset_hash,
            model_version=identity,
            attempted=False,
            synthetic=bool(metadata_extra),
        )
        check = check_and_trigger(
            accumulated,
            threshold,
            rows=rows,
            incumbent=model,
            metadata_extra=metadata_extra,
        )
        trained = check.get("retraining")
        _last_replay.update(
            check=check,
            attempted=bool(trained)
            and os.getenv("MODEL_SOURCE", "local").lower() == "mlflow",
        )
        if trained and trained.get("promoted"):
            # 새 모델의 임계값으로 이전 모델의 오차를 평가하지 않는다.
            recent_predictions.clear()
    else:
        check = {**_last_replay.get("check", assess_drift(accumulated, threshold))}
    _batch_contexts[context_key] = {**_last_replay, "records": accumulated}
    check = {
        **check,
        "new_count": len(fresh),
        "model_version": identity,
        "evaluation_only": False,
    }
    return BatchTestResponse(
        dataset_sha256=dataset_hash,
        predictions=[r["predicted"] for r in records],
        records=records,
        drift_check=check,
    )


def open_bytes(path):
    from pathlib import Path

    return Path(path).read_bytes()


@router.get("/simulation/status")
def simulation_status():
    if not SIMULATION_LOG.is_file():
        return {"exists": False}
    with SIMULATION_LOG.open(encoding="utf-8") as stream:
        last = deque(stream, maxlen=1)
    result = json.loads(last[0]) if last else {"exists": False}
    if result.get("model_name") != MODEL_NAME:
        return {
            "exists": False,
            "reason": "이전 격리 시뮬레이터 기록은 운영 결과에서 제외합니다",
        }
    return result


@router.get("/simulation/history")
def simulation_history(limit: int = Query(default=30, ge=1, le=100)):
    batches = deque(maxlen=limit)
    if not SIMULATION_LOG.is_file():
        return {"batches": []}
    with SIMULATION_LOG.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue  # 실행 로그를 쓰는 도중의 불완전한 줄은 다음 조회에서 읽는다.
            if result.get("model_name") != MODEL_NAME:
                continue
            records = result.get("records") or []
            complete = bool(records) and all(
                isinstance(r.get(key), (int, float)) and math.isfinite(r[key])
                for r in records
                for key in ("actual", "predicted")
            )
            score = (
                wape_pct(
                    [r["actual"] for r in records], [r["predicted"] for r in records]
                )
                if complete
                else None
            )
            batches.append(
                {
                    **{
                        key: result.get(key)
                        for key in (
                            "simulation_id",
                            "start_timestamp",
                            "cutoff_timestamp",
                            "scenario",
                            "base_model_version",
                            "completed_at",
                        )
                    },
                    "wape_pct": score,
                    "evaluation_only": result.get("drift_check", {}).get(
                        "evaluation_only"
                    ),
                }
            )
    return {"batches": list(batches)}


@router.post("/simulation/run")
def simulate(req: SimulationRequest):
    from serving_app.monitoring.simulation import run_simulation

    if not _replay_lock.acquire(blocking=False):
        raise HTTPException(409, "다른 시뮬레이션 또는 재학습이 진행 중입니다")
    try:
        if os.getenv("MODEL_SOURCE", "local") != "mlflow":
            raise HTTPException(422, "시뮬레이션은 MODEL_SOURCE=mlflow에서 실행하세요")
        model = current_model()
        try:
            path = latest_upload()
            source = open_bytes(path)
            result = run_simulation(
                decode_csv(source),
                model,
                req.scenario,
                req.start_timestamp,
                dataset_hash=hashlib.sha256(source).hexdigest(),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        result.update(
            exists=True,
            completed_at=time.time(),
            simulation_id=uuid.uuid4().hex,
            dataset_sha256=hashlib.sha256(source).hexdigest(),
        )
        SIMULATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SIMULATION_LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        return result
    finally:
        _replay_lock.release()
