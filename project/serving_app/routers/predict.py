"""태양광 다음 시간 예측, 과거 이력 재생, 관측 시점으로 제한한 재학습."""

import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from datetime import timedelta

from data.features import load_rows, parse_timestamp, sample_windows
from data.storage import latest_upload
from fastapi import APIRouter, HTTPException

from serving_app import model_loader
from serving_app.config import RUNTIME_DIR
from serving_app.monitoring.retrain_trigger import check_and_trigger, retrain_at
from serving_app.schemas import (
    BatchTestRequest,
    BatchTestResponse,
    PredictRequest,
    PredictResponse,
    RetrainRequest,
    SimulationRequest,
)

router = APIRouter()
recent_predictions: list[dict] = []
REPLAY_LOG = RUNTIME_DIR / "solar_replay.jsonl"
SIMULATION_LOG = RUNTIME_DIR / "solar_simulation.jsonl"
_replay_lock = threading.Lock()
_last_replay: dict = {}


def current_model():
    try:
        return model_loader.get_model()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, f"태양광 모델을 먼저 학습·준비하세요: {exc}") from exc


def model_identity(model):
    return str(model.registry_version or model.version)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = current_model()
    sequence = [p.model_dump() for p in req.sequence]
    try:
        value = model.predict_one(sequence)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not math.isfinite(value):
        raise HTTPException(503, "모델이 유효한 발전량을 반환하지 않았습니다")
    target = parse_timestamp(sequence[-1]["timestamp"]) + timedelta(hours=1)
    return PredictResponse(
        predicted_generation_mwh=round(value, 5),
        target_timestamp=target.isoformat(),
        region="제주",
        model_version=model_identity(model),
    )


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def replay(req: BatchTestRequest):
    if not _replay_lock.acquire(blocking=False):
        raise HTTPException(409, "다른 이력 재생 또는 재학습이 진행 중입니다")
    try:
        model = current_model()
        evaluated_until = model.metadata.get("validation_end") or model.metadata.get(
            "training_end"
        )
        if evaluated_until is None or parse_timestamp(
            req.start_timestamp
        ) <= parse_timestamp(evaluated_until):
            raise HTTPException(
                422, "모델 학습·검증 종료 시각 이후의 이력만 평가할 수 있습니다"
            )
        try:
            path = latest_upload()
            rows = load_rows(path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        records = []
        # 새 재생은 독립된 평가다. 이전 배치와 중복·역순 데이터를 합치지 않는다.
        for window, target in sample_windows(rows):
            if target["timestamp"] < req.start_timestamp:
                continue
            records.append(
                {
                    "timestamp": target["timestamp"],
                    "region": "제주",
                    "predicted": model.predict_one(window),
                    "actual": target["generation_mwh"],
                    "model_version": model_identity(model),
                }
            )
            if len(records) >= req.limit:
                break
        if not records:
            raise HTTPException(
                422, "해당 기간에 결측 없는 72시간 입력과 다음 시간 정답이 없습니다"
            )
        threshold = model.metadata.get("drift_threshold_mwh")
        check = check_and_trigger(records, threshold)
        dataset_hash = hashlib.sha256(open_bytes(path)).hexdigest()
        replay_id = uuid.uuid4().hex
        REPLAY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with REPLAY_LOG.open("a", encoding="utf-8") as stream:
            for record in records:
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
        recent_predictions[:] = records[-168:]
        _last_replay.clear()
        _last_replay.update(
            {
                "cutoff": records[-1]["timestamp"],
                "dataset_hash": dataset_hash,
                "model_version": model_identity(model),
                "check": check,
            }
        )
        return BatchTestResponse(
            predictions=[r["predicted"] for r in records],
            records=records,
            drift_check=check,
        )
    finally:
        _replay_lock.release()


def open_bytes(path):
    from pathlib import Path

    return Path(path).read_bytes()


@router.get("/simulation/status")
def simulation_status():
    if not SIMULATION_LOG.is_file():
        return {"exists": False}
    with SIMULATION_LOG.open(encoding="utf-8") as stream:
        last = deque(stream, maxlen=1)
    return json.loads(last[0]) if last else {"exists": False}


@router.post("/simulation/run")
def simulate(req: SimulationRequest):
    from serving_app.monitoring.simulation import run_simulation

    if not _replay_lock.acquire(blocking=False):
        raise HTTPException(409, "다른 이력 재생 또는 재학습이 진행 중입니다")
    try:
        if os.getenv("MODEL_SOURCE", "local") != "mlflow":
            raise HTTPException(422, "시뮬레이션은 MODEL_SOURCE=mlflow에서 실행하세요")
        model = current_model()
        try:
            path = latest_upload()
            result = run_simulation(
                load_rows(path), model, req.scenario, req.start_timestamp
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        result.update(
            exists=True,
            completed_at=time.time(),
            simulation_id=uuid.uuid4().hex,
            dataset_sha256=hashlib.sha256(open_bytes(path)).hexdigest(),
        )
        SIMULATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SIMULATION_LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        return result
    finally:
        _replay_lock.release()


@router.post("/retrain")
def retrain(req: RetrainRequest):
    if not _replay_lock.acquire(blocking=False):
        raise HTTPException(409, "다른 이력 재생 또는 재학습이 진행 중입니다")
    try:
        if not _last_replay or req.cutoff_timestamp != _last_replay["cutoff"]:
            raise HTTPException(
                422, "재학습 기준은 마지막으로 평가한 실제 관측 시각과 같아야 합니다"
            )
        if os.getenv("MODEL_SOURCE", "local") != "mlflow":
            return {
                "status": "blocked",
                "reason": "MLflow Production 모델 등록 후 "
                "MODEL_SOURCE=mlflow로 실행하세요",
            }
        if _last_replay["check"]["status"] != "performance_degraded":
            return {
                "status": "blocked",
                "reason": "최근 이력에서 지속적인 성능 저하가 확인되지 않았습니다",
            }
        model = current_model()
        if model_identity(model) != _last_replay["model_version"]:
            return {
                "status": "blocked",
                "reason": "현재 모델 버전으로 이력을 다시 평가하세요",
            }
        path = latest_upload()
        if hashlib.sha256(open_bytes(path)).hexdigest() != _last_replay["dataset_hash"]:
            raise HTTPException(
                422, "데이터가 바뀌었습니다. 새 CSV로 이력을 다시 평가하세요"
            )
        try:
            result = retrain_at(load_rows(path), req.cutoff_timestamp)
        except ValueError as exc:
            return {"status": "blocked", "reason": str(exc)}
        _last_replay.clear()
        recent_predictions.clear()
        return {"status": "retrain_completed", **result}
    finally:
        _replay_lock.release()
