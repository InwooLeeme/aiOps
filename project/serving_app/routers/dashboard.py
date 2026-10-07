"""운영 화면에 필요한 요청 지표, 모델 이력, 설정과 최근 이벤트의 읽기 전용 API."""

import math
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Literal

from data.features import SEQ_LEN
from fastapi import APIRouter, Request
from mlflow.tracking import MlflowClient
from sqlalchemy.engine import make_url

from serving_app import config, model_loader
from serving_app.monitoring.drift_detector import (
    RMSE_THRESHOLD,
    WINDOW_SIZE,
    compute_rmse,
)
from serving_app.request_metrics import log_path, summarize
from serving_app.routers.predict import recent_predictions

router = APIRouter()


@router.get("/metrics/summary")
def metrics_summary(request: Request, window: Literal["5m", "1h", "6h", "24h"] = "5m"):
    result = summarize(log_path(request), window)
    recent = list(recent_predictions)
    result["drift"] = {
        "count": len(recent),
        "rmse": compute_rmse(recent) if recent else None,
        "ready": len(recent) >= WINDOW_SIZE,
        "threshold": RMSE_THRESHOLD,
    }
    return result


@router.get("/system/info")
def system_info():
    return {
        "seq_len": SEQ_LEN,
        "rmse_gate": config.RMSE_GATE,
        "window_size": WINDOW_SIZE,
        "rmse_threshold": RMSE_THRESHOLD,
        "base_epochs": config.BASE_EPOCHS,
        "fine_tune_epochs": config.FINE_TUNE_EPOCHS,
        "fine_tune_lr": config.FINE_TUNE_LR,
        "model_source": os.getenv("MODEL_SOURCE", "local"),
        "loading_mode": os.getenv("LOADING_MODE", "lazy"),
    }


def version_info(client: MlflowClient, version) -> dict:
    score, mode = None, None
    if version.run_id:
        run = client.get_run(version.run_id)
        score = run.data.metrics.get("rmse")
        mode = run.data.params.get("mode")
    return {
        "version": str(version.version),
        "created_at": version.creation_timestamp / 1000,
        "stage": version.current_stage,
        "mode": mode,
        "rmse": score if score is not None and math.isfinite(score) else None,
    }


@router.get("/models/overview")
def models_overview():
    cached = model_loader._model_cache
    source = os.getenv("MODEL_SOURCE", "local")
    registry_version = getattr(cached, "registry_version", None)
    result = {
        "model_name": config.MODEL_NAME,
        "source": source,
        "model_loaded": cached is not None,
        "served_model": None,
        "production": None,
        "versions": [],
        "reload_required": False,
        "message": None,
    }
    if cached is not None:
        result["served_model"] = {
            "version": registry_version or cached.version,
            "rmse": None,
            "mode": None,
            "stage": "Local" if source == "local" else "Unknown",
            "created_at": None,
        }
    uri = os.getenv("MLFLOW_TRACKING_URI", config.DEFAULT_TRACKING_URI)
    try:
        database = make_url(uri).database if uri.startswith("sqlite:") else None
        if database and database != ":memory:" and not Path(database).is_file():
            result["message"] = (
                "MLflow 저장소가 없습니다. Day2 학습과 모델 등록을 먼저 실행하세요."
            )
            return result
        client = MlflowClient(tracking_uri=uri)
        versions = client.search_model_versions(
            f"name='{config.MODEL_NAME}'",
            max_results=1000,
            order_by=["version_number DESC"],
        )
        production = next(
            (v for v in versions if v.current_stage == "Production"), None
        )
        result["versions"] = [version_info(client, v) for v in versions[:30]]
        if production:
            result["production"] = version_info(client, production)
            result["reload_required"] = bool(
                source == "mlflow"
                and registry_version
                and registry_version != str(production.version)
            )
        served = next((v for v in versions if str(v.version) == registry_version), None)
        if served:
            result["served_model"] = version_info(client, served)
        if not versions:
            result["message"] = (
                "등록된 HAIC 모델이 없습니다. Day2 학습을 먼저 실행하세요."
            )
    except Exception:
        # 조회 실패가 모델 서빙이나 다른 화면을 중단하지 않도록 상태를 표시합니다.
        result["message"] = (
            "MLflow 정보를 읽지 못했습니다. 서버 로그와 저장소 연결을 확인하세요."
        )
        import logging

        logging.getLogger(__name__).exception("MLflow 대시보드 조회 실패")
    return result


@router.get("/events/recent")
def recent_events():
    path = config.LOG_DIR / "aiops.log"
    if not path.is_file():
        return []
    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[(\w+)\] (.*)$"
    )
    with path.open(encoding="utf-8") as stream:
        lines = deque(stream, maxlen=200)
    events = []
    for line in reversed(lines):
        match = pattern.match(line.rstrip())
        if match:
            stamp, level, message = match.groups()
            try:
                timestamp = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f").timestamp()
            except ValueError:
                continue
            events.append(
                {
                    "timestamp": timestamp,
                    "level": level,
                    "message": message,
                }
            )
        if len(events) == 20:
            break
    return events
