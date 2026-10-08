"""
대시보드 전용 읽기 API (/api/*). 화면은 이 응답만 그리고, 판단 로직은 갖지 않는다.

  GET /api/summary   KPI 카드 + 현재 운영 모델 + 재학습 이력 + 최근 알람 + 파이프라인 단계
  GET /api/datasets  업로드된 CSV 목록
  GET /api/system    서버/모델 소스/MLflow 상태
"""
import glob
import os
import re
import time

from fastapi import APIRouter

from data.storage import UPLOAD_DIR
from serving_app import model_loader
from serving_app.monitoring import metrics
from serving_app import replay
from serving_app.monitoring.drift_detector import WAPE_THRESHOLD, WINDOW_SIZE, compute_bias, compute_wape

router = APIRouter(prefix="/api")

MODEL_NAME = "Solar_Predictor"
LOG_PATH = os.path.join("logs", "aiops.log")
LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[(\w+)\] (.*)$")


def _model_versions() -> list[dict]:
    """MLflow Registry의 Solar_Predictor 버전 목록(최신순). MLflow 미구성 시 빈 목록."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        out = []
        for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
            run = client.get_run(v.run_id)
            out.append({
                "version": int(v.version),
                "registered_at": int(v.creation_timestamp) // 1000,
                "mode": run.data.params.get("mode", "-"),
                **{k: (None if run.data.metrics.get(k) is None else round(run.data.metrics[k], 1))
                   for k in ("rmse_mwh", "mae_mwh", "wape", "bias")},
                "stage": v.current_stage,
            })
        return sorted(out, key=lambda x: -x["version"])
    except Exception:
        return []


def _alerts(limit: int = 8) -> list[dict]:
    if not os.path.isfile(LOG_PATH):
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    alerts = []
    for line in reversed(lines):
        m = LOG_LINE.match(line)
        if not m:
            continue
        ts, level, msg = m.groups()
        epoch = int(time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S")))
        alerts.append({"time": epoch, "level": level, "message": msg})
        if len(alerts) >= limit:
            break
    return alerts


def _drift_state() -> dict:
    """최근 WINDOW_SIZE일 WAPE(%)와 고정 임계치."""
    window = replay.window_snapshot()
    if len(window) < WINDOW_SIZE:
        return {"ran": False, "wape": None, "bias": None, "threshold": WAPE_THRESHOLD, "drifted": False, "n": len(window)}
    w = compute_wape(window[-WINDOW_SIZE:])
    return {"ran": True, "wape": round(w, 1), "bias": round(compute_bias(window[-WINDOW_SIZE:]), 1), "threshold": WAPE_THRESHOLD,
            "drifted": w > WAPE_THRESHOLD, "n": len(window)}


def _pipeline(alerts: list[dict], drift: dict) -> list[dict]:
    """로그에 남은 가장 최근 이벤트로 각 단계의 상태(idle/done)를 추정한다."""
    msgs = " ".join(a["message"] for a in alerts)
    ran = drift["ran"] or bool(alerts)
    retrained = "retrain triggered" in msgs
    promoted = "production promoted" in msgs
    steps = [
        ("데이터 수집", "요청 수신", ran),
        ("데이터 모니터링", "예측 기록 누적", ran),
        ("드리프트 감지", "WAPE vs 임계치", ran),
        ("재학습 트리거", "조건 충족 시 시작", retrained),
        ("모델 학습", "fine-tuning", retrained),
        ("모델 등록", "MLflow Registry", retrained),
        ("배포", "Production 승격", promoted),
    ]
    return [{"name": n, "desc": d, "state": "done" if ok else "idle"} for n, d, ok in steps]


@router.get("/summary")
def summary():
    m = metrics.snapshot()
    versions = _model_versions()
    prod = next((v for v in versions if v["stage"] == "Production"), None)
    drift = _drift_state()
    alerts = _alerts()
    return {
        "kpi": {
            "requests": m["requests"],
            "avg_latency_ms": m["avg_latency_ms"],
            "success_rate": m["success_rate"],
            "rmse_mwh": prod["rmse_mwh"] if prod else None,
            "wape": prod["wape"] if prod else None,
            "drift": drift,
        },
        "series": {**m["series"], "rmse": [v["rmse_mwh"] for v in reversed(versions) if v["rmse_mwh"] is not None]},
        "model": {
            "name": MODEL_NAME,
            "healthy": model_loader._model_cache is not None,
            "production": prod,
            "gate": None,
        },
        "versions": versions,
        "alerts": alerts,
        "pipeline": _pipeline(alerts, drift),
    }


@router.get("/datasets")
def datasets():
    files = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.csv")), key=os.path.getmtime, reverse=True)
    return [
        {"name": os.path.basename(p), "size": os.path.getsize(p), "uploaded_at": int(os.path.getmtime(p)), "latest": i == 0}
        for i, p in enumerate(files)
    ]


@router.get("/system")
def system():
    return {
        "model_source": os.getenv("MODEL_SOURCE", "mlflow"),
        "loading_mode": os.getenv("LOADING_MODE", "lazy"),
        "model_loaded": model_loader._model_cache is not None,
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "(default: ./mlflow.db)"),
        "wape_threshold": WAPE_THRESHOLD,
        "drift_window": WINDOW_SIZE,
    }
