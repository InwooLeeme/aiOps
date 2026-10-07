"""제주 태양광 예측 API와 운영 대시보드 진입점."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from serving_app import model_loader
from serving_app.config import LOG_DIR
from serving_app.request_metrics import record_prediction_request
from serving_app.routers import dashboard, data, health, logs, predict

# 태양광 운영 로그를 기존 주가 실험과 분리한다.
# (routers/logs.py가 같은 디렉토리를 읽기 전용으로 노출한다.) 여기서 이 로거 하나만
# 직접 설정하므로, uvicorn 자체 로깅 설정과 충돌하지 않는다.
_LOG_DIR = str(LOG_DIR)
os.makedirs(_LOG_DIR, exist_ok=True)
_aiops_logger = logging.getLogger("solar_aiops")
_aiops_logger.setLevel(logging.INFO)
if not _aiops_logger.handlers:
    _handler = logging.FileHandler(
        os.path.join(_LOG_DIR, "solar_aiops.log"), encoding="utf-8"
    )
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _aiops_logger.addHandler(_handler)
    _aiops_logger.addHandler(logging.StreamHandler())  # 터미널에서도 동일하게 확인 가능


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("LOADING_MODE", "lazy") == "eager":
        model_loader.load_eager()
    else:
        print("[lazy] 모델은 첫 /predict 요청이 들어올 때 로드됩니다.")
    yield


app = FastAPI(title="Jeju Solar Forecast & Model Operations", lifespan=lifespan)
app.middleware("http")(record_prediction_request)

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(data.router)  # 태양광 데이터 업로드
app.include_router(logs.router)  # 대시보드: 재학습 로그 파일 조회
app.include_router(dashboard.router)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount(
    "/", StaticFiles(directory=_STATIC_DIR, html=True), name="static"
)  # 대시보드 UI
