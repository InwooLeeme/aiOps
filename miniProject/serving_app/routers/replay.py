"""리플레이 시뮬레이션 제어 API: 시작 / 진행 상황 조회."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from serving_app import replay

router = APIRouter(prefix="/replay")


class ReplayRequest(BaseModel):
    start: str = "2023-01-01"
    end: str = "2024-12-31"
    delay: float = 0.0  # 하루당 대기(초) — 화면에서 천천히 보고 싶을 때


@router.post("/start")
def start(req: ReplayRequest):
    if not replay.start(req.start, req.end, req.delay):
        raise HTTPException(409, "이미 리플레이가 실행 중입니다.")
    return {"started": True}


@router.get("/status")
def status():
    return replay.status()
