"""/predict 요청·응답 스키마."""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="예측 대상일 (YYYY-MM-DD). 서버가 보유한 데이터의 전일까지 실적을 사용")


class PredictResponse(BaseModel):
    date: str
    predicted_mwh: float
    predicted_cf: float
    capacity_mw: float
    model_version: str
    actual_mwh: float | None = None  # 실적이 이미 있으면(과거 날짜) 비교용으로 함께 반환
