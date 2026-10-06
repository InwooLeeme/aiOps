from fastapi import FastAPI

from app.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="AI Ops API",
    description="AI Ops 서비스 API",
    version="0.1.0",
)


@app.get("/", tags=["default"])
async def root() -> dict[str, str]:
    return {"message": "AI Ops API", "docs": "/docs"}


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # 예측 로직을 여기에 구현합니다.
    # 예를 들어, 모델을 로드하고 데이터를 입력으로 사용하여 예측을 수행할 수 있습니다.
    # 예측 결과를 반환합니다.
    # label, score = model.predict([req.text])[0]  # 모델 예측 수행
    return PredictResponse(label="positive", score=0.95)  # 예시 응답
    # return PredictResponse(label=label, score=float(score))
