"""POST /predict — 지정한 날짜의 발전량(MWh) 예측. 입력(과거 14일 실적 + 해당일 예보)은 서버가 보유한 데이터에서 구성한다."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from data.features import SEQ_LEN, make_input, row_matrix
from serving_app import model_loader
from serving_app.dataset import get_table
from serving_app.schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = get_table()
    day = pd.Timestamp(req.date)
    if day not in df.index:
        raise HTTPException(404, f"{req.date} 데이터가 없습니다 ({df.index.min().date()} ~ {df.index.max().date()}).")
    i = df.index.get_loc(day)
    if i < SEQ_LEN:
        raise HTTPException(422, f"과거 {SEQ_LEN}일 이력이 필요합니다.")
    A = row_matrix(df)
    x = make_input(A, i)
    if np.isnan(x).any():
        raise HTTPException(422, "해당 구간의 예보 기상 데이터가 비어 있습니다.")

    model = model_loader.get_model()
    cf = model.predict_cf(x)
    cap = float(df["cap"].iloc[i])
    actual = df["gen"].iloc[i]
    return PredictResponse(date=req.date, predicted_mwh=round(cf * cap * 24, 1), predicted_cf=round(cf, 4), capacity_mw=cap,
                           model_version=model.version, actual_mwh=None if np.isnan(actual) else float(actual))
