"""
HAIC 가상 데이터 업로드 - data/generate_haic_data.py로 자동 생성하던 방식을 대체합니다.

/data 폴더는 이 라우터로 업로드된 CSV만 쌓이는 곳입니다(data/uploads/). 여러 번
업로드하면 계속 쌓이고, 학습(train_and_register.py, fine_tune 등)은 항상 가장
최근 파일 하나를 사용합니다(data/storage.py의 latest_upload()).

대시보드(static/index.html)에서 파일을 올리면 이 엔드포인트가 호출됩니다.
"""

import csv
import io
import os
import time

from data.features import SEQ_LEN, load_rows
from data.storage import UPLOAD_DIR, latest_upload
from fastapi import APIRouter, File, HTTPException, UploadFile

from serving_app.config import DATA_DIR
from serving_app.monitoring.drift_detector import WINDOW_SIZE

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = {"Date", "Close", "Volume"}
MIN_ROWS = (
    SEQ_LEN + WINDOW_SIZE
)  # 시퀀스 구성 + 드리프트 판정 윈도우에 필요한 최소 행 수


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
        raise HTTPException(
            400, f"CSV에 {sorted(REQUIRED_COLUMNS)} 컬럼이 모두 있어야 합니다."
        )
    rows = list(reader)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"haic_{int(time.time())}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {"filename": os.path.basename(dest), "rows": len(rows)}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    rows = load_rows(path)
    closes = [r["Close"] for r in rows]
    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": len(rows),
        "start_date": rows[0]["Date"],
        "end_date": rows[-1]["Date"],
        "min_close": min(closes),
        "max_close": max(closes),
    }


@router.get("/preview")
def preview():
    """최신 업로드 데이터, 없으면 샘플 CSV의 통계와 최근 입력 예제를 제공합니다."""
    try:
        path = latest_upload()
        source = "upload"
    except FileNotFoundError:
        path = DATA_DIR / "sample_haic_prices.csv"
        source = "sample"
    try:
        rows = load_rows(path)
        if not rows:
            raise ValueError("empty CSV")
    except (OSError, ValueError, KeyError):
        raise HTTPException(
            422, "CSV 데이터를 읽을 수 없습니다. 날짜·종가·거래량을 확인하세요."
        ) from None
    closes = [row["Close"] for row in rows]
    return {
        "filename": os.path.basename(path),
        "source": source,
        "rows": len(rows),
        "start_date": rows[0]["Date"],
        "end_date": rows[-1]["Date"],
        "min_close": min(closes),
        "max_close": max(closes),
        "avg_volume": sum(row["Volume"] for row in rows) / len(rows),
        "example": {
            "sequence": [
                {"close": row["Close"], "volume": row["Volume"]}
                for row in rows[-SEQ_LEN:]
            ]
        },
    }
