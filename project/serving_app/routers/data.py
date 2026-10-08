"""태양광 CSV 업로드·품질 통계·예측 입력 미리보기."""

import uuid
from pathlib import Path

from data.daily_features import (
    SEQ_LEN,
    dataset_summary,
    decode_csv,
    load_rows,
    validate_sequence,
)
from data.storage import SAMPLE_CSV, UPLOAD_DIR, latest_upload
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from serving_app import forecasts

router = APIRouter(prefix="/data")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV는 20MB 이하여야 합니다")
    try:
        rows = decode_csv(raw)
        if len(rows) < SEQ_LEN + 1:
            raise ValueError(f"최소 {SEQ_LEN + 1}행이 필요합니다")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    directory = Path(UPLOAD_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"solar_{uuid.uuid4().hex}.csv"
    path.write_bytes(raw)
    feedback = await run_in_threadpool(
        forecasts.reconcile, forecasts.db_path(request), rows
    )
    return {"filename": path.name, **dataset_summary(rows), "feedback": feedback}


def current_dataset():
    try:
        path = Path(latest_upload())
        return path, load_rows(path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/status")
def status():
    try:
        path, rows = current_dataset()
    except HTTPException as exc:
        if exc.status_code == 404:
            return {"exists": False}
        raise
    return {"filename": path.name, **dataset_summary(rows)}


@router.get("/preview")
def preview():
    path, rows = current_dataset()
    example = None
    for end in range(len(rows), SEQ_LEN - 1, -1):
        try:
            window = validate_sequence(rows[end - SEQ_LEN : end])
        except ValueError:
            continue
        example = {"sequence": window}
        break
    return {
        "filename": path.name,
        "source": "sample" if path.resolve() == SAMPLE_CSV.resolve() else "upload",
        **dataset_summary(rows),
        "example": example,
        **(
            forecasts.forecast_context(example["sequence"][-1]["timestamp"])
            if example
            else {}
        ),
    }
