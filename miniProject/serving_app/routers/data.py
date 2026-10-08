"""
발전량 CSV 업로드/상태.

업로드한 CSV 는 "가장 최근 파일 하나가 데이터 전체"로 쓰이므로, 덮어쓰기가 아니라 **날짜 기준으로 기존 데이터에 합친다**
(2024년 파일만 올려도 2019~2023 이력이 사라지지 않게). 합친 결과를 새 파일로 저장한다.

받는 형식(둘 중 하나):
  · 원본:      날짜(KST), 일 발전량 합계(MWh), 설비용량 평균(MW)
  · 분할 CSV:  date, generation_mwh, capacity_mw   (scripts/export_splits.py 가 만든 파일)
주의: 예보 기상은 data/weather_jeju_forecast.csv 에서 날짜로 붙인다. 그 파일이 덮지 않는 날짜(예: 2025년)는 예측할 수 없다.
"""
import io
import os
import time

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import SEQ_LEN, UPLOAD_COLS
from data.storage import UPLOAD_DIR, latest_upload

router = APIRouter(prefix="/data")

CANON = [UPLOAD_COLS["date"], UPLOAD_COLS["gen"], UPLOAD_COLS["cap"]]
ALT = ["date", "generation_mwh", "capacity_mw"]
MIN_TOTAL_ROWS = SEQ_LEN + 60  # 합친 뒤 시퀀스 구성 + 재학습에 필요한 최소 이력
TOL = 0.01                     # 이 이하의 차이(분할 CSV 의 반올림 등)는 "같은 값"으로 본다


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    for names in (CANON, ALT):
        if set(names) <= set(df.columns):
            out = df[names].copy()
            out.columns = CANON
            out[CANON[0]] = pd.to_datetime(out[CANON[0]], errors="coerce").dt.strftime("%Y-%m-%d")
            if out[CANON[0]].isna().any():
                raise HTTPException(400, "날짜 컬럼에 해석할 수 없는 값이 있습니다.")
            for c in CANON[1:]:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            return out.drop_duplicates(CANON[0], keep="last").set_index(CANON[0]).sort_index()
    raise HTTPException(400, f"CSV에 다음 중 한 형식의 컬럼이 필요합니다: {CANON} 또는 {ALT}")


def _differs(a: pd.Series, b: pd.Series) -> pd.Series:
    return ~((a.isna() & b.isna()) | ((a - b).abs() <= TOL))


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")
    try:
        new = _normalize(pd.read_csv(io.StringIO(text)))
    except pd.errors.ParserError as e:
        raise HTTPException(400, f"CSV를 읽을 수 없습니다: {e}")
    if new.empty:
        raise HTTPException(400, "데이터 행이 없습니다.")

    try:
        base = _normalize(pd.read_csv(latest_upload(), encoding="utf-8-sig"))
    except FileNotFoundError:
        base = new.iloc[0:0]

    added = new.index.difference(base.index)
    both = new.index.intersection(base.index)
    changed = both[(_differs(new.loc[both, CANON[1]], base.loc[both, CANON[1]]) |
                    _differs(new.loc[both, CANON[2]], base.loc[both, CANON[2]])).values] if len(both) else both
    if len(added) == 0 and len(changed) == 0:
        return {"changed": False, "message": "이미 같은 데이터가 있어 변경 없음", "rows_in_file": len(new), "total_rows": len(base)}

    merged = pd.concat([base.drop(index=changed), new.loc[changed], new.loc[added]]).sort_index()
    if len(merged) < MIN_TOTAL_ROWS:
        raise HTTPException(400, f"합친 데이터가 최소 {MIN_TOTAL_ROWS}행 이상이어야 합니다 (현재 {len(merged)}행).")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"solar_{time.time_ns() // 1_000_000}.csv")
    merged.reset_index().to_csv(dest, index=False, encoding="utf-8")
    return {"changed": True, "filename": os.path.basename(dest), "rows_in_file": len(new), "added": len(added),
            "updated": len(changed), "total_rows": len(merged), "start_date": merged.index.min(), "end_date": merged.index.max()}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}
    from serving_app.dataset import get_table

    df = get_table()
    return {"exists": True, "filename": os.path.basename(path), "rows": len(df),
            "start_date": str(df.index.min().date()), "end_date": str(df.index.max().date()),
            "missing_generation_days": int(df["gen"].isna().sum())}
