"""태양광 서비스 운영 로그의 읽기 전용 조회."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from serving_app.config import LOG_DIR as PROJECT_LOG_DIR

router = APIRouter(prefix="/logs")

LOG_DIR = str(PROJECT_LOG_DIR)


@router.get("")
def list_logs():
    if not os.path.isdir(LOG_DIR):
        return []
    files = []
    for name in sorted(os.listdir(LOG_DIR)):
        path = os.path.join(LOG_DIR, name)
        if os.path.isfile(path) and not os.path.islink(path):
            files.append({"name": name, "size": os.path.getsize(path)})
    return files


@router.get("/{filename}")
def read_log(filename: str):
    # 경로 조작(디렉토리 탈출) 방지: 순수 파일명만 허용
    if (
        filename != os.path.basename(filename)
        or Path(LOG_DIR, filename).is_symlink()
        or Path(LOG_DIR, filename).resolve().parent != Path(LOG_DIR).resolve()
    ):
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다")

    path = os.path.join(LOG_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="로그 파일을 찾을 수 없습니다")

    with open(path, encoding="utf-8") as f:
        content = f.read()
    return {"name": filename, "content": content}
