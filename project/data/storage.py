"""원본 태양광 CSV 보관. 다른 도메인의 과거 업로드는 자동 선택하지 않는다."""

from pathlib import Path

UPLOAD_DIR = str(Path(__file__).resolve().parent / "uploads")


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    files = list(Path(upload_dir).glob("solar_*.csv"))
    if not files:
        raise FileNotFoundError("제주 태양광 CSV를 먼저 업로드하세요")
    return str(max(files, key=lambda p: p.stat().st_mtime_ns))
