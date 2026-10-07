"""원본 태양광 CSV 보관. 다른 도메인의 과거 업로드는 자동 선택하지 않는다."""

from pathlib import Path

UPLOAD_DIR = str(Path(__file__).resolve().parent / "uploads")


SAMPLE_CSV = Path(__file__).resolve().parent / "sample_jeju_solar.csv"


def latest_upload(upload_dir: str | None = None) -> str:
    files = list(Path(upload_dir or UPLOAD_DIR).glob("solar_*.csv"))
    if not files:
        if SAMPLE_CSV.is_file():
            return str(SAMPLE_CSV)
        raise FileNotFoundError("제주 태양광 CSV를 업로드하거나 샘플을 준비하세요")
    return str(max(files, key=lambda p: p.stat().st_mtime_ns))
