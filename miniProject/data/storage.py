"""업로드된 발전량 CSV 관리. 학습·리플레이는 항상 data/uploads/ 의 가장 최근 파일을 사용한다."""
import glob
import os

UPLOAD_DIR = "data/uploads"


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"업로드된 발전량 CSV가 없습니다. 대시보드에서 올리거나 {upload_dir}/ 에 넣어주세요.")
    return files[-1]
