"""제주 CSV로 시간순 학습 후 model/scaler/metadata 로컬 번들을 저장합니다."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving_app.config import BASE_EPOCHS
from serving_app.train_and_register import train_local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="생략하면 최신 업로드 또는 제주 샘플 사용")
    parser.add_argument("--epochs", type=int, default=BASE_EPOCHS)
    args = parser.parse_args()
    print(json.dumps(train_local(args.csv, args.epochs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
