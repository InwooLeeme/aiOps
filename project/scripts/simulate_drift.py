"""업로드된 제주 CSV 평가 기간 재생. 데이터 생성·자동 재학습은 하지 않습니다."""

import argparse
import json

import requests

TARGETS = {"local": "http://localhost:8077", "container": "http://localhost:8099"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["local", "container", "both"], default="local"
    )
    parser.add_argument(
        "--start", default="2024-01-01T00:00:00", help="평가 시작 시각(KST)"
    )
    parser.add_argument("--limit", type=int, default=168)
    args = parser.parse_args()
    names = list(TARGETS) if args.target == "both" else [args.target]
    for name in names:
        response = requests.post(
            f"{TARGETS[name]}/predict/batch-test",
            json={"start_timestamp": args.start, "limit": args.limit},
            timeout=180,
        )
        response.raise_for_status()
        print(
            json.dumps(
                {"target": name, **response.json()["drift_check"]},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
