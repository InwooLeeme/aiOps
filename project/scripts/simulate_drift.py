"""태양광 정상·합성 드리프트 배치를 실행하고 서버별 결과를 비교합니다.

서버의 같은 원본 CSV에 동일한 변환을 적용합니다. 게이트 통과 시 운영 모델이
교체되므로 승격 이후에는 새 모델의 학습·검증 종료 이후 구간을 선택하세요.
"""

import argparse
import json

import requests

TARGETS = {"local": "http://localhost:8077", "container": "http://localhost:8099"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["local", "container", "both"], default="local"
    )
    parser.add_argument(
        "--scenario", choices=["normal", "drift", "both"], default="both"
    )
    parser.add_argument(
        "--start", default="2024-05-23T17:00:00", help="평가 시작 시각(KST)"
    )
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout은 양수여야 합니다")
    names = list(TARGETS) if args.target == "both" else [args.target]
    scenarios = ["normal", "drift"] if args.scenario == "both" else [args.scenario]
    results = []
    for scenario in scenarios:
        payload = {"start_timestamp": args.start, "scenario": scenario}
        for name in names:
            item = {"target": name, "scenario": scenario}
            try:
                response = requests.post(
                    TARGETS[name] + "/simulation/run",
                    json=payload,
                    timeout=args.timeout,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or not isinstance(
                    data.get("drift_check"), dict
                ):
                    raise ValueError("drift_check가 없는 서버 응답입니다")
                item.update(ok=True, result=data)
            except (requests.RequestException, ValueError) as exc:
                item.update(ok=False, error=str(exc))
            results.append(item)
    # 해시가 없거나 서버가 실패하면 동일 데이터 비교로 표시하지 않는다.
    comparable = len(names) > 1 and all(r["ok"] for r in results)
    for scenario in scenarios:
        hashes = [
            r.get("result", {}).get("dataset_sha256")
            for r in results
            if r["scenario"] == scenario
        ]
        comparable = comparable and all(hashes) and len(set(hashes)) == 1
    print(
        json.dumps(
            {
                "comparable": bool(comparable),
                "comparison_note": (
                    "동일 원본 CSV·요청 기준 비교입니다. "
                    "모델 버전은 결과에서 확인하세요."
                )
                if comparable
                else (
                    "단일 서버이거나 데이터 동일성/서버 응답을 "
                    "확인할 수 없어 동등 비교가 아닙니다."
                ),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
