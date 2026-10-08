"""
운영 지표 수집 — 대시보드의 요청 수 / 평균 응답시간 / 성공률 카드용.

/predict* 요청만 인메모리 deque에 기록한다(서버 재시작 시 초기화). 대시보드는 최근 5분을
1분 단위 버킷으로 나눠 스파크라인을 그린다.
"""
import time
from collections import deque
from threading import Lock

WINDOW_SEC = 300  # 최근 5분
BUCKET_SEC = 60

_records: deque = deque(maxlen=5000)  # (timestamp, latency_ms, ok)
_lock = Lock()


def record(latency_ms: float, ok: bool) -> None:
    with _lock:
        _records.append((time.time(), latency_ms, ok))


def snapshot(now: float | None = None) -> dict:
    now = now or time.time()
    start = now - WINDOW_SEC
    with _lock:
        recent = [r for r in _records if r[0] >= start]

    n_buckets = WINDOW_SEC // BUCKET_SEC
    buckets = [{"count": 0, "lat_sum": 0.0, "ok": 0} for _ in range(n_buckets)]
    for ts, lat, ok in recent:
        i = min(int((ts - start) // BUCKET_SEC), n_buckets - 1)
        b = buckets[i]
        b["count"] += 1
        b["lat_sum"] += lat
        b["ok"] += int(ok)

    total = len(recent)
    return {
        "requests": total,
        "avg_latency_ms": round(sum(r[1] for r in recent) / total, 1) if total else None,
        "success_rate": round(100 * sum(r[2] for r in recent) / total, 1) if total else None,
        "series": {
            "requests": [b["count"] for b in buckets],
            "latency": [round(b["lat_sum"] / b["count"], 1) if b["count"] else 0 for b in buckets],
            "success": [round(100 * b["ok"] / b["count"], 1) if b["count"] else 0 for b in buckets],
        },
    }
