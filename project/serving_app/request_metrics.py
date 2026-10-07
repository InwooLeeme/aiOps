"""실제 예측 요청의 응답 코드와 경과 시간을 기록하고 기간별로 집계합니다."""

import json
import logging
import math
import threading
import time
from pathlib import Path

from fastapi import Request

from serving_app.config import LOG_DIR

WINDOWS = {"5m": 300, "1h": 3600, "6h": 21600, "24h": 86400}
_write_lock = threading.Lock()


def log_path(request: Request) -> Path:
    return Path(
        getattr(request.app.state, "request_log_path", LOG_DIR / "solar_requests.log")
    )


async def record_prediction_request(request: Request, call_next):
    if request.url.path not in {"/predict", "/simulation/run"}:
        return await call_next(request)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        record = {
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        try:
            path = log_path(request)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _write_lock, path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError:
            logging.getLogger(__name__).exception(
                "예측 요청 지표를 저장하지 못했습니다"
            )


def summarize(path: Path, window: str, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    start = now - WINDOWS[window]
    records = []
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    ts = float(record["timestamp"])
                    latency = float(record["latency_ms"])
                    status = int(record["status"])
                    if (
                        start <= ts <= now
                        and math.isfinite(latency)
                        and latency >= 0
                        and 100 <= status <= 599
                    ):
                        records.append((ts, latency, status))
                except (ValueError, KeyError, TypeError, OverflowError):
                    continue

    def aggregate(items):
        count = len(items)
        successes = sum(200 <= item[2] < 400 for item in items)
        return {
            "request_count": count,
            "avg_latency_ms": round(sum(item[1] for item in items) / count, 2)
            if count
            else 0,
            "success_rate": round(successes / count * 100, 2) if count else 0,
            "error_rate": (count - successes) / count if count else 0,
        }

    buckets = [[] for _ in range(12)]
    width = WINDOWS[window] / len(buckets)
    for record in records:
        buckets[min(int((record[0] - start) / width), 11)].append(record)
    return {
        "window": window,
        "generated_at": now,
        **aggregate(records),
        "series": [
            {"timestamp": start + i * width, **aggregate(bucket)}
            for i, bucket in enumerate(buckets)
        ],
    }
