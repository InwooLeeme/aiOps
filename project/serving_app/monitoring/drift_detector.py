"""최근 14일 예측 오차 감시. 성능 저하만으로 개념 드리프트를 단정하지 않는다."""

import math
from datetime import timedelta

from data.daily_features import parse_timestamp

from serving_app.evaluation import wape_pct

WINDOW_SIZE = 14


def compute_rmse(records: list[dict]) -> float:
    return (
        math.sqrt(
            sum((r["predicted"] - r["actual"]) ** 2 for r in records) / len(records)
        )
        if records
        else 0.0
    )


def assess_drift(records: list[dict], threshold: float | None) -> dict:
    unique = {
        r["timestamp"]: r
        for r in records
        if r.get("actual") is not None
        and math.isfinite(r["actual"])
        and math.isfinite(r["predicted"])
    }
    recent = sorted(unique.values(), key=lambda r: r["timestamp"])
    if recent:
        cutoff = parse_timestamp(recent[-1]["timestamp"]) - timedelta(days=WINDOW_SIZE)
        recent = [r for r in recent if parse_timestamp(r["timestamp"]) > cutoff]
    ready = len(recent) >= WINDOW_SIZE
    result = {
        "count": len(recent),
        "rmse": compute_rmse(recent) if recent else None,
        "wape_pct": wape_pct(
            [r["actual"] for r in recent], [r["predicted"] for r in recent]
        ),
        "threshold": threshold,
        "ready": ready,
        "unit": "MWh",
    }
    if threshold is None or not math.isfinite(threshold) or threshold <= 0:
        return {**result, "ready": False, "status": "threshold_unavailable"}
    if not ready:
        return {**result, "status": "insufficient_data"}
    return {
        **result,
        "status": "performance_degraded" if result["rmse"] > threshold else "ok",
    }
