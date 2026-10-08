"""학습과 운영 관측에 공통으로 사용하는 예측 평가 지표."""

import math


def wape_pct(actual, predicted) -> float | None:
    """절대 오차 합 / 실제 발전량 절댓값 합 × 100. 분모 0은 미정의."""
    denominator = math.fsum(abs(float(value)) for value in actual)
    if denominator == 0:
        return None
    error = math.fsum(abs(float(p) - float(a)) for a, p in zip(actual, predicted))
    value = error / denominator * 100
    return value if math.isfinite(value) else None
