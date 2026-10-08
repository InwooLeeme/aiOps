"""
드리프트 감지 — 최근 WINDOW_SIZE일의 WAPE(%)가 고정 임계치 WAPE_THRESHOLD 를 넘으면 드리프트.

임계치 산정 규칙(결과를 보기 전에 고정): 검증 2022 의 21일 롤링 WAPE 최대값을 올림한 정수 → 31%.
  근거 (scripts/calibrate_drift.py, v1 고정 모델 = 학습 2019-2021, 재학습 없음):
  · 21일 롤링 WAPE  val 2022: 평균 17.4 / 최대 30.7   normal 2023: 평균 20.0 / 최대 37.5   drift 2024: 평균 30.1 / 최대 61.2
  · 31% 적용 시 초과 일 비율: val 0.0% / normal 2023 4.8%(초과 구간 2개 = 오탐 가능) / drift 2024 30.2%(초과 구간 12개)
  · 보정은 val(2022)로만 했고 2023(오탐 확인)·2024(감지 확인)는 확인용이다. 확인 결과를 보고 임계치를 고치지 않았다.
  · RMSE 는 정상/드리프트 분포가 겹치고, MAPE 는 흐린 날 실측이 작아 일별 오차가 수천 %까지 튀어 쓰지 않는다.
한계: val 은 모델 early stopping 에도 쓰여 오차가 약간 낙관적 → 정상 구간에서 오탐이 날 수 있다(2023 에서 실제로 확인).
      정상 구간이 한 해뿐이라 임계치의 통계적 신뢰도는 낮다. 2023-12 는 발전량이 전부 결측이라 겨울 확인이 약하다.

임계치는 모델 버전과 무관하게 고정한다. 승격 때마다 기준을 새 모델의 작은 검증값으로 갱신하면
기준이 계속 낮아져 재학습이 반복되기 때문이다.
"""
WINDOW_SIZE = 21         # 최근 21일
WAPE_THRESHOLD = 31.0    # %


def compute_wape(records: list[dict]) -> float:
    """records: [{"pred_mwh":…, "actual_mwh":…}, …]. 비어 있거나 실측 합이 0이면 0.0."""
    act = sum(r["actual_mwh"] for r in records)
    if not records or act <= 0:
        return 0.0
    return 100 * sum(abs(r["pred_mwh"] - r["actual_mwh"]) for r in records) / act


def compute_bias(records: list[dict]) -> float:
    """Bias(%) = Σ(예측-실측)/Σ실측×100. 양수=과대예측. 경보 시 방향을 함께 기록하는 데 쓴다(감지 기준은 아님)."""
    act = sum(r["actual_mwh"] for r in records)
    return 100 * sum(r["pred_mwh"] - r["actual_mwh"] for r in records) / act if records and act > 0 else 0.0


def is_drift(records: list[dict]) -> bool:
    if len(records) < WINDOW_SIZE:
        return False
    return compute_wape(records[-WINDOW_SIZE:]) > WAPE_THRESHOLD
