"""
드리프트 → 자동 재학습 → 상대 게이트 → 승격. 전 과정을 "aiops" 로거에 남긴다(대시보드 알람이 이 로그를 읽는다).

  [WARN] drift detected ...        드리프트 감지
  [INFO] retrain triggered ...     재학습 시작
  [OK]   ... production promoted   게이트 통과, 새 Production
  [WARN] ... gate failed           새 모델이 더 낫지 않아 기존 Production 유지(서비스는 멈추지 않음)
※ 대시보드가 문구를 읽으므로 바꾸지 말 것.
"""
import logging

import pandas as pd

from serving_app import model_loader
from serving_app.monitoring.drift_detector import WAPE_THRESHOLD, WINDOW_SIZE, compute_bias, compute_wape, is_drift

logger = logging.getLogger("aiops")


def check_and_trigger(records: list[dict], df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """
    records: 최근 예측 기록, df: 일별 테이블, as_of: 현재 시점(이 날까지의 실적을 알고 있음)
    반환: {"status": "ok"} 또는 {"status": "retrain_triggered", "promoted": bool, ...}
    """
    if not is_drift(records):
        return {"status": "ok"}

    cur, bias = compute_wape(records[-WINDOW_SIZE:]), compute_bias(records[-WINDOW_SIZE:])
    direction = "과대예측" if bias > 0 else "과소예측"
    logger.warning(f"[WARN] drift detected - triggering retrain ({as_of.date()}, "
                   f"{WINDOW_SIZE}d WAPE {cur:.1f}% > threshold {WAPE_THRESHOLD:.0f}%, Bias {bias:+.1f}% {direction})")

    from serving_app.train_and_register import RECENT_DAYS, fine_tune

    logger.info(f"[INFO] retrain triggered (window=last_{RECENT_DAYS}_days)")
    result = fine_tune(df, as_of)

    if result.get("promoted"):
        new = model_loader.reload()
        logger.info(f"[OK] new_rmse={result['rmse_mwh']:.0f}MWh (prev {result['prod_rmse_mwh']:.0f}MWh) "
                    f"- production promoted: Solar_Predictor {new.version}")
    else:
        why = result.get("reason") or f"cand RMSE {result['rmse_mwh']:.0f}MWh vs prod {result['prod_rmse_mwh']:.0f}MWh"
        logger.warning(f"[WARN] gate failed ({why}) - keeping current Production")
    return {"status": "retrain_triggered", **{k: v for k, v in result.items() if k != "run_id"}}
