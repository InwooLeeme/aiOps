"""
리플레이 시뮬레이션 — 실제 과거 데이터를 하루씩 "오늘 들어온 것처럼" 흘려보내며 AIOps 루프를 돌린다.

  하루마다: 예측(어제까지의 실적 + 오늘 예보) → 실적과 비교해 기록 → 드리프트 판단
            → 감지되면 최근 90일로 fine-tuning → 상대 게이트 → 통과 시 Production 교체

기본 구간 2023-01-01 ~ 2024-12-31: 2023은 정상 운영(오탐이 없어야 함), 2024는 실제 이용률 하락(드리프트).
백그라운드 스레드로 돌고, 진행 상황은 status() 로 조회한다.
비교용으로 시작 시점 모델을 고정한 shadow 모델도 같은 날을 예측한다(재학습이 없었다면의 대조군).
※ 리플레이는 "현재" Production 에서 시작한다. 처음 상태에서 다시 보려면 레지스트리를 초기화하고 train_and_register.py 를 다시 실행.
"""
import threading
import time

import numpy as np
import pandas as pd

from data.features import make_input, row_matrix
from serving_app import model_loader
from serving_app.dataset import get_table
from serving_app.monitoring import metrics
from serving_app.monitoring.drift_detector import WINDOW_SIZE, WAPE_THRESHOLD, compute_wape
from serving_app.monitoring.retrain_trigger import check_and_trigger

COOLDOWN_DAYS = 30  # 재학습 시도 후 이 기간은 다시 시도하지 않는다(게이트 실패 시 반복 방지)

window: list[dict] = []   # 현재 모델의 최근 예측 기록(재학습 승격 시 비운다) — 드리프트 판단 대상
_state = {"status": "idle"}
_lock = threading.Lock()


def window_snapshot() -> list[dict]:
    return list(window)


def status() -> dict:
    with _lock:
        return {**_state, "history": list(_state.get("history", [])), "events": list(_state.get("events", []))}


def start(start_date: str, end_date: str, delay: float = 0.0) -> bool:
    with _lock:
        if _state.get("status") == "running":
            return False
        _state.clear()
        _state.update(status="running", start=start_date, end=end_date, day=None, done=0, history=[], events=[])
    window.clear()
    threading.Thread(target=_run, args=(start_date, end_date, delay), daemon=True).start()
    return True


def _run(start_date: str, end_date: str, delay: float):
    try:
        df = get_table()
        A = row_matrix(df)
        days = [d for d in df.index if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]
        with _lock:
            _state["total"] = len(days)
        shadow = model_loader._load_model()  # 재학습 없이 운영했다면? — 시작 시점 Production 모델을 고정해 같은 날을 예측(비교용)
        cooldown_until = None
        for n, day in enumerate(days):
            i = df.index.get_loc(day)
            actual = df["cf"].iloc[i]
            model = model_loader.get_model()
            t0 = time.perf_counter()
            pred = model.predict_cf(make_input(A, i))
            metrics.record((time.perf_counter() - t0) * 1000, True)
            cap = float(df["cap"].iloc[i])
            shadow_pred = shadow.predict_cf(make_input(A, i))
            row = {"date": str(day.date()), "pred_mwh": round(pred * cap * 24, 1), "version": model.version,
                   "shadow_mwh": round(shadow_pred * cap * 24, 1), "shadow_version": shadow.version,
                   "actual_mwh": None if np.isnan(actual) else round(float(actual) * cap * 24, 1), "event": None}
            if not np.isnan(actual):
                window.append({"pred_mwh": row["pred_mwh"], "actual_mwh": row["actual_mwh"]})
                if cooldown_until is None or day >= cooldown_until:
                    out = check_and_trigger(window, df, day)
                    if out["status"] == "retrain_triggered":
                        cooldown_until = day + pd.Timedelta(days=COOLDOWN_DAYS)
                        promoted = bool(out.get("promoted"))
                        row["event"] = "promoted" if promoted else "gate_failed"
                        with _lock:
                            _state["events"].append({"date": row["date"], "promoted": promoted,
                                                     "prod_rmse": out.get("prod_rmse_mwh"), "new_rmse": out.get("rmse_mwh"),
                                                     "version": out.get("version")})
                        if promoted:
                            window.clear()  # 새 모델 기준으로 오차를 다시 쌓는다
            row["window_wape"] = round(compute_wape(window[-WINDOW_SIZE:]), 1) if len(window) >= WINDOW_SIZE else None
            row["threshold"] = WAPE_THRESHOLD
            with _lock:
                _state["history"].append(row)
                _state.update(day=row["date"], done=n + 1)
            if delay:
                time.sleep(delay)
        with _lock:
            _state["status"] = "done"
    except Exception as e:  # 화면에 원인을 보여주기 위해 상태에 남긴다
        with _lock:
            _state.update(status="error", error=f"{type(e).__name__}: {e}")
        raise
