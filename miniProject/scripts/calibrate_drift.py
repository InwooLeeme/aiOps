"""
드리프트 임계치 보정 분석 — 재학습 없이 고정한 모델(기본 v1)로 2022~2024 일별 오차를 계산한다.

  python scripts/calibrate_drift.py [model_version]

임계치 규칙(결과를 보기 전에 고정):  임계치 = 검증 2022 의 21일 롤링 WAPE 최대값을 올림한 정수(%)
  · 보정은 검증(2022)로만 한다. 정상 운영(2023)·드리프트(2024)는 정한 임계치를 "확인"하는 데만 쓴다.
  · 확인 결과를 보고 임계치를 다시 조정하면 순환이 되므로 하지 않는다.
참고 출력: MAPE vs WAPE 안정성, 구간별 21일 롤링 WAPE/RMSE 분포.
"""
import math
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow.tensorflow
import numpy as np
import pandas as pd

from data.features import SolarScaler, build_xy, load_table
from data.storage import latest_upload

VERSION = sys.argv[1] if len(sys.argv) > 1 else "1"
W = 21

df = load_table(latest_upload())
scaler = SolarScaler.load()
X, y, d, cap = build_xy(df, start="2022-01-01", end="2025-01-01")
model = mlflow.tensorflow.load_model(f"models:/Solar_Predictor/{VERSION}")
p = np.clip(model.predict(scaler.transform(X), verbose=0).flatten(), 0, 1)

r = pd.DataFrame({"cf": y, "pcf": p, "cap": cap}, index=d)
r["act"], r["pred"] = r.cf * r.cap * 24, r.pcf * r.cap * 24
r["err"] = r.pred - r.act
r["sq_cf"] = (r.pcf - r.cf) ** 2
r["ape"] = r.err.abs() / r.act * 100
yrs = {"val 2022": "2022", "normal 2023": "2023", "drift 2024": "2024"}

print("== (1) MAPE vs WAPE (고정 모델 v%s) ==" % VERSION)
for name, yr in yrs.items():
    s = r.loc[yr]
    print(f"{name}: MAPE={s.ape.mean():.1f}% (일별 APE 최대 {s.ape.max():.0f}%, 100% 초과 {int((s.ape > 100).sum())}일) | WAPE={100 * s.err.abs().sum() / s.act.sum():.1f}%"
          f" | RMSE={np.sqrt((s.err ** 2).mean()):.0f} MWh (cf {100 * np.sqrt(s.sq_cf.mean()):.2f}%p)")

# 21일 롤링. 창이 연도 경계를 넘지 않도록 연도별로 계산한다.
def rolling(s: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"rmse_cf": np.sqrt(s.sq_cf.rolling(W).mean()) * 100,
                         "wape": 100 * s.err.abs().rolling(W).sum() / s.act.rolling(W).sum()}).dropna()

roll = {name: rolling(r.loc[yr]) for name, yr in yrs.items()}
print("\n== (2) 21일 롤링 오차 분포 ==")
for col in ["wape", "rmse_cf"]:
    for name, v in roll.items():
        a = v[col]
        print(f"[{col:7s}] {name:12s} 평균 {a.mean():6.2f} / p90 {a.quantile(.9):6.2f} / p99 {a.quantile(.99):6.2f} / 최대 {a.max():6.2f}")

thr = math.ceil(roll["val 2022"]["wape"].max())
print(f"\n== (3) 임계치(규칙: 검증 2022 롤링 WAPE 최대값 올림) = {thr}% ==")
ep = lambda s: int(((s > thr) & ~(s > thr).shift(fill_value=False)).sum())  # 초과 '구간'(연속) 수
for name in ["val 2022", "normal 2023", "drift 2024"]:
    a = roll[name]["wape"]
    print(f"{name:12s} 임계치 초과 일 {100 * (a > thr).mean():5.1f}% ({int((a > thr).sum())}일) / 초과 구간 {ep(a)}개")
print("→ normal 2023 의 초과 구간 수가 '오탐'(재학습이 걸렸을 횟수의 상한), drift 2024 가 감지.")
