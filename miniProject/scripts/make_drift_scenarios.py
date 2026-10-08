"""
드리프트 시험용 CSV 생성 — 원본 일별 파일에서 '발전량' 열만 바꾸고 나머지 열은 그대로 둔다.

  python scripts/make_drift_scenarios.py

만드는 파일(data/drift_scenarios/):
  · drift_level_shift_from_2024-09-08.csv  2024-09-08부터 발전량 x0.5 가 계속 유지(수준 하락)
  · drift_volatility_from_2024-08-25.csv   2024-08-25부터 하루마다 무작위 배율(변동성 급증, 로그정규 sigma=0.6, 평균 보존, seed=42)
  · drift_alternating_from_2024-08-25.csv  2024-08-25부터 짝수일 x2.0 / 홀수일 x0.1 교차 (극단적·비현실적, 감지→재학습 경로 확인용)
형식은 원본과 같아서 어느 대시보드(우리 것, 팀원 것)에도 그대로 올릴 수 있다.
"""
import numpy as np
import pandas as pd

SRC = "data/uploads/jeju_solar_daily_2019_2024.csv"
GEN, PART = "일 발전량 합계(MWh)", "관측 발전량 부분합(MWh)"


def write(df, name):
    df.to_csv(f"data/drift_scenarios/{name}", index=False, encoding="utf-8")
    return name


df = pd.read_csv(SRC)
date = pd.to_datetime(df["날짜(KST)"])

# 1) 수준 하락: 2024-09-08 부터 x0.5
a = df.copy()
m = date >= "2024-09-08"
a.loc[m, [GEN, PART]] = a.loc[m, [GEN, PART]] * 0.5
print(write(a, "drift_level_shift_from_2024-09-08.csv"), "| 변경 행:", int(m.sum()))

# 2) 변동성 급증: 2024-08-25 부터 하루마다 무작위 배율 (평균이 1이 되도록 보정)
b = df.copy()
m = date >= "2024-08-25"
rng = np.random.default_rng(42)
sigma = 0.6
factor = rng.lognormal(mean=-sigma**2 / 2, sigma=sigma, size=int(m.sum()))
b.loc[m, GEN] = b.loc[m, GEN].values * factor
b.loc[m, PART] = b.loc[m, PART].values * factor
print(write(b, "drift_volatility_from_2024-08-25.csv"), "| 변경 행:", int(m.sum()), "| 배율 범위:", round(factor.min(), 2), "-", round(factor.max(), 2))

# 3) 극단 교차(경로 확인용, 비현실적): 2024-08-25 부터 날짜 번호가 짝수인 날 x2.0, 홀수인 날 x0.1
c = df.copy()
m = date >= "2024-08-25"
f = np.where(date[m].map(pd.Timestamp.toordinal) % 2 == 0, 2.0, 0.1)
c.loc[m, GEN] = c.loc[m, GEN].values * f
c.loc[m, PART] = c.loc[m, PART].values * f
print(write(c, "drift_alternating_from_2024-08-25.csv"), "| 변경 행:", int(m.sum()))

# 검증: 발전량 외 열은 원본과 동일한지
for name in ["drift_level_shift_from_2024-09-08.csv", "drift_volatility_from_2024-08-25.csv", "drift_alternating_from_2024-08-25.csv"]:
    x = pd.read_csv(f"data/drift_scenarios/{name}")
    others = [c for c in df.columns if c not in (GEN, PART)]
    same = x[others].equals(df[others])
    changed = int(((x[GEN] - df[GEN]).abs() > 1e-6).sum())
    print(f"검증 {name}: 발전량 외 열 동일={same}, 발전량이 바뀐 행={changed}, 전체 행={len(x)}")
