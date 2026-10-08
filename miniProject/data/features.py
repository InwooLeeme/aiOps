"""
제주 태양광 일별 발전량 → 다음날 이용률 예측용 데이터/피처 모듈 (학습·서빙·재학습 공용).

· 한 행(day) = 발전 실적(발전량, 설비용량) + 그날의 "예보" 기상(기온·습도·풍속·전운량·일사량)
· 타깃     = 이용률 cf = 발전량(MWh) / (설비용량(MW) × 24)   → 설비가 늘어도 비교 가능
· 모델 입력 = 과거 SEQ_LEN일 행 + 예측 대상일 행(발전 cf 칸은 어제 값으로 대체: 타깃 누수 방지)
· 스케일러  = train 기간 통계로 한 번만 fit 해 저장, 재학습에서도 그대로 재사용
"""
import os
import pickle

import numpy as np
import pandas as pd

SEQ_LEN = 14
UPLOAD_COLS = {"date": "날짜(KST)", "gen": "일 발전량 합계(MWh)", "cap": "설비용량 평균(MW)"}
WEATHER_CSV = "data/weather_jeju_forecast.csv"
WX = ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_mean", "cloud_cover_mean",
      "shortwave_radiation_sum"]
FEATURES = ["cf"] + WX + ["sin", "cos"]
N_FEATURES = len(FEATURES)
N_STEPS = SEQ_LEN + 1  # 과거 SEQ_LEN일 + 예측 대상일
SCALER_PATH = "serving_app/models/scaler.pkl"


def load_table(obs_csv: str, weather_csv: str = WEATHER_CSV) -> pd.DataFrame:
    """발전 실적 CSV + 예보 CSV → date 인덱스의 일별 테이블(cf, 예보 기상, 계절 sin/cos, cap)."""
    obs = pd.read_csv(obs_csv, usecols=list(UPLOAD_COLS.values())).rename(columns={v: k for k, v in UPLOAD_COLS.items()})
    obs["date"] = pd.to_datetime(obs["date"])
    wx = pd.read_csv(weather_csv)
    wx["date"] = pd.to_datetime(wx["date"])
    df = obs.merge(wx, on="date", how="left").set_index("date").sort_index()
    df["cap"] = df["cap"].ffill().bfill()
    df["cf"] = df["gen"] / (df["cap"] * 24)  # 발전량 결측은 NaN 유지(타깃으로 쓰지 않음)
    df[WX] = df[WX].interpolate(limit=3, limit_direction="both")
    doy = df.index.dayofyear.values
    df["sin"], df["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["cf_in"] = df["cf"].interpolate(limit=7, limit_direction="both").ffill().bfill()  # 입력용(긴 결측은 앞값)
    return df


def row_matrix(df: pd.DataFrame) -> np.ndarray:
    """테이블 → (일수, N_FEATURES). cf 칸은 입력용 보간값(cf_in)을 쓴다."""
    return df.assign(cf=df["cf_in"])[FEATURES].values.astype("float32")


def make_input(A: np.ndarray, i: int) -> np.ndarray:
    """i번째 날을 예측하기 위한 (N_STEPS, N_FEATURES) 입력. A는 row_matrix 결과."""
    target = A[i : i + 1].copy()
    target[0, 0] = A[i - 1, 0]  # 타깃 자체이므로 누수 방지: 어제 cf 로 대체 (기상·계절만 당일 값)
    return np.vstack([A[i - SEQ_LEN : i], target])


def build_xy(df: pd.DataFrame, start=None, end=None):
    """[start, end) 기간의 (X, y_cf, dates, cap) — 타깃이 결측이거나 입력 cf 결측이 7일 초과인 창은 제외."""
    A = row_matrix(df)
    X, y, d, c = [], [], [], []
    for i in range(SEQ_LEN, len(df)):
        day = df.index[i]
        if (start is not None and day < pd.Timestamp(start)) or (end is not None and day >= pd.Timestamp(end)):
            continue
        if np.isnan(df["cf"].iloc[i]) or df["cf"].iloc[i - SEQ_LEN : i].isna().sum() > 7 or np.isnan(A[i]).any():
            continue
        X.append(make_input(A, i)); y.append(df["cf"].iloc[i]); d.append(day); c.append(df["cap"].iloc[i])
    return np.array(X, "float32"), np.array(y, "float32"), pd.DatetimeIndex(d), np.array(c)


class SolarScaler:
    """피처별 표준화(평균 0, 표준편차 1). train 기간 통계로 한 번 fit 하고 이후 고정."""

    def __init__(self):
        self.mean = self.std = None

    def fit(self, X: np.ndarray) -> "SolarScaler":
        flat = X.reshape(-1, X.shape[-1])
        self.mean, self.std = flat.mean(0), flat.std(0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype("float32")

    def save(self, path: str = SCALER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = SCALER_PATH) -> "SolarScaler":
        s = cls()
        with open(path, "rb") as f:
            s.__dict__.update(pickle.load(f))
        return s


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def wape(act_mwh, pred_mwh) -> float:
    """WAPE(%) = Σ|예측-실측| / Σ실측 × 100. 실측이 0에 가까운 날이 있어도 MAPE처럼 폭주하지 않는다."""
    act, pred = np.asarray(act_mwh, dtype=float), np.asarray(pred_mwh, dtype=float)
    return float(100 * np.abs(pred - act).sum() / act.sum())


def eval_metrics(act_mwh, pred_mwh) -> dict:
    """지표 4종(MWh 기준). 상황별 권장 지표는 README '지표별 역할' 참고.
    · rmse_mwh : 모델 선정·배포 판정 (큰 오차 억제, 동일 데이터 비교)
    · wape     : 드리프트 탐지·재학습 트리거 (물량 변동과 무관한 비율)
    · mae_mwh  : 물량 변동 구간 포함 평가 (WAPE 와 함께, 특정 구간이 수치를 지배하지 않음)
    · bias     : 과대/과소 치우침(%). Σ(예측-실측)/Σ실측×100, 양수=과대예측
    """
    act, pred = np.asarray(act_mwh, dtype=float), np.asarray(pred_mwh, dtype=float)
    err = pred - act
    return {"rmse_mwh": float(np.sqrt((err ** 2).mean())), "mae_mwh": float(np.abs(err).mean()),
            "wape": float(100 * np.abs(err).sum() / act.sum()), "bias": float(100 * err.sum() / act.sum())}
