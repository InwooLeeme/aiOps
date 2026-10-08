"""
제주 태양광 일별 발전량 → 다음날 발전량 예측: 학습/검증/테스트 평가 스크립트.

  python scripts/train_solar_eval.py [csv경로]

· 타깃  : 다음날 이용률 cf = 발전량(MWh) / (설비용량(MW) × 24h)  (설비가 177→450MW로 늘어 MWh 그대로는 추세가 섞임)
          평가는 cf × 설비용량 × 24 로 MWh로 되돌려서 계산
· 분할  : 시간순  train 2019-2022 / val 2023 (early stopping·모델 선택) / test 2024 (최종 보고, 한 번만 사용)
· 입력  : 최근 SEQ_LEN일의 [cf, 기온, 습도, 풍속, 전운량, sin(doy), cos(doy)]
· 비교  : persistence(어제 값) / 7일 평균 / 계절 평균(train의 day-of-year 평균) / LSTM-history / LSTM+당일기상(예보 가정)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from tensorflow import keras  # noqa: E402

SEQ_LEN = 14
SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "data/uploads/jeju_solar_daily_2019_2024.csv"
WX = ["t_mean", "hum", "wind", "cloud"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["date", "sido", "gen", "gen_part", "obs_h", "gen_h", "status", "cap", "t_mean", "t_min", "t_max",
                  "hum", "wind", "cloud", "cap_h", "t_h", "hum_h", "wind_h", "cloud_h", "complete"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["cap"] = df["cap"].ffill().bfill()
    df["cf"] = df["gen"] / (df["cap"] * 24)  # 결측은 NaN 유지 → 타깃으로는 쓰지 않음
    doy = df.index.dayofyear.values
    df["sin"], df["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    # 입력용 보간: 짧은 결측만 선형보간, 긴 결측(2023-12)은 앞값 유지 후 타깃 쪽에서 제외
    for c in ["cf"] + WX:
        df[c + "_in"] = df[c].interpolate(limit=7, limit_direction="both").ffill().bfill()
    return df


def windows(df: pd.DataFrame, use_target_wx: bool):
    feats = ["cf_in"] + [c + "_in" for c in WX] + ["sin", "cos"]
    X, y, dates, caps, prev = [], [], [], [], []
    A = df[feats].values
    for i in range(SEQ_LEN, len(df)):
        tgt = df["cf"].iloc[i]
        if np.isnan(tgt) or df["cf"].iloc[i - SEQ_LEN:i].isna().sum() > 7:
            continue
        x = A[i - SEQ_LEN:i].copy()
        if use_target_wx:  # 예보를 안다고 가정: 타깃일의 기상(+계절)을 마지막 스텝 입력에 덧붙임
            row = A[i:i + 1].copy()
            row[0, 0] = A[i - 1, 0]  # cf 칸은 타깃 자체이므로 누수 방지: 어제 값으로 대체 (기상·계절만 당일 값)
            x = np.vstack([x, row])
        X.append(x); y.append(tgt); dates.append(df.index[i]); caps.append(df["cap"].iloc[i])
        prev.append(df["cf_in"].iloc[i - 1])
    return np.array(X, "float32"), np.array(y, "float32"), pd.DatetimeIndex(dates), np.array(caps), np.array(prev)


def metrics(cf_true, cf_pred, caps):
    t, p = cf_true * caps * 24, cf_pred * caps * 24  # MWh
    err = p - t
    ss_res, ss_tot = (err ** 2).sum(), ((t - t.mean()) ** 2).sum()
    return {
        "RMSE_MWh": float(np.sqrt((err ** 2).mean())),
        "MAE_MWh": float(np.abs(err).mean()),
        "nMAE_%cap": float(100 * (np.abs(err) / (caps * 24)).mean()),  # 설비용량 대비 평균 오차율
        "R2": float(1 - ss_res / ss_tot),
        "n": int(len(t)),
    }


def build(n_steps, n_feat):
    m = keras.Sequential([
        keras.layers.Input((n_steps, n_feat)),
        keras.layers.LSTM(32, return_sequences=True), keras.layers.LSTM(16),
        keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def fit_lstm(df, use_wx, tr, va):
    keras.utils.set_random_seed(SEED)
    X, y, d, c, _ = windows(df, use_wx)
    mu, sd = X[tr(d)].reshape(-1, X.shape[2]).mean(0), X[tr(d)].reshape(-1, X.shape[2]).std(0) + 1e-6  # train 통계만 사용
    Xn = (X - mu) / sd
    m = build(X.shape[1], X.shape[2])
    h = m.fit(Xn[tr(d)], y[tr(d)], validation_data=(Xn[va(d)], y[va(d)]), epochs=200, batch_size=32, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    pred = m.predict(Xn, verbose=0).flatten().clip(0, 1)
    return pred, d, y, c, len(h.history["loss"])


def main():
    df = load(CSV)
    tr = lambda d: d < "2023-01-01"
    va = lambda d: (d >= "2023-01-01") & (d < "2024-01-01")
    te = lambda d: d >= "2024-01-01"

    X, y, d, c, prev = windows(df, False)
    doy_mean = df.loc[df.index < "2023-01-01"].groupby(df.loc[df.index < "2023-01-01"].index.dayofyear)["cf"].mean()
    roll7 = df["cf_in"].rolling(7).mean().shift(1).reindex(d).values
    season = np.array([doy_mean.get(x.dayofyear, doy_mean.mean()) for x in d])
    preds = {"persistence(어제)": prev, "7일 평균": roll7, "계절 평균(train doy)": season}
    for name, use_wx in [("LSTM history", False), ("LSTM +당일기상(예보 가정)", True)]:
        p, dd, yy, cc, ep = fit_lstm(df, use_wx, tr, va)
        assert (dd == d).all()
        preds[name] = p
        print(f"[{name}] 학습 epoch={ep}, 기반 샘플 train={tr(d).sum()} val={va(d).sum()} test={te(d).sum()}")

    out = {}
    for split, f in [("train", tr), ("val(2023)", va), ("test(2024)", te)]:
        out[split] = {k: metrics(y[f(d)], np.nan_to_num(p[f(d)], nan=0.0), c[f(d)]) for k, p in preds.items()}
    # 테스트 구간 분기별 오차(드리프트 확인용): 최고 성능 LSTM 기준
    q = pd.Series(np.abs(preds["LSTM +당일기상(예보 가정)"] - y) * c * 24, index=d)
    out["test_quarterly_MAE_MWh"] = q[te(d)].groupby(q[te(d)].index.quarter).mean().round(1).to_dict()
    out["cf_by_year"] = df.groupby(df.index.year)["cf"].mean().round(3).to_dict()
    os.makedirs("reports", exist_ok=True)
    json.dump(out, open("reports/solar_eval.json", "w"), ensure_ascii=False, indent=2, default=str)

    for split in ["train", "val(2023)", "test(2024)"]:
        print(f"\n== {split} ==")
        print(pd.DataFrame(out[split]).T.round(3).to_string())
    print("\n테스트 분기별 MAE(MWh):", out["test_quarterly_MAE_MWh"])
    print("연도별 평균 이용률:", out["cf_by_year"])


if __name__ == "__main__":
    main()
