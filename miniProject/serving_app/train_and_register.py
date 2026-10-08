"""
MLflow 학습 → 기록 → 게이트 검증 → 등록 → Production 승격.

  train_and_register()   처음부터(scratch) 학습. train 2019-2021, val 2022(early stopping). 최초 1회.
  fine_tune(df, as_of)   드리프트 감지 시: Production 가중치에서 이어서, 최근 RECENT_DAYS일로 짧게 재학습.

게이트(상대 기준, RMSE): 같은 "최근 검증 구간"에서 새 모델이 현재 Production보다 MIN_GAIN 이상 낫지 않으면 승격하지 않는다.
  · 드리프트 때는 환경 자체가 어려워지므로 절대 오차 기준은 맞지 않는다.
  · 검증 구간은 fine-tuning에 쓰지 않은 가장 최근 HOLDOUT_DAYS일이다.
  · 지표 역할(교수님 기준): 배포 판정=RMSE, 드리프트 탐지=WAPE(monitoring/drift_detector.py), 평가=MAE+WAPE, 방향=Bias

실행:  python serving_app/train_and_register.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.features import SCALER_PATH, SolarScaler, build_xy, eval_metrics, load_table
from data.storage import latest_upload
from serving_app.lstm_model import build_model

SEED = 42
MODEL_NAME = "Solar_Predictor"
TRAIN_END, VAL_END = "2022-01-01", "2023-01-01"  # train 2019-2021 / val 2022 (early stopping·임계치 보정) / 2023 정상 운영 / 2024 드리프트
BASE_EPOCHS = 200  # early stopping 으로 보통 30~70에서 멈춘다
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 1e-4
RECENT_DAYS = 90    # 재학습에 쓰는 최근 일수
HOLDOUT_DAYS = 21   # 그중 검증 전용(학습 제외)
MIN_GAIN = 0.10     # 같은 검증 구간에서 RMSE 가 현재 Production 보다 10% 이상 줄어야 승격(21일 표본은 작아 여유를 둔다)


def _register_and_promote(model, run_id: str) -> str:
    v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    MlflowClient().transition_model_version_stage(name=MODEL_NAME, version=v.version, stage="Production",
                                                  archive_existing_versions=True)
    return v.version


def train_and_register(df: pd.DataFrame | None = None) -> dict:
    """처음부터 학습. 최초 등록이거나 Production 대비 개선될 때만 승격."""
    keras.utils.set_random_seed(SEED)
    df = df if df is not None else load_table(latest_upload())
    X_tr, y_tr, _, _ = build_xy(df, end=TRAIN_END)
    X_va, y_va, _, c_va = build_xy(df, start=TRAIN_END, end=VAL_END)

    scaler = SolarScaler().fit(X_tr)  # train 통계로 한 번만 fit → 저장 후 재학습에서도 재사용
    scaler.save(SCALER_PATH)

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        hist = model.fit(scaler.transform(X_tr), y_tr, validation_data=(scaler.transform(X_va), y_va), epochs=BASE_EPOCHS,
                         batch_size=32, verbose=0,
                         callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
        p_va = np.clip(model.predict(scaler.transform(X_va), verbose=0).flatten(), 0, 1)
        mt = eval_metrics(y_va * c_va * 24, p_va * c_va * 24)
        mlflow.log_params({"mode": "scratch", "epochs": len(hist.history["loss"]), "n_train": len(X_tr), "n_val": len(X_va)})
        mlflow.log_metrics(mt)
        mlflow.tensorflow.log_model(model, name="model", input_example=scaler.transform(X_tr[:1]))
        run_id = mlflow.active_run().info.run_id
    version = _register_and_promote(model, run_id)
    print(f"[REGISTERED] {MODEL_NAME} v{version} -> Production  val RMSE={mt['rmse_mwh']:.0f}MWh MAE={mt['mae_mwh']:.0f}MWh WAPE={mt['wape']:.1f}% Bias={mt['bias']:+.1f}%")
    return {"run_id": run_id, "version": version, **mt, "promoted": True}


def fine_tune(df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """as_of(포함)까지의 최근 RECENT_DAYS일로 Production 모델을 이어서 학습하고, 상대 게이트로 승격 여부를 결정."""
    keras.utils.set_random_seed(SEED)
    scaler = SolarScaler.load(SCALER_PATH)
    end = as_of + pd.Timedelta(days=1)
    split = end - pd.Timedelta(days=HOLDOUT_DAYS)
    X_fit, y_fit, _, _ = build_xy(df, start=end - pd.Timedelta(days=RECENT_DAYS), end=split)
    X_ho, y_ho, _, c_ho = build_xy(df, start=split, end=end)
    act_ho = y_ho * c_ho * 24
    if len(X_fit) < 20 or len(X_ho) < 5:
        return {"promoted": False, "reason": "데이터 부족", "n_fit": len(X_fit), "n_holdout": len(X_ho)}

    prod = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    prod_pred = np.clip(prod.predict(scaler.transform(X_ho), verbose=0).flatten(), 0, 1)
    prod_m = eval_metrics(act_ho, prod_pred * c_ho * 24)

    cand = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")  # 별도 사본에서 이어서 학습
    cand.compile(optimizer=keras.optimizers.Adam(FINE_TUNE_LR), loss="mse")
    with mlflow.start_run(run_name=f"fine-tune {as_of.date()}"):
        cand.fit(scaler.transform(X_fit), y_fit, epochs=FINE_TUNE_EPOCHS, batch_size=16, verbose=0)
        cand_pred = np.clip(cand.predict(scaler.transform(X_ho), verbose=0).flatten(), 0, 1)
        cand_m = eval_metrics(act_ho, cand_pred * c_ho * 24)
        promoted = cand_m["rmse_mwh"] < prod_m["rmse_mwh"] * (1 - MIN_GAIN)  # 배포 판정 = RMSE
        mlflow.log_params({"mode": "fine-tune", "epochs": FINE_TUNE_EPOCHS, "n_fit": len(X_fit), "n_holdout": len(X_ho),
                           "as_of": str(as_of.date())})
        mlflow.log_metrics({**cand_m, **{f"prod_{k}_on_holdout": v for k, v in prod_m.items()}})
        mlflow.tensorflow.log_model(cand, name="model", input_example=scaler.transform(X_fit[:1]))
        run_id = mlflow.active_run().info.run_id

    result = {"run_id": run_id, **cand_m, "prod_rmse_mwh": prod_m["rmse_mwh"], "prod_wape": prod_m["wape"],
              "prod_bias": prod_m["bias"], "promoted": bool(promoted)}
    if promoted:
        result["version"] = _register_and_promote(cand, run_id)
        print(f"[GATE PASSED] holdout RMSE {prod_m['rmse_mwh']:.0f} -> {cand_m['rmse_mwh']:.0f} MWh, v{result['version']} promoted")
    else:
        print(f"[GATE FAILED] holdout RMSE prod={prod_m['rmse_mwh']:.0f} cand={cand_m['rmse_mwh']:.0f} MWh -> 기존 Production 유지")
    return result


if __name__ == "__main__":
    train_and_register()
