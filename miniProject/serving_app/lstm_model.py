"""LSTM 구조 정의 — 학습·재학습이 공유한다. 입력 (N_STEPS, N_FEATURES) → 다음날 이용률(cf)."""
from tensorflow import keras

from data.features import N_FEATURES, N_STEPS


def build_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input((N_STEPS, N_FEATURES)),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(16),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model
