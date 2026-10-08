"""완료된 14일 관측으로 다음 날짜 총발전량을 예측하는 Keras 후보."""

import numpy as np
from data.daily_features import FEATURE_COLUMNS, SEQ_LEN
from tensorflow import keras

N_FEATURES = len(FEATURE_COLUMNS)


def _baseline(inputs):
    last = keras.layers.Flatten()(keras.layers.Cropping1D((SEQ_LEN - 1, 0))(inputs))
    weights = np.zeros((N_FEATURES, 1))
    weights[0, 0] = 1
    return keras.layers.Dense(
        1,
        use_bias=False,
        trainable=False,
        kernel_initializer=keras.initializers.Constant(weights),
        name="previous_day_generation",
    )(last)


def build_daily_residual():
    """전날 발전량 + 과거 14일의 Ridge 보정. 저장 가능한 표준 레이어만 사용."""
    inputs = keras.layers.Input((SEQ_LEN, N_FEATURES))
    history = keras.layers.Flatten(name="daily_context")(inputs)
    adjustment = keras.layers.Dense(
        1, kernel_initializer="zeros", bias_initializer="zeros", name="daily_correction"
    )(history)
    model = keras.Model(
        inputs,
        keras.layers.Add()([_baseline(inputs), adjustment]),
        name="solar_daily_ridge",
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")
    return model


def build_model(architecture="daily_lstm"):
    if architecture not in {"daily_lstm", "persistence", "seasonal"}:
        raise ValueError("지원하지 않는 모델 구조입니다")
    inputs = keras.layers.Input((SEQ_LEN, N_FEATURES))
    history = keras.layers.LSTM(16)(inputs)
    history = keras.layers.Dense(8, activation="relu")(history)
    correction = keras.layers.Dense(1, kernel_initializer="zeros", name="correction")(
        history
    )
    model = keras.Model(
        inputs,
        keras.layers.Add(name="next_day_generation")([_baseline(inputs), correction]),
        name="solar_daily_lstm",
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model
