"""직전 시간 발전량에 LSTM 보정량을 더하는 다음 한 시간 예측 모델."""

from data.features import FEATURE_COLUMNS, SEQ_LEN
from tensorflow import keras

N_FEATURES = len(FEATURE_COLUMNS)


def build_model() -> keras.Model:
    inputs = keras.layers.Input(shape=(SEQ_LEN, N_FEATURES))
    last_hour = keras.layers.Cropping1D((SEQ_LEN - 1, 0))(inputs)
    last_features = keras.layers.Flatten()(last_hour)
    # Frozen selection weights retain the exact persistence baseline. Built-in
    # layers also keep the saved model safe-mode loadable without custom code.
    selection = [[float(name == "generation_mwh")] for name in FEATURE_COLUMNS]
    persistence = keras.layers.Dense(
        1,
        use_bias=False,
        trainable=False,
        kernel_initializer=keras.initializers.Constant(selection),
        name="last_generation",
    )(last_features)
    history = keras.layers.LSTM(32)(inputs)
    history = keras.layers.Dense(16, activation="relu")(history)
    residual = keras.layers.Dense(
        1, kernel_initializer="zeros", bias_initializer="zeros", name="correction"
    )(history)
    output = keras.layers.Add(name="next_hour_generation")([persistence, residual])
    model = keras.Model(inputs, output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
