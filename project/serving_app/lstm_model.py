"""직전 발전량과 과거 같은 시간대의 증감을 활용하는 다음 한 시간 모델."""

from data.features import FEATURE_COLUMNS, SEQ_LEN
from tensorflow import keras

N_FEATURES = len(FEATURE_COLUMNS)


def build_model(architecture="seasonal") -> keras.Model:
    if architecture not in {"persistence", "seasonal"}:
        raise ValueError("지원하지 않는 모델 구조입니다")
    inputs = keras.layers.Input(shape=(SEQ_LEN, N_FEATURES))
    selection = [[float(name == "generation_mwh")] for name in FEATURE_COLUMNS]

    def generation(lag):
        cropped = keras.layers.Cropping1D((SEQ_LEN - lag, lag - 1))(inputs)
        return keras.layers.Dense(
            1,
            use_bias=False,
            trainable=False,
            kernel_initializer=keras.initializers.Constant(selection),
        )(keras.layers.Flatten()(cropped))

    last = generation(1)
    terms = [last]
    if architecture == "seasonal":
        # Target t follows the 72-point window. t-24 is window[-24],
        # so yesterday's same-hour ramp is window[-24] - window[-25].
        yesterday, yesterday_before = generation(24), generation(25)
        two_days, two_days_before = generation(48), generation(49)
        ramps = keras.layers.Concatenate(name="seasonal_ramps")(
            [
                keras.layers.Subtract()([yesterday, yesterday_before]),
                keras.layers.Subtract()([two_days, two_days_before]),
            ]
        )
        terms.append(
            keras.layers.Dense(
                1,
                use_bias=False,
                kernel_initializer="zeros",
                name="seasonal_adjustment",
            )(ramps)
        )
    history = keras.layers.LSTM(32)(inputs)
    history = keras.layers.Dense(16, activation="relu")(history)
    terms.append(
        keras.layers.Dense(
            1, kernel_initializer="zeros", bias_initializer="zeros", name="correction"
        )(history)
    )
    output = keras.layers.Add(name="next_hour_generation")(terms)
    model = keras.Model(inputs, output, name=f"solar_{architecture}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
