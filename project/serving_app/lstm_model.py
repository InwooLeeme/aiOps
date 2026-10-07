"""직전 발전량과 과거 같은 시간대의 증감을 활용하는 다음 한 시간 모델."""

from data.features import FEATURE_COLUMNS, SEQ_LEN
from tensorflow import keras

N_FEATURES = len(FEATURE_COLUMNS)


def build_daily_residual() -> keras.Model:
    """전날 같은 시간 값 + 시간대별 Ridge 보정. 표준 Keras 레이어만 사용."""
    import numpy as np

    inputs = keras.layers.Input(shape=(SEQ_LEN, N_FEATURES))

    def point(lag):
        return keras.layers.Flatten()(
            keras.layers.Cropping1D((SEQ_LEN - lag, lag - 1))(inputs)
        )

    last, yesterday = point(1), point(24)
    generation = np.zeros((N_FEATURES, 1))
    generation[FEATURE_COLUMNS.index("generation_mwh"), 0] = 1
    baseline = keras.layers.Dense(
        1,
        use_bias=False,
        trainable=False,
        kernel_initializer=keras.initializers.Constant(generation),
        name="previous_day_generation",
    )(yesterday)
    context = keras.layers.Concatenate()(
        [
            last,
            yesterday,
            keras.layers.Subtract()([last, point(25)]),
            keras.layers.Subtract()([point(2), point(26)]),
        ]
    )
    constant = keras.layers.Dense(
        1, trainable=False, kernel_initializer="zeros", bias_initializer="ones"
    )(last)
    context = keras.layers.Concatenate()([context, constant])
    # Scaler fixes sin/cos ranges at [-1, 1]. At integer hours these frozen
    # cosine-distance gates are one-hot; target hour is last observed hour + 1.
    phase = 2 * np.pi * np.arange(24) / 24
    boundary = np.cos(2 * np.pi / 24)
    span = 1 - boundary
    weights = np.zeros((N_FEATURES, 24))
    weights[FEATURE_COLUMNS.index("hour_sin")] = 2 * np.sin(phase) / span
    weights[FEATURE_COLUMNS.index("hour_cos")] = 2 * np.cos(phase) / span
    hours = keras.layers.Dense(
        24,
        trainable=False,
        kernel_initializer=keras.initializers.Constant(weights),
        bias_initializer=keras.initializers.Constant(
            (-np.sin(phase) - np.cos(phase) - boundary) / span
        ),
    )(last)
    hours = keras.layers.ReLU(threshold=1e-4, name="observed_hour")(hours)
    features = keras.layers.Multiply(name="hourly_context")(
        [
            keras.layers.Reshape((24, 1))(hours),
            keras.layers.Reshape((1, 4 * N_FEATURES + 1))(context),
        ]
    )
    correction = keras.layers.Dense(
        1, use_bias=False, kernel_initializer="zeros", name="daily_correction"
    )(keras.layers.Flatten()(features))
    return keras.Model(
        inputs, keras.layers.Add()([baseline, correction]), name="solar_daily_residual"
    )


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
