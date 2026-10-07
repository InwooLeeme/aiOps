"""학습 구간만 사용하여 전날 발전량 대비 시간대별 Ridge 보정을 적합한다."""

import numpy as np
from tensorflow import keras


def fit_daily_residual(model, scaler, train, *, alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha는 양수여야 합니다")
    if not len(train["y"]):
        raise ValueError("보정 모델 학습 데이터가 없습니다")
    extractor = keras.Model(
        model.input,
        [
            model.get_layer("hourly_context").output,
            model.get_layer("previous_day_generation").output,
        ],
    )
    features, baseline = extractor.predict(train["X"], batch_size=256, verbose=0)
    residual = np.asarray([scaler.scale_target(y) for y in train["y"]])
    residual -= baseline.reshape(-1)
    features = np.asarray(features, dtype="float64")
    if not np.isfinite(features).all() or not np.isfinite(residual).all():
        raise ValueError("보정 모델 학습값은 유한해야 합니다")
    coefficients = np.zeros(features.shape[1:])
    for hour in range(24):
        active = features[:, hour, -1] > 0.5
        if not active.any():
            continue  # No historical evidence at this hour: keep previous-day value.
        x = features[active, hour, :]
        penalty = np.eye(x.shape[1]) * alpha
        penalty[-1, -1] = 0  # Hour-specific intercept is not penalized.
        coefficients[hour] = np.linalg.solve(x.T @ x + penalty, x.T @ residual[active])
    model.get_layer("daily_correction").set_weights(
        [coefficients.reshape(-1, 1).astype("float32")]
    )
