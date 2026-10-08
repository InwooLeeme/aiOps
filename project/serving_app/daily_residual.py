"""학습 구간만 사용하여 일별 Ridge 보정을 적합한다."""

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
            model.get_layer("daily_context").output,
            model.get_layer("previous_day_generation").output,
        ],
    )
    features, baseline = extractor.predict(train["X"], batch_size=256, verbose=0)
    residual = np.asarray(
        [scaler.scale_target(y) for y in train["y"]]
    ) - baseline.reshape(-1)
    x = np.column_stack([features, np.ones(len(features))]).astype("float64")
    if not np.isfinite(x).all() or not np.isfinite(residual).all():
        raise ValueError("보정 모델 학습값은 유한해야 합니다")
    penalty = np.eye(x.shape[1]) * alpha
    penalty[-1, -1] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ residual)
    model.get_layer("daily_correction").set_weights(
        [weights[:-1, None].astype("float32"), weights[-1:].astype("float32")]
    )
