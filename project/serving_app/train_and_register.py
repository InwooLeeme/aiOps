"""시간순 분리, 검증 기준선 게이트, 동일 버전 아티팩트로 태양광 LSTM을 학습합니다."""

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from data.daily_features import (
    FEATURE_COLUMNS,
    SEQ_LEN,
    SolarScaler,
    load_rows,
    sample_windows,
)

from serving_app.config import (
    BASE_EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    MODEL_DIR,
    MODEL_NAME,
    PROJECT_ROOT,
    SEED,
)


def evaluate_predictions(y_true, y_pred, timestamps) -> dict:
    truth, prediction = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if not len(truth) or len(truth) != len(prediction) or len(truth) != len(timestamps):
        raise ValueError("평가 정답·예측·시간의 길이가 같고 비어 있지 않아야 합니다")
    errors = prediction - truth
    if not np.isfinite(errors).all():
        raise ValueError("평가에 유한하지 않은 발전량이 있습니다")
    monthly = {}
    for month in sorted({t[:7] for t in timestamps}):
        mask = np.asarray([t.startswith(month) for t in timestamps])
        monthly[month] = {
            "mae": float(np.mean(np.abs(errors[mask]))),
            "rmse": float(np.sqrt(np.mean(errors[mask] ** 2))),
            "n_samples": int(mask.sum()),
        }
    return {
        "monthly": monthly,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "n_samples": len(truth),
    }


def passes_gate(
    candidate: float, baselines: dict, incumbent: float | None = None
) -> bool:
    limits = list(baselines.values())
    if incumbent is not None:
        limits.append(incumbent)
    return bool(
        limits
        and all(math.isfinite(v) and v >= 0 for v in [candidate, *limits])
        and candidate < min(limits)
    )


def _partition(windows, scaler):
    # Transform each historical point once: adjacent 14-day windows share most rows.
    transformed = {}
    inputs, targets, timestamps, persistence, weekly_mean = [], [], [], [], []
    for window, target in windows:
        for point in window:
            key = point["timestamp"]
            if key not in transformed:
                transformed[key] = scaler.transform_point(point)
        inputs.append([transformed[p["timestamp"]] for p in window])
        targets.append(target["generation_mwh"])
        timestamps.append(target["timestamp"])
        persistence.append(window[-1]["generation_mwh"])
        weekly_mean.append(float(np.mean([r["generation_mwh"] for r in window[-7:]])))
    return {
        "X": np.asarray(inputs, dtype="float32"),
        "y": np.asarray(targets),
        "timestamps": timestamps,
        "persistence": persistence,
        "weekly_mean": weekly_mean,
    }


def prepare_training_data(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["timestamp"])
    fitting = [r for r in rows if r["timestamp"] < "2023-01-01"]
    if not fitting:
        raise ValueError("2023년 이전 학습 데이터가 필요합니다")
    scaler = SolarScaler().fit(fitting)
    groups = {"train": [], "validation": [], "test": []}
    for window, target in sample_windows(rows):
        timestamp = target["timestamp"]
        split = (
            "train"
            if timestamp < "2023-01-01"
            else "validation"
            if timestamp < "2024-01-01"
            else "test"
            if timestamp < "2025-01-01"
            else None
        )
        if split:
            groups[split].append((window, target))
    if any(not groups[key] for key in ("train", "validation", "test")):
        raise ValueError(
            "2023년 이전 학습·2023년 검증·2024년 테스트 시퀀스가 모두 필요합니다"
        )
    return {
        "scaler": scaler,
        **{name: _partition(group, scaler) for name, group in groups.items()},
    }


def split_selection_validation(partition):
    """2023년 전반은 조기 종료/후보 선택, 후반은 최종 게이트에만 사용한다."""
    masks = [
        np.asarray([t < "2023-07-01" for t in partition["timestamps"]]),
        np.asarray([t >= "2023-07-01" for t in partition["timestamps"]]),
    ]
    if not all(mask.any() for mask in masks):
        raise ValueError("2023년 전반 선택 구간과 후반 검증 구간이 모두 필요합니다")
    return tuple(
        {
            key: np.asarray(value)[mask].tolist()
            if key == "timestamps"
            else np.asarray(value)[mask]
            for key, value in partition.items()
        }
        for mask in masks
    )


def _predictions(model, partition, scaler):
    values = model.predict(partition["X"], batch_size=256, verbose=0).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("모델이 유한한 발전량을 반환하지 않았습니다")
    return [max(0.0, scaler.inverse_target(float(value))) for value in values]


def _evaluate(model, partition, scaler):
    args = (
        partition["y"],
        _predictions(model, partition, scaler),
        partition["timestamps"],
    )
    result = {"model": evaluate_predictions(*args)}
    for name in ("persistence", "weekly_mean"):
        result[name] = evaluate_predictions(
            partition["y"], partition[name], partition["timestamps"]
        )
    return result


def _fit(model, scaler, train, validation, epochs):
    from tensorflow import keras

    if epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다")
    model.fit(
        train["X"],
        np.asarray([scaler.scale_target(v) for v in train["y"]], dtype="float32"),
        validation_data=(
            validation["X"],
            np.asarray(
                [scaler.scale_target(v) for v in validation["y"]], dtype="float32"
            ),
        ),
        epochs=epochs,
        batch_size=128,
        shuffle=False,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
    )


def _metadata(train, validation, metrics, epochs, mode):
    return {
        "training_end": train["timestamps"][-1],
        "training_start": train["timestamps"][0],
        "validation_start": validation["timestamps"][0],
        "validation_end": validation["timestamps"][-1],
        "error_threshold_mwh": max(1e-6, 1.5 * metrics["validation"]["model"]["rmse"]),
        "drift_threshold_mwh": max(1e-6, 1.5 * metrics["validation"]["model"]["rmse"]),
        "validation_metrics": metrics["validation"]["model"],
        "gate_baseline_rmse": min(
            metrics["validation"][name]["rmse"]
            for name in ("persistence", "weekly_mean")
        ),
        "threshold_rule": "1.5 * validation RMSE; minimum 1e-6 MWh",
        "feature_columns": FEATURE_COLUMNS,
        "seq_len": SEQ_LEN,
        "unit": "MWh",
        "granularity": "daily",
        "target": "next_day_generation_mwh",
        "region": "제주",
        "timezone": "Asia/Seoul",
        "metrics": metrics,
        "seed": SEED,
        "epochs": epochs,
        "mode": mode,
    }


def save_bundle(model, scaler, metadata: dict, directory: Path) -> dict:
    from serving_app.model_loader import artifact_hash

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Metadata is committed last; hashes make interrupted or mixed bundles fail closed.
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        temp = Path(temporary)
        model.save(temp / "model.keras")
        scaler.save(temp / "scaler.json")
        metadata = {
            **metadata,
            "feature_columns": FEATURE_COLUMNS,
            "seq_len": SEQ_LEN,
            "unit": "MWh",
            "granularity": "daily",
            "target": "next_day_generation_mwh",
        }
        metadata["artifact_sha256"] = {
            name: artifact_hash(temp / name) for name in ("model.keras", "scaler.json")
        }
        if metadata.get("version") == "solar-local":
            metadata["version"] = (
                f"solar-local-{metadata['artifact_sha256']['model.keras'][:12]}"
            )
        (temp / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        for name in ("model.keras", "scaler.json", "metadata.json"):
            os.replace(temp / name, directory / name)
    return metadata


def select_daily_candidate(model, scaler, train, selection):
    """후보·규제 강도는 앞선 선택 구간으로만 결정한다."""
    from serving_app.daily_residual import fit_daily_residual
    from serving_app.lstm_model import build_daily_residual

    scores = {"lstm_or_fine_tune": _evaluate(model, selection, scaler)["model"]}
    selected, selected_alpha = "lstm_or_fine_tune", None
    for alpha in (0.01, 0.1, 1.0, 10.0):
        candidate = build_daily_residual()
        fit_daily_residual(candidate, scaler, train, alpha=alpha)
        name = f"daily_ridge_{alpha:g}"
        scores[name] = _evaluate(candidate, selection, scaler)["model"]
        if scores[name]["rmse"] < scores[selected]["rmse"]:
            model, selected, selected_alpha = candidate, name, alpha
    return model, selected, selected_alpha, scores


def _train(rows, epochs):
    from tensorflow import keras

    from serving_app.lstm_model import build_model

    keras.utils.set_random_seed(SEED)
    prepared = prepare_training_data(rows)
    selection, validation_partition = split_selection_validation(prepared["validation"])
    prepared["selection"] = selection
    prepared["validation"] = validation_partition
    model = build_model()
    _fit(model, prepared["scaler"], prepared["train"], selection, epochs)
    model, selected, alpha, candidates = select_daily_candidate(
        model, prepared["scaler"], prepared["train"], selection
    )
    validation = _evaluate(model, prepared["validation"], prepared["scaler"])
    promoted = passes_gate(
        validation["model"]["rmse"],
        {k: validation[k]["rmse"] for k in ("persistence", "weekly_mean")},
    )
    # Test is only reported after fitting and the validation-only gate decision.
    metrics = {
        "validation": validation,
        "test": _evaluate(model, prepared["test"], prepared["scaler"]),
    }
    metadata = _metadata(
        prepared["train"], prepared["validation"], metrics, epochs, "scratch"
    )
    metadata.update(
        selected_candidate=selected,
        residual_alpha=alpha,
        selection_candidates=candidates,
    )
    metadata["architecture"] = model.name
    metadata["selection_start"] = selection["timestamps"][0]
    metadata["selection_end"] = selection["timestamps"][-1]
    metadata["scaler_fit_end"] = max(
        r["timestamp"] for r in rows if r["timestamp"] < "2023-01-01"
    )
    metadata["gate_passed"] = promoted
    return model, prepared, metadata, promoted


def train_local(csv_path=None, epochs=BASE_EPOCHS, directory=None) -> dict:
    if csv_path is None:
        from data.storage import latest_upload

        csv_path = latest_upload()
    model, prepared, metadata, passed = _train(load_rows(csv_path), epochs)
    metadata["version"] = "solar-local"
    metadata = save_bundle(
        model, prepared["scaler"], metadata, directory or MODEL_DIR / "solar-daily"
    )
    return {
        "metrics": metadata["metrics"],
        "promoted": False,
        "gate_passed": passed,
        "version": metadata["version"],
        "metadata": metadata,
    }


def _log_and_register(
    model, scaler, metadata, passed, example, model_name=MODEL_NAME, *, promote=True
) -> dict:
    import mlflow
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient

    from serving_app.tracking import configure_experiment

    configure_experiment()
    with mlflow.start_run(run_name=f"solar-{metadata['mode']}") as run:
        mlflow.log_params(
            {key: metadata[key] for key in ("epochs", "seed", "seq_len", "mode")}
        )
        for key in (
            "selected_candidate",
            "residual_alpha",
            "validation_start",
            "validation_end",
            "parent_version",
        ):
            if metadata.get(key) is not None:
                mlflow.log_param(key, metadata[key])
        for name, scores in metadata.get("selection_candidates", {}).items():
            mlflow.log_metric(f"selection_{name}_rmse", scores["rmse"])
        mlflow.set_tags(
            {
                "simulation": str(metadata.get("simulation", False)).lower(),
                "target_model": model_name,
            }
        )
        for split, methods in metadata["metrics"].items():
            for method, scores in methods.items():
                for metric, value in scores.items():
                    if metric == "monthly":
                        for month, monthly_scores in value.items():
                            for key, number in monthly_scores.items():
                                mlflow.log_metric(
                                    f"{split}_{method}_{month}_{key}", number
                                )
                    elif value is not None:
                        mlflow.log_metric(f"{split}_{method}_{metric}", value)
        mlflow.log_metric("rmse", metadata["metrics"]["validation"]["model"]["rmse"])
        mlflow.log_metric("error_threshold_mwh", metadata["error_threshold_mwh"])
        mlflow.tensorflow.log_model(
            model,
            name="model",
            input_example=example,
            pip_requirements=str(PROJECT_ROOT / "requirements.txt"),
        )
        with tempfile.TemporaryDirectory() as temp:
            save_bundle(model, scaler, metadata, Path(temp))
            mlflow.log_artifacts(temp, "bundle")
        result = {
            "run_id": run.info.run_id,
            "metrics": metadata["metrics"],
            "rmse": metadata["metrics"]["validation"]["model"]["rmse"],
            "promoted": False,
            "gate_passed": bool(passed),
            "version": None,
            "validation_start": metadata.get("validation_start"),
            "validation_end": metadata.get("validation_end"),
            "parent_version": metadata.get("parent_version"),
        }
        if "selected_candidate" in metadata:
            result["selected_candidate"] = metadata["selected_candidate"]
        if passed:
            version = mlflow.register_model(
                f"runs:/{run.info.run_id}/model", model_name
            )
            if promote:
                MlflowClient().transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
            result.update(promoted=promote, version=str(version.version))
        return result


def train_and_register(
    csv_path: str | None = None,
    rows: list[dict] | None = None,
    epochs: int | None = None,
    *,
    promote: bool = True,
) -> dict:
    if rows is None:
        if csv_path is None:
            from data.storage import latest_upload

            csv_path = latest_upload()
        rows = load_rows(csv_path)
    model, prepared, metadata, passed = _train(
        rows, BASE_EPOCHS if epochs is None else epochs
    )
    return _log_and_register(
        model,
        prepared["scaler"],
        metadata,
        passed,
        prepared["train"]["X"][:1],
        promote=promote,
    )


def fine_tune(
    rows: list[dict],
    epochs: int = FINE_TUNE_EPOCHS,
    *,
    incumbent=None,
    model_name=MODEL_NAME,
    metadata_extra=None,
    promote=True,
) -> dict:
    """최근 365일 중 마지막 28일 선택, 마지막 14일 최종 게이트를 분리한다."""
    from tensorflow import keras

    from serving_app.model_loader import get_model

    ordered = sorted(rows, key=lambda r: r["timestamp"])
    if len(ordered) < 90:
        raise ValueError("재학습에는 최소 90일의 과거 관측치가 필요합니다")
    end = datetime.fromisoformat(ordered[-1]["timestamp"])
    if end + timedelta(days=1) > datetime.now(ZoneInfo("Asia/Seoul")).replace(
        tzinfo=None
    ):
        raise ValueError("미래 시각의 관측치로 재학습할 수 없습니다")
    earliest = (end - timedelta(days=365) + timedelta(days=1)).isoformat()
    ordered = [row for row in ordered if row["timestamp"] >= earliest]
    cutoff = (end - timedelta(days=14) + timedelta(days=1)).isoformat()
    incumbent = incumbent if incumbent is not None else get_model()
    selection_cutoff = (end - timedelta(days=42) + timedelta(days=1)).isoformat()
    windows = list(sample_windows(ordered))
    train = _partition(
        [(w, t) for w, t in windows if t["timestamp"] < selection_cutoff],
        incumbent.scaler,
    )
    selection = _partition(
        [(w, t) for w, t in windows if selection_cutoff <= t["timestamp"] < cutoff],
        incumbent.scaler,
    )
    validation = _partition(
        [(w, t) for w, t in windows if t["timestamp"] >= cutoff], incumbent.scaler
    )
    if not len(train["y"]) or not len(selection["y"]) or len(validation["y"]) != 14:
        raise ValueError(
            "재학습에는 완전한 학습 구간과 "
            "유효한 선택 구간·연속된 최종 검증 14일 구간이 필요합니다"
        )
    selection_end = max(
        incumbent.metadata.get("training_end", ""),
        incumbent.metadata.get("validation_end", ""),
    )
    if cutoff <= selection_end:
        raise ValueError("재학습 검증 구간은 기존 모델 학습·검증 종료 이후여야 합니다")
    keras.utils.set_random_seed(SEED)
    model = keras.models.clone_model(incumbent._keras_model)
    model.set_weights(incumbent._keras_model.get_weights())
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse"
    )
    _fit(model, incumbent.scaler, train, selection, epochs)
    model, selected_candidate, selected_alpha, selection_candidates = (
        select_daily_candidate(model, incumbent.scaler, train, selection)
    )
    scores = _evaluate(model, validation, incumbent.scaler)
    scores["incumbent"] = _evaluate(
        incumbent._keras_model, validation, incumbent.scaler
    )["model"]
    passed = passes_gate(
        scores["model"]["rmse"],
        {k: scores[k]["rmse"] for k in ("persistence", "weekly_mean")},
        scores["incumbent"]["rmse"],
    )
    metadata = _metadata(train, validation, {"validation": scores}, epochs, "fine-tune")
    metadata.update(
        scaler_fit_end=incumbent.metadata.get("scaler_fit_end"),
        architecture=model.name,
        selection_start=selection["timestamps"][0],
        selection_end=selection["timestamps"][-1],
        parent_version=incumbent.registry_version or incumbent.version,
        gate_passed=passed,
        selected_candidate=selected_candidate,
        residual_alpha=selected_alpha,
        selection_candidates=selection_candidates,
        candidate_selection_rule="minimum selection RMSE; final gate excluded",
    )
    metadata.update(metadata_extra or {})
    return _log_and_register(
        model,
        incumbent.scaler,
        metadata,
        passed,
        train["X"][:1],
        model_name=model_name,
        promote=promote,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="생략하면 최신 업로드 또는 제주 샘플 사용")
    parser.add_argument("--epochs", type=int, default=BASE_EPOCHS)
    args = parser.parse_args()
    print(
        json.dumps(
            train_and_register(args.csv, epochs=args.epochs),
            ensure_ascii=False,
            indent=2,
        )
    )
