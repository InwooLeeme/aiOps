"""발행 시각을 보존한 예측 기록과 이후 CSV 관측값의 연결."""

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta

from data.daily_features import KST, parse_timestamp

from serving_app import config
from serving_app.monitoring.drift_detector import assess_drift


def now_kst():
    return datetime.now(KST).replace(tzinfo=None)


def forecast_context(input_end, *, now=None):
    now = now or now_kst()
    end = parse_timestamp(input_end)
    target = end + timedelta(days=1)
    return {
        "input_end_timestamp": end.isoformat(),
        "target_timestamp": target.isoformat(),
        "forecast_context": "future_input"
        if target > now
        else "historical"
        if target + timedelta(hours=1) <= now
        else "current",
        "data_age_hours": max(0, (now - end).total_seconds() / 3600),
        "issued_at": now.replace(tzinfo=KST).isoformat(),
    }


def db_path(request):
    return getattr(
        request.app.state, "forecast_db_path", config.RUNTIME_DIR / "forecasts-daily.db"
    )


@contextmanager
def database(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS forecasts (
                prediction_id TEXT PRIMARY KEY, model_version TEXT NOT NULL,
                target_timestamp TEXT NOT NULL, input_end_timestamp TEXT NOT NULL,
                issued_at TEXT NOT NULL, predicted REAL NOT NULL,
                monitoring_eligible INTEGER NOT NULL, exclusion_reason TEXT,
                forecast_context TEXT NOT NULL, actual REAL,
                UNIQUE(model_version, target_timestamp))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS monitoring_runs (
                model_version TEXT NOT NULL, cutoff TEXT NOT NULL, result TEXT NOT NULL,
                PRIMARY KEY(model_version, cutoff))""")
            yield connection
    finally:
        connection.close()


def save_prediction(path, context, predicted, model):
    identity = str(model.registry_version or model.version)
    boundaries = [
        parse_timestamp(value)
        for key in ("training_end", "validation_end", "selection_end")
        if (value := model.metadata.get(key))
    ]
    reason = None
    if context["forecast_context"] != "current":
        reason = "historical"
    elif not boundaries:
        reason = "unknown_training_boundary"
    elif parse_timestamp(context["target_timestamp"]) <= max(boundaries):
        reason = "training_overlap"
    elif model.metadata.get("simulation"):
        reason = "simulation_model"
    identifier = uuid.uuid4().hex
    with database(path) as db:
        db.execute(
            "INSERT OR IGNORE INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                identifier,
                identity,
                context["target_timestamp"],
                context["input_end_timestamp"],
                context["issued_at"],
                predicted,
                reason is None,
                reason,
                context["forecast_context"],
            ),
        )
        record = dict(
            db.execute(
                "SELECT * FROM forecasts WHERE model_version=? AND target_timestamp=?",
                (identity, context["target_timestamp"]),
            ).fetchone()
        )
    return record


def monitoring_records(db, version):
    return [
        dict(r)
        for r in db.execute(
            "SELECT target_timestamp AS timestamp, predicted, actual, model_version "
            "FROM forecasts WHERE model_version=? AND monitoring_eligible=1 "
            "AND actual IS NOT NULL ORDER BY target_timestamp DESC LIMIT 14",
            (version,),
        )
    ][::-1]


def summary(path, model):
    version = str(model.registry_version or model.version) if model else None
    threshold = model.metadata.get("drift_threshold_mwh") if model else None
    with database(path) as db:
        records = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM forecasts ORDER BY issued_at DESC, rowid DESC LIMIT 30"
            )
        ]
        monitored = monitoring_records(db, version)
        last = db.execute(
            "SELECT result FROM monitoring_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        pending = db.execute(
            "SELECT COUNT(*) FROM forecasts WHERE model_version=? "
            "AND monitoring_eligible=1 AND actual IS NULL",
            (version,),
        ).fetchone()[0]
    for record in records:
        record["monitoring_eligible"] = bool(record["monitoring_eligible"])
        record["absolute_error_mwh"] = (
            abs(record["predicted"] - record["actual"])
            if record["actual"] is not None
            else None
        )
    return {
        "records": records,
        "monitoring": assess_drift(monitored, threshold),
        "model_version": version,
        "pending_count": pending,
        "last_evaluation": json.loads(last[0]) if last else None,
    }


def reconcile(path, rows):
    """관측은 기존 예측에만 연결하며, 같은 모델·시각의 중복 평가를 방지한다."""
    from serving_app import model_loader
    from serving_app.monitoring.retrain_trigger import check_and_trigger
    from serving_app.routers.predict import _replay_lock

    now = now_kst()
    matched = 0
    with database(path) as db:
        for row in rows:
            if (
                row.get("generation_mwh") is None
                or parse_timestamp(row["timestamp"]) + timedelta(days=1) > now
            ):
                continue
            matched += db.execute(
                "UPDATE forecasts SET actual=? "
                "WHERE target_timestamp=? AND actual IS NULL",
                (row["generation_mwh"], row["timestamp"]),
            ).rowcount
    if not _replay_lock.acquire(blocking=False):
        return {
            "matched": matched,
            "status": "busy",
            "message": "관측값은 저장했습니다. 작업 종료 후 다시 업로드하세요.",
        }
    try:
        model = model_loader._model_cache
        if model is None:
            return {"matched": matched, "status": "model_unavailable"}
        version = str(model.registry_version or model.version)
        with database(path) as db:
            db.execute("BEGIN IMMEDIATE")
            records = monitoring_records(db, version)
            check = assess_drift(records, model.metadata.get("drift_threshold_mwh"))
            if not check["ready"]:
                return {"matched": matched, "status": "waiting", "monitoring": check}
            cutoff = records[-1]["timestamp"]
            actuals = {r["timestamp"]: r.get("generation_mwh") for r in rows}
            if any(actuals.get(r["timestamp"]) != r["actual"] for r in records):
                return {
                    "matched": matched,
                    "status": "observation_mismatch",
                    "message": "평가 구간 전체의 원래 관측값이 필요합니다.",
                }
            existing = db.execute(
                "SELECT result FROM monitoring_runs WHERE model_version=? AND cutoff=?",
                (version, cutoff),
            ).fetchone()
            epoch = now.replace(tzinfo=KST).timestamp()
            if existing:
                previous = json.loads(existing[0])
                training = previous.get("check", {}).get("retraining") or {}
                retryable = training.get("status") in {
                    "blocked",
                    "failed",
                    "activation_failed",
                }
                expired = previous.get("status") == "running" and (
                    epoch - previous.get("started_at", 0) >= 600
                )
                if not (retryable or expired):
                    return {"matched": matched, "status": "already_evaluated"}
            db.execute(
                "INSERT OR REPLACE INTO monitoring_runs VALUES (?, ?, ?)",
                (
                    version,
                    cutoff,
                    json.dumps(
                        {
                            "status": "running",
                            "model_version": version,
                            "cutoff_timestamp": cutoff,
                            "started_at": epoch,
                        }
                    ),
                ),
            )
        try:
            check = check_and_trigger(
                records,
                model.metadata.get("drift_threshold_mwh"),
                rows=rows,
                incumbent=model,
            )
        except Exception:
            logging.getLogger(__name__).exception("운영 관측 평가 실패")
            check = {
                **check,
                "retraining": {
                    "status": "failed",
                    "promoted": False,
                    "reason": "평가 실패: 서버 로그 확인 후 다시 업로드하세요",
                },
            }
        result = {
            "model_version": version,
            "cutoff_timestamp": cutoff,
            "completed_at": now_kst().replace(tzinfo=KST).timestamp(),
            "check": check,
        }
        with database(path) as db:
            db.execute(
                "UPDATE monitoring_runs SET result=? "
                "WHERE model_version=? AND cutoff=?",
                (
                    json.dumps(result, ensure_ascii=False, allow_nan=False),
                    version,
                    cutoff,
                ),
            )
        return {"matched": matched, "status": "evaluated", **result}
    finally:
        _replay_lock.release()
