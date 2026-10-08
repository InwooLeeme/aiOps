"""제주 일별 총발전량 전처리. 날짜는 KST 00:00, 입력은 완료된 14일."""

import csv
import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEQ_LEN = 14
COLUMN_MAP = {
    "시도명": "region",
    "일자 및 시각": "timestamp",
    "태양광 발전량 합계(MWh)": "generation_mwh",
    "태양광설비용량(MW)": "capacity_mw",
    "기온": "temperature",
    "습도": "humidity",
    "풍속": "wind_speed",
    "전운량(10분위)": "cloud_cover",
}
NUMERIC_COLUMNS = [
    "generation_mwh",
    "capacity_mw",
    "temperature",
    "humidity",
    "wind_speed",
    "cloud_cover",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + ["year_sin", "year_cos"]
KST = timezone(timedelta(hours=9))


def parse_timestamp(value) -> datetime:
    t = datetime.fromisoformat(str(value))
    if t.tzinfo:
        t = t.astimezone(KST).replace(tzinfo=None)
    if t.hour or t.minute or t.second or t.microsecond:
        raise ValueError("일자 및 시각은 일자(KST 00:00)여야 합니다")
    return t


def validate_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("CSV에 데이터가 없습니다")
    result = []
    seen = set()
    for i, source in enumerate(rows, 2):
        row = dict(source)
        try:
            row["timestamp"] = parse_timestamp(row["timestamp"]).isoformat()
            if row["timestamp"] in seen:
                raise ValueError("중복 시각")
            seen.add(row["timestamp"])
            if row.get("region") not in ("제주", "제주특별자치도"):
                raise ValueError("현재 모델은 제주 단일 지역만 지원합니다")
            row["region"] = "제주"
            for col in NUMERIC_COLUMNS:
                raw = row.get(col)
                value = None if raw is None or str(raw).strip() == "" else float(raw)
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"{col}: 유한한 숫자가 아닙니다")
                row[col] = value
            if row["capacity_mw"] is not None and row["capacity_mw"] <= 0:
                raise ValueError("설비용량은 양수여야 합니다")
            for col in ("generation_mwh", "wind_speed", "humidity", "cloud_cover"):
                if row[col] is not None and row[col] < 0:
                    raise ValueError(f"{col}: 음수는 허용하지 않습니다")
            if row["humidity"] is not None and row["humidity"] > 100:
                raise ValueError("습도는 100 이하여야 합니다")
            if row["cloud_cover"] is not None and row["cloud_cover"] > 10:
                raise ValueError("전운량은 10 이하여야 합니다")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{i}행 데이터 오류: {exc}") from exc
        result.append(row)
    return sorted(result, key=lambda r: r["timestamp"])


DAILY_COLUMN_MAP = {
    "날짜(KST)": "timestamp",
    "시도명": "region",
    "일 발전량 합계(MWh)": "generation_mwh",
    "설비용량 평균(MW)": "capacity_mw",
    "기온 평균": "temperature",
    "습도 평균": "humidity",
    "풍속 평균": "wind_speed",
    "전운량 평균(10분위)": "cloud_cover",
}
COUNT_COLUMNS = dict(
    zip(
        NUMERIC_COLUMNS,
        [
            "발전량 유효 시간 수",
            "설비용량 유효 시간 수",
            "기온 유효 시간 수",
            "습도 유효 시간 수",
            "풍속 유효 시간 수",
            "전운량 유효 시간 수",
        ],
        strict=True,
    )
)


def aggregate_hourly(rows):
    """KST 날짜별 24개 관측만 완전한 합계/평균으로 사용한다."""
    from collections import defaultdict

    from data.features import validate_rows as validate_hourly

    groups = defaultdict(list)
    for row in validate_hourly(rows, allow_missing_capacity=True):
        groups[row["timestamp"][:10]].append(row)
    start, end = map(datetime.fromisoformat, (min(groups), max(groups)))
    result = []
    while start <= end:
        observations = groups.get(start.date().isoformat(), [])
        point = {"timestamp": start.isoformat(), "region": "제주"}
        for column in NUMERIC_COLUMNS:
            values = [r[column] for r in observations if r[column] is not None]
            point[column] = (
                math.fsum(values) / (1 if column == "generation_mwh" else 24)
                if len(values) == 24
                else None
            )
        result.append(point)
        start += timedelta(days=1)
    return result


def decode_csv(raw: bytes) -> list[dict]:
    from data.features import decode_csv as decode_hourly

    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("UTF-8 또는 CP949 CSV를 사용하세요")
    reader = csv.DictReader(io.StringIO(text))
    fields = set(reader.fieldnames or [])
    if set(DAILY_COLUMN_MAP).issubset(fields):
        if not set(COUNT_COLUMNS.values()).issubset(fields):
            raise ValueError("일별 집계 CSV에는 변수별 유효 시간 수가 필요합니다")
        rows = []
        for source in reader:
            point = {
                target: source[column] for column, target in DAILY_COLUMN_MAP.items()
            }
            for column, count_column in COUNT_COLUMNS.items():
                count = float(source[count_column])
                if not count.is_integer() or not 0 <= count <= 24:
                    raise ValueError("유효 시간 수는 0~24 정수여야 합니다")
                if count != 24:
                    point[column] = None
            rows.append(point)
        return validate_rows(rows)
    # Normalized daily files require an explicit granularity marker; midnight-only
    # hourly uploads must still be aggregated as incomplete hourly observations.
    if set(COLUMN_MAP.values()).issubset(fields) and "granularity" in fields:
        rows = list(reader)
        if any(row["granularity"] != "daily" for row in rows):
            raise ValueError("granularity는 daily여야 합니다")
        return validate_rows([{c: r[c] for c in COLUMN_MAP.values()} for r in rows])
    return aggregate_hourly(decode_hourly(raw, allow_missing_capacity=True))


def load_rows(csv_path: str | Path) -> list[dict]:
    return decode_csv(Path(csv_path).read_bytes())


def complete_point(row: dict) -> bool:
    return all(
        row.get(col) is not None and math.isfinite(row[col]) for col in NUMERIC_COLUMNS
    )


def validate_sequence(rows: list[dict], seq_len: int = SEQ_LEN) -> list[dict]:
    normalized = validate_rows(rows)
    if len(normalized) != seq_len:
        raise ValueError(f"입력은 정확히 {seq_len}일이어야 합니다")
    if [r["timestamp"] for r in normalized] != [
        parse_timestamp(r["timestamp"]).isoformat() for r in rows
    ]:
        raise ValueError("입력은 시간 오름차순이어야 합니다")
    for i, row in enumerate(normalized):
        if not complete_point(row):
            raise ValueError(
                "입력 구간에 결측이 있습니다. 연속된 완전한 관측 구간을 사용하세요"
            )
        if i and parse_timestamp(row["timestamp"]) - parse_timestamp(
            normalized[i - 1]["timestamp"]
        ) != timedelta(days=1):
            raise ValueError("입력 구간에 누락된 날짜가 있습니다")
    return normalized


def sample_windows(rows: list[dict], seq_len: int = SEQ_LEN):
    """정답의 미래 날씨를 읽지 않으며 결측/공백을 가로지르는 입력은 만들지 않는다."""
    run = 0
    previous = None
    for i, row in enumerate(rows):
        stamp = parse_timestamp(row["timestamp"])
        contiguous = previous is not None and stamp - previous == timedelta(days=1)
        if not contiguous:
            run = 0
        if run >= seq_len and row.get("generation_mwh") is not None:
            yield rows[i - seq_len : i], row
        run = run + 1 if complete_point(row) else 0
        previous = stamp


def point_features(row: dict) -> list[float]:
    if not complete_point(row):
        raise ValueError("입력 특징에 결측이 있습니다")
    stamp = parse_timestamp(row["timestamp"])
    year_start = datetime(stamp.year, 1, 1)
    year_end = datetime(stamp.year + 1, 1, 1)
    phase = (
        2
        * math.pi
        * (stamp - year_start).total_seconds()
        / (year_end - year_start).total_seconds()
    )
    return [float(row[c]) for c in NUMERIC_COLUMNS] + [
        math.sin(phase),
        math.cos(phase),
    ]


class SolarScaler:
    def fit(self, rows: list[dict]):
        points = [point_features(r) for r in rows if complete_point(r)]
        if not points:
            raise ValueError("스케일러를 학습할 완전한 관측값이 없습니다")
        self.minimum = [min(col) for col in zip(*points, strict=True)]
        self.maximum = [max(col) for col in zip(*points, strict=True)]
        # 시간 주기는 학습 연도의 범위에 관계없이 정의된 범위를 사용한다.
        self.minimum[-2:] = [-1.0] * 2
        self.maximum[-2:] = [1.0] * 2
        return self

    def transform_point(self, row: dict) -> list[float]:
        return [
            (v - lo) / (hi - lo) if hi != lo else v - lo
            for v, lo, hi in zip(
                point_features(row), self.minimum, self.maximum, strict=True
            )
        ]

    def scale_target(self, value: float) -> float:
        span = self.maximum[0] - self.minimum[0]
        return (value - self.minimum[0]) / (span or 1.0)

    def inverse_target(self, value: float) -> float:
        span = self.maximum[0] - self.minimum[0]
        return float(value) * (span or 1.0) + self.minimum[0]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "features": FEATURE_COLUMNS,
                    "seq_len": SEQ_LEN,
                    "minimum": self.minimum,
                    "maximum": self.maximum,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 2
            or payload.get("features") != FEATURE_COLUMNS
            or payload.get("seq_len") != SEQ_LEN
        ):
            raise ValueError("모델 전처리 버전/특징/입력 길이가 일치하지 않습니다")
        scaler = cls()
        scaler.minimum, scaler.maximum = payload["minimum"], payload["maximum"]
        if len(scaler.minimum) != len(FEATURE_COLUMNS) or len(scaler.maximum) != len(
            FEATURE_COLUMNS
        ):
            raise ValueError("스케일러 특징 수가 다릅니다")
        if not all(math.isfinite(v) for v in scaler.minimum + scaler.maximum):
            raise ValueError("스케일러에 유효하지 않은 값이 있습니다")
        return scaler


def build_sequences(rows, scaler, seq_len=SEQ_LEN):
    x, y = [], []
    for window, target in sample_windows(rows, seq_len):
        x.append([scaler.transform_point(r) for r in window])
        y.append(target["generation_mwh"])
    return x, y


def dataset_summary(rows: list[dict]) -> dict:
    values = [r["generation_mwh"] for r in rows if r["generation_mwh"] is not None]
    expected = (
        int(
            (
                parse_timestamp(rows[-1]["timestamp"])
                - parse_timestamp(rows[0]["timestamp"])
            ).total_seconds()
            / 86400
        )
        + 1
    )
    return {
        "exists": True,
        "rows": len(rows),
        "region": "제주",
        "start_date": rows[0]["timestamp"],
        "end_date": rows[-1]["timestamp"],
        "missing_days": expected - len(rows),
        "missing_targets": len(rows) - len(values),
        "min_generation_mwh": min(values) if values else None,
        "max_generation_mwh": max(values) if values else None,
        "capacity_mw": rows[-1]["capacity_mw"],
        "missing_features": {
            c: sum(r[c] is None for r in rows) for c in NUMERIC_COLUMNS
        },
        "unit": "MWh",
        "granularity": "daily",
        "complete_days": sum(complete_point(r) for r in rows),
        "incomplete_days": sum(not complete_point(r) for r in rows),
    }
