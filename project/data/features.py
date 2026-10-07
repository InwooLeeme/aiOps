"""제주 태양광 CSV 검증과 학습·서빙 공용 전처리. 시각은 KST 시간 구간 라벨."""

import csv
import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEQ_LEN = 72
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
FEATURE_COLUMNS = NUMERIC_COLUMNS + ["hour_sin", "hour_cos", "year_sin", "year_cos"]
KST = timezone(timedelta(hours=9))


def parse_timestamp(value) -> datetime:
    t = datetime.fromisoformat(str(value))
    if t.tzinfo:
        t = t.astimezone(KST).replace(tzinfo=None)
    if t.minute or t.second or t.microsecond:
        raise ValueError("일자 및 시각은 정시(1시간 간격)여야 합니다")
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
            if row["capacity_mw"] is None or row["capacity_mw"] <= 0:
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


def decode_csv(raw: bytes) -> list[dict]:
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
    if set(COLUMN_MAP).issubset(fields):
        rows = [
            {target: r[source] for source, target in COLUMN_MAP.items()} for r in reader
        ]
    elif set(COLUMN_MAP.values()).issubset(fields):
        rows = [{col: r[col] for col in COLUMN_MAP.values()} for r in reader]
    else:
        raise ValueError(
            "제주 태양광 CSV 필수 컬럼이 없습니다: " + ", ".join(COLUMN_MAP)
        )
    return validate_rows(rows)


def load_rows(csv_path: str | Path) -> list[dict]:
    return decode_csv(Path(csv_path).read_bytes())


def complete_point(row: dict) -> bool:
    return all(
        row.get(col) is not None and math.isfinite(row[col]) for col in NUMERIC_COLUMNS
    )


def validate_sequence(rows: list[dict], seq_len: int = SEQ_LEN) -> list[dict]:
    normalized = validate_rows(rows)
    if len(normalized) != seq_len:
        raise ValueError(f"입력은 정확히 {seq_len}시간이어야 합니다")
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
        ) != timedelta(hours=1):
            raise ValueError("입력 구간에 누락된 시간이 있습니다")
    return normalized


def sample_windows(rows: list[dict], seq_len: int = SEQ_LEN):
    """정답의 미래 날씨를 읽지 않으며 결측/공백을 가로지르는 입력은 만들지 않는다."""
    run = 0
    previous = None
    for i, row in enumerate(rows):
        stamp = parse_timestamp(row["timestamp"])
        contiguous = previous is not None and stamp - previous == timedelta(hours=1)
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
    hour_phase = 2 * math.pi * stamp.hour / 24
    return [float(row[c]) for c in NUMERIC_COLUMNS] + [
        math.sin(hour_phase),
        math.cos(hour_phase),
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
        self.minimum[-4:] = [-1.0] * 4
        self.maximum[-4:] = [1.0] * 4
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
                    "schema_version": 1,
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
            payload.get("schema_version") != 1
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
            / 3600
        )
        + 1
    )
    return {
        "exists": True,
        "rows": len(rows),
        "region": "제주",
        "start_date": rows[0]["timestamp"],
        "end_date": rows[-1]["timestamp"],
        "missing_hours": expected - len(rows),
        "missing_targets": len(rows) - len(values),
        "min_generation_mwh": min(values) if values else None,
        "max_generation_mwh": max(values) if values else None,
        "capacity_mw": rows[-1]["capacity_mw"],
        "missing_features": {
            c: sum(r[c] is None for r in rows) for c in NUMERIC_COLUMNS
        },
        "unit": "MWh",
    }
