"""태양광 관측값 및 예측 API. 시각은 한국 표준시(KST)의 시간 구간 라벨."""

from typing import Literal

from data.features import SEQ_LEN, parse_timestamp, validate_sequence
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HourlyPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamp: str
    region: Literal["제주"] = "제주"
    generation_mwh: float = Field(ge=0)
    capacity_mw: float = Field(gt=0)
    temperature: float
    humidity: float = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0)
    cloud_cover: float = Field(ge=0, le=10)

    @field_validator("timestamp")
    @classmethod
    def timestamp_is_hourly(cls, value):
        return parse_timestamp(value).isoformat()


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence: list[HourlyPoint] = Field(min_length=SEQ_LEN, max_length=SEQ_LEN)

    @model_validator(mode="after")
    def continuous_hours(self):
        validate_sequence([r.model_dump() for r in self.sequence])
        return self


class PredictResponse(BaseModel):
    predicted_generation_mwh: float
    target_timestamp: str
    region: str
    model_version: str


class BatchTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_timestamp: str = "2024-01-01T00:00:00"
    limit: int = Field(default=168, ge=1, le=744)

    @field_validator("start_timestamp")
    @classmethod
    def timestamp_is_hourly(cls, value):
        return parse_timestamp(value).isoformat()


class BatchTestResponse(BaseModel):
    predictions: list[float]
    records: list[dict]
    drift_check: dict


class RetrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cutoff_timestamp: str

    @field_validator("cutoff_timestamp")
    @classmethod
    def timestamp_is_hourly(cls, value):
        return parse_timestamp(value).isoformat()


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: Literal["normal", "drift"]
    start_timestamp: str = "2024-05-23T17:00:00"

    @field_validator("start_timestamp")
    @classmethod
    def timestamp_is_hourly(cls, value):
        return parse_timestamp(value).isoformat()
