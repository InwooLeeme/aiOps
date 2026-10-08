# miniProject — 태양광 발전량 예측 AIOps 서비스

제주 태양광 일별 발전량을 예측하는 LSTM을 FastAPI로 서빙하고, 드리프트가 감지되면 스스로 재학습해 Production에 승격하는
파이프라인(MLflow Registry)과 5탭 대시보드(Dashboard / Simulation / Datasets / Logs / System)입니다.

## 실행

```bash
cd miniProject
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
# data/uploads/ 에 발전량 CSV(jeju_solar_daily_2019_2024.csv), data/ 에 weather_jeju_forecast.csv 가 있어야 함
.venv/bin/python scripts/fetch_weather_forecast.py       # (예보 CSV가 없을 때) Open-Meteo 과거 예보 수집
.venv/bin/python serving_app/train_and_register.py       # base 모델 학습 → MLflow 등록 (v1, Production)
LOADING_MODE=eager .venv/bin/uvicorn serving_app.main:app --port 8077
# 대시보드 http://localhost:8077/  ·  API 문서 /docs
```

Simulation 탭에서 "리플레이 시작"(2023 정상 → 2024 드리프트). 처음 상태로 되돌리려면
`rm -rf mlflow.db mlruns logs/aiops.log serving_app/models/*` 후 train_and_register.py 를 다시 실행.

## 동작 원리

| 단계 | 내용 |
|---|---|
| 예측 | 과거 14일 실적 + 해당일 **예보** 기상(기온·습도·풍속·전운량·일사량) → 다음날 이용률 → MWh |
| 감지 | 최근 21일 **WAPE** > 31% (고정 임계치 = 검증 2022 롤링 WAPE 최대값 올림, 근거는 `monitoring/drift_detector.py`). 경보 로그에 Bias 방향을 함께 기록 |
| 재학습 | Production 가중치에서 최근 90일로 fine-tuning (마지막 21일은 검증 전용) |
| 게이트(배포 판정) | 같은 검증 구간에서 현재 Production 대비 **RMSE** 10% 이상 개선 시에만 승격 (상대 기준) |
| 로그 | `logs/aiops.log` — 대시보드 Logs 탭에서 전체 이력 조회·필터·검색·다운로드 |

## 지표별 역할 (교수님 권장 기준 대응)

| 상황 | 권장 지표 | 우리 구현 |
|---|---|---|
| 모델 선정·배포 판정 | RMSE | 승격 게이트(`train_and_register.fine_tune`), early stopping(MSE 손실), 버전 이력 표 |
| 드리프트 탐지·재학습 트리거 | WAPE | 21일 WAPE > 임계치 (`monitoring/drift_detector.py`) |
| 물량 변동 구간 포함 평가 | MAE + WAPE | 시뮬레이션 비교표·버전 이력 표에 병기 |
| 과소·과대예측 방향 확인 | Bias | 비교표·버전 이력·경보 로그에 표시 (양수=과대예측) |

MAPE 는 쓰지 않습니다: 흐린 날 실측이 작아 일별 오차가 수천 %까지 튑니다(2024 최대 2,841%).
감지 임계치(WAPE 31%)는 데이터·모델·계절에 의존하므로 모델이나 데이터가 바뀌면 `scripts/calibrate_drift.py` 로 재보정해야 합니다(겨울에 WAPE 가 구조적으로 높음).

## 알려진 한계
- 데이터 분할: 학습 2019-2021 / 검증 2022(임계치 보정) / 정상 운영 2023(오탐 확인) / 드리프트 2024. 임계치(31%)는 검증 한 해의 롤링 최대값이라 통계적 신뢰도가 낮고, 2023 에서 실제로 오탐 2건이 나왔습니다(둘 다 게이트가 막음).
- 드리프트 사건이 2024년 한 번뿐이라 일반화된 검증은 아닙니다. 이용률 하락의 원인(출력제어·설비 집계 기준 등)은 확인하지 못했습니다.
- 예보 CSV 는 Open-Meteo 과거 예보(해당일 시점의 예보값)입니다. 실시간 예보 API 연동은 아직 없습니다.
- Docker 이미지는 수정만 했고 빌드·실행은 확인하지 않았습니다. 대시보드 화면은 브라우저로 직접 확인하지 못했습니다(스크립트 문법과 API 응답만 검증).
