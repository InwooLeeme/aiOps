# 제주 태양광 발전량 예측 서비스

아래 명령은 별도 표기가 없으면 **저장소 루트** 기준입니다.
데이터 출처: [한국동서발전 제주지역 태양광 발전량 예측 학습용 데이터셋](https://www.data.go.kr/data/15143885/fileData.do).
예측 대상은 제공된 제주 집계 `태양광 발전량 합계(MWh)`이며 개별 발전소 고장 진단이나
하루 전 24시간 예측 모델이 아닙니다.

## 데이터와 예측 계약

- 입력: 과거 72개 연속 시간의 실제 발전량, 설비용량, 기온, 습도, 풍속, 전운량(10분위).
- 시간대: Asia/Seoul. 시각은 제공 자료의 시간 구간 라벨입니다. 구간 시작/종료 의미와 실제 수집 지연은 운영 연동 전에 제공기관에 확인해야 합니다.
- 파생 변수: 시간 sin/cos, 윤년 길이를 반영한 연중 시각 sin/cos. 총 10개 특징.
- 정답: 입력 마지막 시각보다 정확히 한 시간 뒤의 발전량(MWh).
- 미래 관측 기상값, 예측 대상 시간의 이용률은 입력하지 않습니다.
- 발전량 0, 기온 음수, 소수 기상값은 정상 입력입니다. 관측된 야간 양수 발전량은 보존합니다.
- 일사량·강수·적설은 결측 의미가 불명확하여 초기 특징에서 제외했습니다.

첨부 CSV에서 확인한 품질: 2019-01-01~2024-12-31, 52,248행, CP949,
중복 시각 없음, 2024년 누락 시간 360개, 2023년 12월 매일 00시의 정답 결측 31개.

결측 정답은 0이나 보간값으로 대체하지 않습니다. 누락 시간을 가로지르거나 입력 관측값에
결측이 있는 72시간 구간은 학습·재생에서 제외합니다. 타깃 행의 미래 기상값은 사용하지
않으므로 해당 값의 결측은 정답 발전량이 있는 한 평가를 막지 않습니다.
CSV를 정렬하고 중복·잘못된 시각·무한대·범위 밖 값을 검증합니다.

## 설치·업로드·학습

```bash
uv sync --locked
uv run uvicorn serving_app.main:app --app-dir project --port 8077
```

[대시보드](http://localhost:8077/) 또는 `/data/upload`에 원본 CSV를 올립니다.

```bash
curl -F 'file=@/실제/경로/제주태양광.csv' http://localhost:8077/data/upload
uv run python project/scripts/train_baseline_v1.py \
  --csv /실제/경로/제주태양광.csv --epochs 20
```

원본은 `project/data/uploads/solar_*.csv`로 보관합니다. 과거 HAIC 업로드는 선택하지 않습니다.
현재 작업의 원본 사본은 `project/data/uploads/solar_jeju_2019_2024.csv`입니다.
로컬 번들은 `project/serving_app/models/solar/`의 `model.keras`, `scaler.json`, `metadata.json`에 저장됩니다.
처음부터 소스에 포함된 주가 가중치·스케일러로 예측하지 않습니다.

학습은 2019~2022년, 조기 종료와 설정 검증은 2023년, 최종 평가는 2024년 타깃으로 분리합니다.
스케일러는 2023년 이전 관측치만으로 적합합니다. 평가 입력에 평가 시점보다 앞선
관측치를 사용하는 순차 1시간 예측이며, 2024년 전체를 한 번에 예측하는 방식이 아닙니다.

검증 기간에서 직전 시간 유지·전날 같은 시간 유지 기준선과 같은 타깃으로 비교합니다.
MAE·RMSE·06~18시의 주간 대용 지표를 기록합니다. 주간 지표는 실제 일출/일몰 판정이 아닙니다.
검증 오차가 두 기준선보다 모두 낮아야 Production 등록을 통과합니다.
로컬 번들은 기준을 통과하지 않아도 실험용으로 저장되며 `gate_passed`를 확인해야 합니다.
최종 테스트 결과는 학습·조기 종료·승격 판단에 사용하지 않습니다.

## 단일 예측

```bash
curl -H 'Content-Type: application/json' \
  --data-binary @project/data/predict_example.json http://localhost:8077/predict
```

응답 필드: `predicted_generation_mwh`, `target_timestamp`, `region`, `model_version`.
예제는 첨부 CSV 마지막 72시간의 관측값입니다. `GET /data/preview`의 `example`에서도
현재 업로드 자료의 최근 완전한 관측 구간을 받을 수 있습니다.
잘못된 시각 순서·중복·누락 시간·필수값 결측은 422입니다.

## MLflow 등록과 서빙

```bash
uv run python project/serving_app/train_and_register.py \
  --csv project/data/uploads/solar_jeju_2019_2024.csv --epochs 20
uv run mlflow ui --backend-store-uri sqlite:///project/runtime/mlflow.db --port 5001
```

실험 `JejuSolar`, 모델 `JejuSolarPredictor`를 사용합니다. 승격을 통과한 뒤 별도 서버를 실행합니다.

```bash
MODEL_SOURCE=mlflow LOADING_MODE=eager uv run uvicorn \
  serving_app.main:app --app-dir project --port 8077
```

모델의 숫자 버전을 고정하여 해당 run의 스케일러와 메타데이터를 함께 로딩합니다.
모델·스케일러 파일 해시와 특징 순서·길이·단위가 다르면 로딩을 거부합니다.
MLFLOW_TRACKING_URI를 지정하면 학습·서빙·MLflow UI에서 같은 저장소를 사용하세요.
외부 학습 명령으로 모델이 승격되면 서버 재시작이 필요합니다. `/retrain`을 통한 승격은
새 번들 로딩까지 수행하며 로딩 실패 시 메모리의 기존 모델을 유지합니다.

## 실제 이력 재생과 성능 감시

### 드리프트 시뮬레이션과 자동 재학습

대시보드의 **Simulation → 정상 배치 전송 / 드리프트 배치 전송**으로 실행합니다.
기본 시연 구간은 `2024-05-23 17:00`부터 연속 168시간입니다.

- 정상 배치: 원본 관측값으로 평가합니다. 임계값 이내면 재학습하지 않습니다.
- 드리프트 배치: 최근 90일 사본에서 08~18시 짝수 시간의 발전량을 10%로 제한합니다.
  같은 합성 관측값을 과거 입력·정답·재학습에 일관되게 사용합니다.
- 168시간 RMSE가 기존 모델의 임계값을 넘으면 자동 재학습합니다. 마지막 7일은 검증으로
  남기고, 새 모델이 기존 모델과 두 기준선보다 모두 좋을 때만 승격합니다.
- 결과에는 배치 RMSE, 재학습 검증 RMSE, 승격 여부와 시뮬레이션 모델 버전이 표시됩니다.
  구간이나 운영 모델에 따라 감지·승격 결과가 달라지며, 성공을 강제로 만들지 않습니다.

시뮬레이션은 매번 현재 운영 모델을 출발점으로 사용합니다. 원본 CSV와 운영 모델
`JejuSolarPredictor`는 변경하지 않으며, 합성 데이터 모델은 **JejuSolarSimulator**의
Production으로 등록하고 별도로 로딩·추론하여 검증합니다. 화면에서도 시뮬레이션으로
표시합니다. 마지막 결과는 `runtime/solar_simulation.jsonl`에 저장해 새로고침 후 복원합니다.
실제 관측 평가 지표와 재학습 승인 상태에는 시뮬레이션 결과를 섞지 않습니다.

```bash
curl -H 'Content-Type: application/json' \
  -d '{"scenario":"normal"}' http://localhost:8099/simulation/run
curl -H 'Content-Type: application/json' \
  -d '{"scenario":"drift"}' http://localhost:8099/simulation/run
```

### 원본 데이터 리플레이와 수동 재학습

```bash
uv run python project/scripts/simulate_drift.py \
  --target local --start 2024-01-01T00:00:00 --limit 168
```

같은 기능을 대시보드의 **실제 과거 구간 리플레이** 또는 `POST /predict/batch-test`에서 사용할 수 있습니다.
요청은 `{"start_timestamp":"2024-01-01T00:00:00","limit":168}` 형태입니다.
`limit`는 유효한 정답 건수(최대 744)입니다. 결측 구간을 건너뛰므로 실제 경과시간은 더 길 수 있습니다.
모델 학습·검증 종료 시각 이후만 평가하며, 재생 요청끼리는 기록을 섞지 않습니다.
`project/runtime/solar_replay.jsonl`에 재생 ID, 자료 해시, 타깃 시각, 정답·예측·모델 버전을 저장합니다.
단일 `/predict` 요청의 정답 자동 수집 기능은 없으며, 실시간 발전량 수집 연동은 별도입니다.

최근 **168시간 안에 168개의 서로 다른 정답**이 있어야 성능을 판정합니다.
기준은 해당 모델의 검증 RMSE × 1.5입니다. 이는 초기 감시 규칙이며 통계적으로 검증된
개념 드리프트 판정이 아닙니다. `performance_degraded`는 원인 확인이 필요한 성능 저하 신호입니다.
결측으로 관측이 부족하면 `insufficient_data`, 모델 기준이 없으면 `threshold_unavailable`입니다.

재생은 자동으로 모델을 바꾸지 않습니다. 데이터 품질·기상 변화·설비 상태 등을 확인한 뒤
대시보드에서 명시적으로 재학습합니다. `/retrain`의 `cutoff_timestamp`는 마지막 재생의
마지막 정답 시각과 같아야 합니다. MLflow Production 모드, 동일 자료·모델 버전,
성능 저하 판정을 요구합니다.
재학습은 그 시각까지의 최근 90일만 읽고, 마지막 연속 7일은 검증으로 남깁니다.
최소 720행이 필요하며 새 모델은 기준선과 기존 모델을 같은 검증 구간에서 모두 이겨야 승격됩니다.
같은 요청의 미래 자료나 최신 파일 끝부분을 무조건 학습하지 않습니다.
서버 재시작 후에는 해당 이력을 다시 평가해야 재학습할 수 있습니다.

운영 로그는 `project/logs/solar_aiops.log`, 요청 지표는 `solar_requests.log`를 사용합니다.
이전 주가 실험 기록과 분리됩니다.

## Docker

기본 실행 모드는 MLflow Production입니다. 이미지 빌드 중 학습하지 않습니다.
최초 실행 시 컨테이너의 영속 `solar-runtime` 볼륨에 모델을 학습·등록합니다.
이미 해당 볼륨에 Production 모델이 있다면 초기 등록 명령은 생략합니다.

```bash
docker compose -f project/serving_app/docker-compose.yml build
docker compose -f project/serving_app/docker-compose.yml run --rm serving-app \
  python serving_app/train_and_register.py \
  --csv /app/data/uploads/solar_jeju_2019_2024.csv --epochs 20
# 위 결과에서 promoted=true를 확인한 뒤 실행합니다.
docker compose -f project/serving_app/docker-compose.yml up -d
```

[컨테이너 대시보드](http://localhost:8099/)에서 `MODEL_SOURCE=mlflow`, 현재 운영 모델의
`Production` 스테이지와 버전을 확인합니다. 업로드 폴더는 호스트와 공유하며,
MLflow DB·모델 파일·평가 이력은 `solar-runtime` 볼륨에 함께 보관됩니다.
호스트의 MLflow 저장소와는 별도입니다. 호스트 DB에는 호스트 절대 경로가 기록되므로
DB 파일만 복사해서 연결하지 마세요. `docker compose down -v`는 저장 볼륨도 삭제합니다.

시작 시 Production 모델과 스케일러를 즉시 로딩하므로 등록이 없거나 파일이 손상되면
시작에 실패합니다. 서버 재시작 후 Replay를 다시 실행하고, 성능 저하가 확인된 경우에만
재학습을 요청하세요. 새 모델이 기존 모델과 기준선보다 좋아야 승격됩니다.

기존 로컬 번들로 실행하려면 아래처럼 모드를 명시할 수 있습니다.
로컬 번들은 읽기 전용으로 마운트하며 이 모드에서는 재학습이 차단됩니다.

```bash
MODEL_SOURCE=local docker compose -f project/serving_app/docker-compose.yml up -d
```

## 검사

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check project tests
uv run ruff format --check project tests
```

테스트는 CP949·결측·시간 공백·미래 타깃 누수·스케일러 학습 범위·0 발전량 API·
중복 재생·재학습 cutoff·기준선 게이트·아티팩트 버전 일치·캐시 교체를 확인합니다.
