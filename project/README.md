# 제주 일별 태양광 발전량 예측 · AIOps

저장소 루트에서 실행합니다. 기존 Dashboard의 API 지표, 운영 예측 기록,
성능 감시, 자동 재학습, MLflow 이력을 유지하며 **완료된 14일 → 다음 날짜 총발전량**으로 전환했습니다.
데이터 출처: [한국동서발전 제주지역 태양광 발전량 예측 학습용 데이터셋](https://www.data.go.kr/data/15143885/fileData.do).
개별 발전소나 다음 날의 시간별 발전 곡선을 예측하는 모델은 아닙니다.

## 데이터와 예측 계약

- `data/aggregated/jeju_solar_daily_2019_2024.csv`: 2019~2024년 2,192일.
  발전량이 완전한 날짜 2,146일, 모든 입력 변수가 완전한 날짜 2,098일.
- 입력은 연속 14일의 발전량 합계(MWh), 설비용량 평균(MW), 기온·습도·풍속·전운량 평균입니다.
  연중 날짜 sin/cos를 더해 총 8개 특징을 사용합니다.
- 시간별 원본 CSV(CP949/UTF-8)도 업로드할 수 있습니다. KST 날짜별로 자동 집계합니다.
  각 변수의 유효 관측이 24개일 때만 합계/평균을 사용합니다. 원본은 보존합니다.
- 일별 집계 CSV는 변수별 유효 시간 수를 검사합니다. `관측 발전량 부분합`은 정답으로 사용하지 않습니다.
  정규화 CSV는 `timestamp,region,generation_mwh,capacity_mw,temperature,humidity,wind_speed,cloud_cover,granularity`
  컬럼을 사용하며 `granularity=daily`를 명시합니다. 정규화 값은 완전한 일별 관측을 의미합니다.
- 날짜는 KST 00:00 라벨이며 해당 날짜 24시간의 값입니다. 누락일·결측값을 0이나 보간값으로 채우지 않습니다.
  결측일을 가로질러 14일 입력을 만들지 않습니다. 정답 날짜의 기상값은 입력하지 않습니다.
- 하루가 끝나기 전의 집계값은 `/predict` 입력이나 실제값 연결에 사용할 수 없습니다.
  직전 날짜의 관측이 확정된 **대상 날짜 KST 00:00~00:59** 발행분만 운영 예측으로 집계합니다.
  이 수집 유예 시간에는 이미 대상 날짜가 시작되었으므로 엄밀한 하루 전 예보는 아닙니다.
  이후 발행분은 점검용으로 분류합니다. 실제 수집 지연·시간 구간 라벨은 운영 연동 시 확인해야 합니다.

현재는 CSV 업로드 방식이며 실시간 수집기와 주기적 예측 스케줄러는 없습니다.
과거 CSV로 실행하면 과거 날짜 점검으로 표시하며 운영 성능에 포함하지 않습니다.

## 설치와 학습

```bash
uv sync --locked
uv run python project/scripts/train_baseline_v1.py \
  --csv project/data/aggregated/jeju_solar_daily_2019_2024.csv --epochs 20
uv run uvicorn serving_app.main:app --app-dir project --port 8077
```

학습 데이터는 2019~2022년, 후보 선택·조기 종료는 2023년 1~6월,
최종 검증 게이트는 2023년 7~12월, 테스트 보고는 2024년으로 분리합니다.
스케일러는 학습 기간에만 맞춥니다. 결측 때문에 실제 유효 평가 기간과 건수는 줄어듭니다.
각 예측 시점까지의 실제 관측을 사용하는 순차 하루 예측이며, 1년 전체를 한 번에 예측하지 않습니다.
2024년 자료는 기존 실험에서도 확인했으므로 완전히 새로운 데이터의 성능은 아닙니다.

초기 후보는 LSTM 보정과 Ridge 보정입니다. Ridge 강도 `0.01 / 0.1 / 1 / 10`과 후보는
앞선 선택 기간 RMSE로만 고릅니다. 최종 검증·테스트 정답은 후보 선택에 사용하지 않습니다.
같은 날짜의 **전날 발전량 유지**와 **최근 7일 발전량 평균**보다 RMSE가 모두 낮아야 등록합니다.
MAE/RMSE/WAPE와 월별 지표를 기록하며 시간별 모델의 주간/야간 지표는 사용하지 않습니다.
WAPE는 `절대 오차 합 / 실제 발전량 합 × 100`으로 계산한 백분율입니다.
실제 합계가 0이면 `null`(계산 불가)이며, MLflow 숫자 지표에서는 생략합니다.
운영 WAPE는 RMSE와 동일한 최근 14일 관측 구간으로 계산하며 14일 미만은 잠정값입니다.
WAPE는 보조 지표이고 감지·후보 선택·승격 기준은 기존 RMSE를 유지합니다.
기존 모델의 WAPE는 소급 추정하지 않고 미기록으로 표시하며 새 학습부터 저장합니다.
로컬 번들은 게이트 미통과 시에도 실험용으로 저장되므로 `gate_passed`를 확인해야 합니다.

실제 일별 CSV 학습 결과(2026-10-07): 선택 후보 `daily_ridge_1`.

| 구간 | 일별 모델 RMSE | 전날 유지 | 최근 7일 평균 |
| --- | ---: | ---: | ---: |
| 2023년 후반 유효 검증 날짜 | 484.00 MWh | 596.10 MWh | 498.21 MWh |
| 2024년 유효 테스트 날짜 | 515.16 MWh | 623.32 MWh | 566.96 MWh |

일별 합계 오차이므로 기존 시간별 RMSE와 수치를 직접 비교하지 않습니다.

## 예측과 운영 기록

[대시보드](http://localhost:8077/)에서 CSV를 업로드하고 입력 예제를 불러옵니다.
`GET /data/preview`는 최근 완전한 14일 입력을 `example`으로 반환합니다.
`POST /predict`에 이 JSON을 보내면 다음 날짜 총발전량을 예측합니다.
`project/data/predict_example.json`에도 일별 입력 예제를 제공합니다.

응답에는 `predicted_generation_mwh`, `target_timestamp`, `input_end_timestamp`,
`issued_at`, `prediction_id`, `model_version`, `forecast_context`, `monitoring_eligible`이 포함됩니다.
날짜 순서·중복·누락·결측 오류는 422입니다.

`runtime/forecasts-daily.db`에 최초 예측을 보존합니다. 같은 모델·대상 날짜를 다시 요청해도
최초 예측을 반환합니다. CSV 업로드가 도착하면 해당 날짜 종료 후 실제값을 연결하며 한 번
연결된 값은 덮어쓰지 않습니다. `GET /predictions/recent`에서 최근 30건을 조회합니다.
학습·후보 선택·검증 범위와 겹치는 예측, 과거 점검, 합성 시연 모델의 예측은 운영 감시에서 제외합니다.

현재 모델의 연속 14일 실제값이 연결되면 성능을 평가합니다. 검증 RMSE × 1.5를 넘으면
자동 재학습합니다. 성능 저하는 드리프트의 확정 진단이 아닙니다.
재학습에는 최소 90일 관측이 필요하며 최근 365일을 사용합니다.
마지막 14일은 최종 검증, 그 직전 28일은 후보 선택, 그 이전은 학습입니다.
최종 검증 날짜가 기존 모델의 학습·선택·검증 종료 이후여야 합니다.
후보가 같은 날짜의 기존 모델·전날 유지·7일 평균보다 모두 우수해야 승격합니다.
후보 번들을 로딩하고 실제 추론까지 확인한 뒤 Production과 서빙 캐시를 교체합니다.

같은 모델·관측 종료 날짜를 중복 평가하지 않습니다. 이력 부족·학습 실패·활성화 실패는
원래 관측값이 같은 전체 이력 CSV를 다시 업로드하여 재시도합니다.
동시 작업은 `busy`로 반환하며 작업 종료 후 다시 업로드합니다. 중단된 평가 예약은 10분 후 재시도합니다.

## MLflow와 시연

```bash
uv run python project/serving_app/train_and_register.py \
  --csv project/data/aggregated/jeju_solar_daily_2019_2024.csv
MODEL_SOURCE=mlflow LOADING_MODE=eager uv run uvicorn \
  serving_app.main:app --app-dir project --port 8077
uv run mlflow ui --backend-store-uri sqlite:///project/runtime/mlflow.db --port 5001
```

모델 이름은 **JejuSolarDailyPredictor**입니다. 시간별 `JejuSolarPredictor`와 분리합니다.
로컬 번들은 `serving_app/models/solar-daily/`이며 모델·스케일러·메타데이터의 해시,
입력 길이·특징·일별 대상 계약을 확인합니다. 시간별 번들은 재사용하지 않습니다.
외부 명령으로 승격한 경우 서버를 재시작합니다. 자동 재학습은 즉시 캐시까지 교체합니다.

Dashboard의 Simulation에서 정상/드리프트 배치를 실행합니다. 기본은 **2024-07-21부터 14일**입니다.
정상 배치는 원본 관측값, 드리프트 배치는 3일마다(날짜 ordinal이 3의 배수)에 발전량을 10%로
제한한 합성 사본을 사용합니다. 과거 입력·정답·재학습에 같은 변환을 적용합니다.
기본 구간은 초기 모델의 정상/저하 차이를 보여 주기 위해 선택한 시연 구간입니다.
모델의 테스트 성능 근거나 게이트 통과 보장은 아닙니다.

감지 → 재학습 → 같은 구간 검증 → 통과 시 승격을 실행합니다. 이 실행은 실제
**일별 Production 모델을 교체**하며 메타데이터에 합성 시연 여부를 기록합니다.
게이트 미통과나 로딩 실패 시 기존 모델을 유지합니다. 승격 후에는 새 모델의 검증 종료 이후
데이터를 선택합니다. 원본 CSV는 수정하지 않으며 결과는 `runtime/solar_daily_simulation.jsonl`에 보존합니다.
수동 과거 리플레이 API와 시간 경계를 지정하는 수동 학습 화면은 제공하지 않습니다.

```bash
uv run python project/scripts/simulate_drift.py --target container --scenario both
```

CLI는 `--target local|container|both`, `--scenario normal|drift|both`, `--start YYYY-MM-DD`를 지원합니다.
같은 원본 해시인 경우에만 데이터가 같은 비교로 표시합니다.

## Docker

```bash
docker compose -f project/serving_app/docker-compose.yml up --build -d
docker compose -f project/serving_app/docker-compose.yml logs -f serving-app
```

[컨테이너 Dashboard](http://localhost:8099/)에서 확인합니다.
`AUTO_PREPARE=1`, `MODEL_SOURCE=mlflow`가 기본입니다. 처음에는 일별 모델을 학습·검증·등록하고,
기존 일별 Production이 있으면 재시작 시 재사용합니다. 게이트 미통과 시 시작을 중단합니다.
기존 시간별 모델이 있는 볼륨도 일별 모델 이름·예측 DB·시연 로그가 분리되어 공존합니다.
업로드 폴더는 호스트와 공유하고 MLflow와 이력은 `solar-runtime`에 보존합니다.
호스트 DB에는 호스트 경로가 있으므로 컨테이너에 DB 파일만 복사하지 마세요.
`docker compose down -v`는 저장 볼륨까지 삭제합니다.

`MODEL_SOURCE=local`도 지원하며 로컬 일별 번들이 없으면 `runtime/bootstrap-solar-daily`에 준비합니다.
이 모드에서는 운영 자동 재학습이 차단됩니다. API 요청 지표는 실제 요청의 시간 단위를 유지합니다.

## 검사

```bash
uv run python -m unittest discover -s tests -v
node tests/dashboard_ui.cjs
uv run ruff check project tests
uv run ruff format --check project tests
```
