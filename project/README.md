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
업로드가 없으면 `project/data/sample_jeju_solar.csv`를 사용합니다. 이 파일은 제공받은
제주 원본 52,248행의 CP949 사본이며 결측·누락 시간을 그대로 보존합니다.
학습 명령에서 `--csv`를 생략해도 최신 업로드 → 샘플 순으로 선택합니다.
잘못된 업로드는 샘플로 숨기지 않고 오류를 반환합니다.
로컬 번들은 `project/serving_app/models/solar/`의 `model.keras`, `scaler.json`, `metadata.json`에 저장됩니다.
처음부터 소스에 포함된 주가 가중치·스케일러로 예측하지 않습니다.

학습은 2019~2022년, 조기 종료·후보 선택은 2023년 1~6월, 최종 검증 게이트는
2023년 7~12월, 비교 평가는 2024년 타깃으로 분리합니다. 2024년은 이전 실험에서도
확인한 자료이므로 완전히 새로운 미사용 데이터의 성능으로 해석하지 않습니다.
모델은 직전 발전량에 전날·이틀 전 같은 시간대의 발전량 증감과 LSTM 보정값을 더합니다.
입력은 기존과 같은 72시간 × 10개 특징이며 미래 관측값을 읽지 않습니다.
전체·주간 RMSE/MAE와 월별 RMSE/MAE를 번들 메타데이터와 MLflow에 기록합니다.
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
외부 학습 명령으로 모델이 승격되면 서버 재시작이 필요합니다. 시뮬레이션 자동 재학습의 승격은
새 번들 로딩까지 수행하며 로딩 실패 시 메모리의 기존 모델을 유지합니다.

## 시뮬레이션과 성능 감시

### 드리프트 시뮬레이션과 자동 재학습

대시보드의 **Simulation → 정상 배치 전송 / 드리프트 배치 전송**으로 실행합니다.
기본 시연 구간은 `2024-05-23 17:00`부터 연속 168시간입니다.

- 정상 배치: 원본 관측값으로 평가합니다. 임계값 이내면 재학습하지 않습니다.
- 드리프트 배치: 최근 90일 사본에서 08~18시 짝수 시간의 발전량을 10%로 제한합니다.
  같은 합성 관측값을 과거 입력·정답·재학습에 일관되게 사용합니다.
- 168시간 RMSE가 기존 모델의 임계값을 넘으면 자동 재학습합니다. 마지막 7일은 최종 검증, 그 직전 7일은 조기 종료 선택 구간으로
  남기고, 새 모델이 기존 모델과 두 기준선보다 모두 좋을 때만 승격합니다.
- 재학습 후보는 기존 모델 미세 조정과 **전날 같은 시간 발전량 + 시간대별 Ridge 보정**입니다.
  보정에는 직전 관측, 전날 예측 대상 시간의 관측, 최근 두 시간의 전날 대비 변화량을 사용합니다.
  발전량·기상·설비용량·시간 특징은 모두 예측 시점 이전의 값입니다.
  Ridge 규제 강도 `0.001 / 0.01 / 0.1 / 1.0`와 후보 구조는 앞선 선택 7일의 RMSE로 결정합니다.
  마지막 검증 7일은 선택된 후보의 승격 판정에만 사용하며, 기준선과 동점이어도 탈락합니다.
  선택 결과는 응답의 `selected_candidate`, MLflow 파라미터·선택 지표와 번들 메타데이터에 기록됩니다.
- 결과에는 배치 RMSE, 재학습 검증 RMSE, 승격 여부와 실행 후 운영 모델 버전이 표시됩니다.
  구간이나 운영 모델에 따라 감지·승격 결과가 달라지며, 성공을 강제로 만들지 않습니다.

정상/합성 배치는 현재 운영 모델에서 평가하며, 원본 CSV는 수정하지 않습니다.
검증을 통과하면 **JejuSolarPredictor Production과 실제 서빙 캐시를 함께 교체**합니다.
학습 실패·게이트 미통과·후보 로딩 실패 시 기존 모델을 유지합니다.
합성 여부와 원본 해시를 모델 메타데이터에 기록하며, 마지막 결과는
`runtime/solar_simulation.jsonl`에서 새로고침 후 복원합니다.
원본과 합성 관측의 감시 창은 분리합니다. 승격 후에는 새 모델의 학습·검증 종료 이후
구간을 입력하세요. 기본 출력 제한 시나리오는 드리프트가 감지되어도 게이트에서
탈락할 수 있으며 이는 정상적인 보호 동작입니다.

CLI에서도 동일한 변환과 운영 파이프라인을 실행합니다.

```bash
uv run python project/scripts/simulate_drift.py --target both --scenario both
```

`--scenario normal|drift|both`, `--target local|container|both`를 지원합니다.
서버 하나가 실패해도 나머지를 실행하고 종료 코드는 1을 반환합니다.
동일 원본 해시일 때만 데이터 비교 가능으로 표시하므로, 모델 버전도 결과에서 확인하세요.

```bash
curl -H 'Content-Type: application/json' \
  -d '{"scenario":"normal"}' http://localhost:8099/simulation/run
curl -H 'Content-Type: application/json' \
  -d '{"scenario":"drift"}' http://localhost:8099/simulation/run
```

시뮬레이션에서 성능 저하가 확인되면 해당 배치의 마지막 관측까지 최근 90일로
자동 재학습합니다. 학습·후보 선택·최종 검증 구간은 분리하며, 기존 모델과 후보는
같은 최종 검증 데이터에서 비교합니다. 별도의 과거 구간 리플레이와 수동 재학습
화면·API는 제공하지 않습니다.

성능 감시 기준은 해당 모델의 검증 RMSE × 1.5입니다. 이는 초기 감시 규칙이며
`performance_degraded`는 원인 확인이 필요한 성능 저하 신호입니다.
단일 `/predict` 요청의 정답 자동 수집과 실시간 발전량 수집 연동은 별도입니다.

운영 로그는 `project/logs/solar_aiops.log`, 요청 지표는 `solar_requests.log`를 사용합니다.
이전 주가 실험 기록과 분리됩니다.

## Docker

기본 실행 모드는 MLflow Production이며 `AUTO_PREPARE=1`입니다.
이미지 빌드 중 학습하지 않습니다. 최초 시작 때 영속 `solar-runtime` 볼륨이 비어 있으면
최신 업로드 또는 포함된 제주 샘플로 학습·검증·등록하고 서비스를 시작합니다.
후보 번들을 로딩하고 실제 예측한 뒤 Production으로 승격합니다. 게이트에서 탈락하면
시작을 중단하고 오류를 남깁니다. 기존 Production이 있으면 재학습 없이 재사용합니다.

```bash
docker compose -f project/serving_app/docker-compose.yml up --build -d
# 최초 실행은 학습 시간이 필요합니다.
docker compose -f project/serving_app/docker-compose.yml logs -f serving-app
```

로컬에서도 자동 준비를 사용할 수 있습니다.

```bash
AUTO_PREPARE=1 MODEL_SOURCE=mlflow LOADING_MODE=eager uv run uvicorn \
  serving_app.main:app --app-dir project --port 8077
```

[컨테이너 대시보드](http://localhost:8099/)에서 `MODEL_SOURCE=mlflow`, 현재 운영 모델의
`Production` 스테이지와 버전을 확인합니다. 업로드 폴더는 호스트와 공유하며,
MLflow DB·모델 파일·평가 이력은 `solar-runtime` 볼륨에 함께 보관됩니다.
호스트의 MLflow 저장소와는 별도입니다. 호스트 DB에는 호스트 절대 경로가 기록되므로
DB 파일만 복사해서 연결하지 마세요. `docker compose down -v`는 저장 볼륨도 삭제합니다.

시작 시 Production 모델과 스케일러를 즉시 로딩하므로 기존 파일이 손상되면
자동으로 덮어쓰지 않고 시작에 실패합니다. 서버 재시작 시 메모리 감시 창은 초기화됩니다.
새 관측을 평가하면 성능 저하 여부에 따라 자동 재학습합니다. 새 모델이 기존 모델과 기준선보다 좋아야 승격됩니다.

기존 로컬 번들로 실행하려면 아래처럼 모드를 명시할 수 있습니다.
기존 로컬 번들은 읽기 전용으로 마운트합니다. 없으면 runtime/bootstrap-solar에 준비하며
이 모드에서는 운영 재학습이 차단됩니다.

```bash
MODEL_SOURCE=local docker compose -f project/serving_app/docker-compose.yml up -d
```

## 검사

```bash
uv run python -m unittest discover -s tests -v
node tests/dashboard_ui.cjs
uv run ruff check project tests
uv run ruff format --check project tests
```

테스트는 CP949·결측·시간 공백·미래 타깃 누수·스케일러 학습 범위·0 발전량 API·
중복 재생·재학습 cutoff·기준선 게이트·아티팩트 버전 일치·캐시 교체를 확인합니다.
