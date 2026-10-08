# 제주 태양광 발전량 예측과 모델 운영

한국동서발전 제주지역 학습용 CSV의 완료된 14일 발전량·기상정보로 다음 날짜의
총발전량(MWh)을 예측합니다. FastAPI 대시보드, MLflow 모델 등록, 예측·실제값 연결,
성능 저하 감시와 자동 재학습·검증 후 운영 모델 교체를 제공합니다.

## 실행

저장소 루트에서 Python 3.11 환경을 준비합니다.

```bash
uv sync --locked
uv run uvicorn serving_app.main:app --app-dir project --port 8077
```

[대시보드](http://localhost:8077/)에서 한국동서발전 원본 CSV를 업로드합니다.
UTF-8·UTF-8 BOM·CP949를 지원합니다. 기존 주가 CSV와 모델은 사용하지 않습니다.
모델이 없으면 화면과 데이터 조회는 가능하고 예측 API는 학습 안내와 503을 반환합니다.

별도 터미널에서 **업로드한 실제 경로**를 지정해 학습합니다.

```bash
uv run python project/scripts/train_baseline_v1.py \
  --csv project/data/aggregated/jeju_solar_daily_2019_2024.csv --epochs 20
```

위 경로는 제공받은 원본을 일별로 집계한 CSV입니다. 다른 환경에서는
다운로드한 CSV 경로 또는 업로드 응답의 파일명을 사용하세요. 업로드와 생성 모델은 Git에서 제외되며, 제공받은 제주 원본 샘플은 포함됩니다.
학습을 다시 실행했다면 예측 서버를 재시작해 새 로컬 번들을 불러옵니다.

업로드가 없으면 제주 샘플을 사용하고 `--csv`도 생략할 수 있습니다.
자동 학습·등록을 포함해 실행하려면 다음 명령을 사용하세요.

```bash
docker compose -f project/serving_app/docker-compose.yml up --build -d
```

빈 환경에서는 최초 학습·검증에 시간이 필요합니다. 이미 일별 JejuSolarDailyPredictor Production 모델이 있으면
재사용합니다. 대시보드의 정상/드리프트 버튼은 검증 통과 시 실제 운영 모델을 교체합니다.

## 검증

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check project tests
uv run ruff format --check project tests
```

모델 학습·평가 기간, 결측 처리, API와 Docker 사용법은 [상세 안내](project/README.md)를 참고하세요.
