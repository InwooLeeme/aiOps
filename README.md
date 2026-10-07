# 제주 태양광 발전량 예측과 모델 운영

한국동서발전 제주지역 학습용 CSV의 과거 72시간 발전량·기상정보로 다음 한 시간의
발전량(MWh)을 예측합니다. FastAPI 대시보드, MLflow 모델 등록, 실제 이력 재생,
성능 저하 감시와 명시적 재학습을 제공합니다.

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
  --csv project/data/uploads/solar_jeju_2019_2024.csv --epochs 20
```

위 경로는 현재 작업에서 첨부 원본을 복사한 로컬 경로입니다. 다른 환경에서는
다운로드한 CSV 경로 또는 업로드 응답의 파일명을 사용하세요. 데이터와 생성 모델은 Git에서 제외됩니다.
학습을 다시 실행했다면 예측 서버를 재시작해 새 로컬 번들을 불러옵니다.

## 검증

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check project tests
uv run ruff format --check project tests
```

모델 학습·평가 기간, 결측 처리, API와 Docker 사용법은 [상세 안내](project/README.md)를 참고하세요.
