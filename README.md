# HAIC 모델 서빙과 MLOps 실습

Python 3.11과 FastAPI로 최근 20거래일의 종가·거래량을 받아 다음 거래일의 종가를 예측합니다.
Day1 로컬 LSTM 서빙, Day2 MLflow 학습·등록, Day3 드리프트 감지와 재학습 코드까지 작성했습니다.
Day3 전체 흐름은 실행 결과와 로그로 확인하며, 실행·종료 명령은 [project/README.md](project/README.md)를 참고합니다.
아래 명령은 모두 저장소 루트 aiOps에서 실행합니다.

## 설치

```bash
uv sync --locked
```

의존성 기준은 pyproject.toml과 uv.lock입니다. requirement.txt와 project/requirements.txt는
같은 잠금 파일에서 내보낸 pip용 목록입니다. TensorFlow와 MLflow도 설치됩니다.

## Day1 실행

```bash
uv run uvicorn serving_app.main:app --app-dir project --port 8077
```

대시보드는 http://127.0.0.1:8077/, Swagger는 http://127.0.0.1:8077/docs,
모델 준비 상태는 http://127.0.0.1:8077/health에서 확인합니다.
기본값은 MODEL_SOURCE=local과 LOADING_MODE=lazy입니다.
첫 정상 예측에서 모델과 스케일러를 불러오고 이후에는 캐시를 재사용합니다.

기존 서버를 Ctrl+C로 종료한 뒤 다음 명령으로 Eager와 비교합니다.
측정에는 --reload를 사용하지 않습니다.

```bash
LOADING_MODE=eager uv run uvicorn serving_app.main:app --app-dir project --port 8077
```

Lazy의 model_loaded는 첫 예측 전 false, 이후 true입니다. Eager는 서버 기동 후부터 true입니다.
정확히 20개 시퀀스와 close > 0, volume >= 0을 요구합니다. 잘못된 입력에는 422를 반환합니다.
정상 응답에는 predicted_close와 model_version이 포함됩니다.

최근 20거래일을 담은 요청 예제로 바로 예측을 확인할 수 있습니다.

```bash
curl -H 'Content-Type: application/json' \
  --data-binary @project/data/predict_example.json http://127.0.0.1:8077/predict
```

대시보드에서 project/data/sample_haic_prices.csv를 업로드하거나 터미널에서 실행합니다.

```bash
curl -F 'file=@project/data/sample_haic_prices.csv' http://127.0.0.1:8077/data/upload
uv run python project/scripts/train_baseline_v1.py
```

baseline은 최신 업로드 데이터로 100 epoch을 학습하고 평가 RMSE를 출력합니다.
project/serving_app/models의 haic_v1.keras와 scaler.pkl을 갱신합니다.
Day1 학습 후 스케일러를 고정하고 Day2·Day3에서는 다시 fit하지 않습니다.
제공된 모델 파일로 먼저 서빙을 확인할 수도 있습니다.

## Day2 학습과 Production 서빙

```bash
uv run python project/serving_app/train_and_register.py
```

시드 42와 100 epoch으로 학습하고 HAIC 실험에 설정과 RMSE를 기록합니다.
평가 RMSE가 4.00 이하이면 HAIC_Predictor에 등록하고 Production으로 승격합니다.
기준을 넘으면 새 모델 배포를 차단하고 기존 Production을 유지합니다.
학습과 서빙은 project/runtime/mlflow.db를 공유하고, 아티팩트는 project/runtime/mlartifacts에 저장합니다.
제공된 project/mlflow.db는 원 작성자 PC의 경로가 들어 있어 참고 자료로 보존하고 실행에는 사용하지 않습니다.

MLflow 화면은 별도 터미널에서 실행합니다.

```bash
uv run mlflow ui --backend-store-uri sqlite:///project/runtime/mlflow.db --port 5001
```

http://127.0.0.1:5001에서 HAIC 실험과 HAIC_Predictor 모델을 확인합니다.
Production 승격 후 기존 서버를 종료하고 실행합니다.

```bash
MODEL_SOURCE=mlflow LOADING_MODE=eager uv run uvicorn serving_app.main:app --app-dir project --port 8077
```

응답의 model_version은 production입니다. 스케일러는 Day1 로컬 파일을 재사용합니다.
저장소를 바꾸려면 학습·서빙·UI가 모두 같은 MLFLOW_TRACKING_URI를 사용해야 합니다.

## Docker 실행

```bash
docker compose -p aiops-day2 -f project/serving_app/docker-compose.yml up --build
```

컨테이너 주소는 http://127.0.0.1:8099입니다. 빌드 중 baseline과 Day2 학습을 수행하고
컨테이너 내부의 MLflow Production 모델을 Eager로 불러옵니다. 게이트 통과가 필요합니다.

## 검사

```bash
uv run python -m unittest discover -s tests -v
```

테스트는 전처리와 출력 복원, Lazy 캐시, 422 검증, 실제 로컬 모델 예측,
MLflow Production 로딩과 배포 게이트의 경계를 확인합니다.

의존성 파일은 다음 명령으로 갱신합니다.

```bash
uv export --locked --no-dev --no-hashes --no-emit-project -o requirement.txt
cp requirement.txt project/requirements.txt
```

## 구조

```text
aiOps/
  pyproject.toml  uv.lock  requirement.txt
  tests/
  project/
    requirements.txt
    data/                    # 샘플 CSV와 업로드 및 전처리
    scripts/                 # baseline 학습과 Day3 시뮬레이션
    serving_app/
      main.py                # FastAPI 앱과 lifespan
      config.py              # 프로젝트 기준 경로
      tracking.py            # MLflow 저장소 설정
      model_loader.py        # Lazy·Eager 및 로컬·Production 로딩
      train_and_register.py  # Day2 학습과 배포 게이트
      models/                # Day1 모델과 고정 스케일러
      routers/  monitoring/  static/
    runtime/                 # 로컬 DB와 아티팩트  Git 제외
    logs/                    # 실행 로그  Git 제외
```
