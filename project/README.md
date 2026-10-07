> 아래 실행 명령은 모두 이 README가 있는 `project/` 폴더 기준입니다.
> 빈칸 1~11은 작성 완료했습니다. Day3 전체 흐름은 실행 결과와 로그로 확인합니다.
> 로컬 실행은 부모 폴더의 `pyproject.toml`과 `uv.lock` 및 Python 3.11 가상환경을 사용합니다.
> 로컬 MLflow 결과는 `runtime/mlflow.db`와 `runtime/mlartifacts/`에 저장합니다.

#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : architect@sk.com, audit@korea.ac.kr 임성열 Ph.D.

# HAIC 모델 서빙 및 AIOps 3일 실습 스켈레톤

가상 종목 **HumanAI Corporation(HAIC)** 의 일별 시세로 다음날 종가를 예측하는
**LSTM** 모델을 Day1(서빙) → Day2(MLOps) → Day3(AIOps) 순서로 하나의 서빙 서버 위에
쌓아 올리는 실습 스켈레톤입니다. 데이터는 미리 생성해두지 않고, 대시보드에서 CSV
파일을 업로드하는 방식으로 공급합니다 - 실전에서 "새 데이터가 들어온다"는 상황을
그대로 가정한 것입니다.

완성된 전체 기능(4탭 대시보드, 운영 지표, 알람 등)을 보고 싶다면 별도로 제공되는
**데모 패키지**를 참고하세요. 이 실습 스켈레톤은 실습 난이도를 낮추기 위해서
핵심 루프(업로드 → 학습/서빙 → 드리프트 감지 → 재학습)만
남기고 나머지는 들어내 뒀습니다.

## 데이터 - 대시보드에서 업로드

`data/sample_haic_prices.csv`는 실제 **IBM 2007-01-03 ~ 2009-12-31** 시세를 참조해
만든 예시 데이터(756거래일 ≈ 3년, `Date,Close,Volume` 컬럼)입니다. 상승(2007) →
금융위기로 고점 대비 최대 -45% 폭락(2008) → 회복(2009)까지 실제 시장의 세 국면을
모두 포함하고 있어, Day3 드리프트 감지 실습에서 "정상적인 시장 변동성" vs "이상
드리프트"를 구분하는 근거가 뚜렷합니다.

서버를 띄운 뒤 대시보드(`http://localhost:8077/`)의 업로드 카드에서 이 파일을
그대로 올리면 됩니다. 업로드된 CSV는 `data/uploads/`에 타임스탬프 파일명으로
쌓이고, 학습·시뮬레이션 코드는 항상 **가장 최근에 업로드된 파일**을 사용합니다
(`data/storage.py`의 `latest_upload()`). 여러 번 업로드하면 그때마다 최신 파일로
전환되므로, 다른 시세 CSV(같은 컬럼 형식)로 바꿔 실험해볼 수도 있습니다.

## 모델 아키텍처

최근 20거래일(SEQ_LEN)의 (종가, 거래량) 시퀀스를 입력받아 다음날 종가를 예측하는
3층 LSTM입니다.

```
Input (20, 2)  ->  LSTM(32, return_sequences=True)  ->  LSTM(32, return_sequences=True)
               ->  LSTM(16)  ->  Dense(16, relu)  ->  Dense(1)
```

3년치(~756거래일) 데이터 + SEQ_LEN(20)을 적용하면 학습 시퀀스가 약 590개까지 늘어나,
파라미터(약 1.6만 개) 대비 샘플 비율이 충분히 확보됩니다. 그래서 층을 깊게(LSTM 3층)
쌓았습니다 - CPU로 100 epoch을 학습해도 1분 내외면 끝납니다. (아키텍처 정의는
`serving_app/lstm_model.py`, Day1·Day2가 공유합니다.)

재학습 방식도 유의해서 보세요. Day3에서 드리프트가 감지되면 **처음부터 다시 학습하지
않습니다.** 최근 1개월(21거래일)만으로 LSTM을 스크래치로 학습시키기엔 샘플이 너무
적어 불안정하기 때문에, 이미 전체 데이터로 학습된 **Production 가중치에서 이어서
(warm start) 짧게(10 epoch) fine-tuning**합니다. `serving_app/train_and_register.py`의
`train_and_register()`(Day2, 처음부터 학습)와 `fine_tune()`(Day3, 이어서 학습)이 이 구분입니다.

## 디렉토리 구조

```
project/
├── requirements.txt
├── data/
│   ├── sample_haic_prices.csv  # 대시보드에 업로드해볼 예시 데이터 (IBM 참조 3년치)
│   ├── storage.py              # 업로드된 CSV 중 최신 파일을 찾는 latest_upload()
│   ├── uploads/                # 업로드된 CSV가 쌓이는 곳
│   └── features.py             # 시퀀스 빌더(SEQ_LEN=20) + HAICScaler (전 Day 공용)
├── scripts/                    # 서빙 앱 밖에서 실행하는 실습/시뮬레이션 도구
│   ├── train_baseline_v1.py    # Day1 완료: MLflow 없이 로컬 baseline LSTM 생성
│   └── simulate_drift.py       # Day3 : 정상/드리프트 배치 생성 + 서버로 주입
├── runtime/                    # 로컬 MLflow DB와 모델 아티팩트 (자동 생성)
├── logs/                       # aiops.log 실행 로그 (자동 생성)
└── serving_app/
    ├── main.py                     # Day1 - app 생성, 라우터 등록, 로딩 모드 분기
    ├── schemas.py                  # Day1
    ├── lstm_model.py               # Day1·Day2 공유 아키텍처 정의
    ├── model_loader.py             # Day1·Day2 완료: 로컬·MLflow 로딩
    ├── request_metrics.py          # 실제 예측 요청 기록과 기간별 집계
    ├── train_and_register.py       # Day2 (base 학습) + Day3 (fine-tuning)
    ├── Dockerfile, docker-compose.yml   # Day2 (단일 컨테이너)
    ├── models/
    │   ├── haic_v1.keras           # Day1 로컬 baseline 모델 (train_baseline_v1.py가 생성)
    │   └── scaler.pkl              # Day1~3 공용 정규화 스케일러 (train_baseline_v1.py가 생성)
    ├── routers/
    │   ├── dashboard.py            # 요청 지표·모델 이력·설정·알람 조회
    │   ├── predict.py              # 단일 예측과 Day3 배치 예측
    │   ├── health.py               # Day1
    │   ├── data.py                 # Day2 : CSV 업로드 (완성형)
    │   └── logs.py                 # Day3 : logs/aiops.log 파일 조회 (완성형)
    ├── monitoring/                 # Day3
    │   ├── drift_detector.py       # 최근 21건 RMSE와 감지 기준
    │   └── retrain_trigger.py      # 최신 CSV로 fine-tuning과 승격 결과 확인
    └── static/
        ├── index.html              # 4개 탭의 운영 대시보드
        ├── dashboard.css           # 밝은 테마와 반응형 레이아웃
        └── dashboard.js            # 실제 API 연결과 화면 갱신
```

`serving_app/routers/data.py`, `routers/logs.py`, `static/index.html`은 실습
목표가 아니라 업로드·결과 확인을 위한 배관 코드라 처음부터 완성된 형태로
제공됩니다 - 전부 방금 업로드된 파일이나 로그 파일처럼 이미 존재하는 데이터를
읽거나 저장할 뿐, 가짜 데이터를 만들지 않습니다. 재학습 이력을 MLflow Model
Registry API로 따로 조회하는 대신, `retrain_trigger.py`가 남기는 로그 파일을
그대로 보여주는 쪽을 택했습니다 - 학생이 봐야 할 것은 "재학습이 실제로
일어났다는 증거"이지 레지스트리 조회 API 설계가 아니기 때문입니다.

## 실습용 대시보드

`http://localhost:8077/`은 개발자 대시보드입니다.

- **Dashboard** - 기간별 요청 수·평균 응답시간·성공률, 드리프트 점수,
  7단계 파이프라인, MLflow 등록 이력과 최근 재학습 알람을 표시합니다.
- **Simulation** - 최근 20거래일의 입력을 편집해 단일 예측을 실행하거나,
  정상·드리프트 배치를 `/predict/batch-test`로 전송합니다.
- **Datasets** - 최신 업로드 CSV의 통계를 조회하고 새 CSV를 업로드합니다.
  업로드 데이터가 없으면 제공된 샘플을 미리보기로 표시합니다.
- **System** - 현재 모델 소스·로딩 모드와 실제 학습·드리프트 상수를 조회합니다.

`logs/requests.log`에는 `/predict`와 `/predict/batch-test`의 실제 응답 코드와
응답시간만 기록합니다. 화면의 자동 새로고침 요청은 지표 집계에서 제외합니다.
기간은 5분·1시간·6시간·24시간이며, 요청이 없으면 지표는 0입니다.
최근 알람은 기존 `logs/aiops.log`에서 읽고, 모델 이력의 스테이지는 MLflow의
실제 상태(Production, Archived 등)를 표시합니다.

대시보드 조회 API는 `/metrics/summary`, `/models/overview`, `/events/recent`,
`/data/preview`, `/system/info`입니다. CSV 통계와 예측 입력 예제는 같은 데이터에서
만듭니다. Production 승격 후 서버 캐시의 버전이 다르면 재시작 안내가 표시됩니다.
조회와 새로고침은 모델 재학습이나 캐시 교체를 실행하지 않습니다.

## 실습 시나리오 (Day1 → Day2 → Day3)

**Day1 — HAIC 로컬 baseline LSTM을 FastAPI로 서빙**
Lazy/Eager 로딩 비교, `/predict`·`/health` 동작 확인, `http://localhost:8077/`에서 대시보드로 확인

**Day2 — Day1 서버 + MLflow 학습·레지스트리·컨테이너화**
base 학습(scratch, 100 epoch), RMSE 게이트($4.00) 통과 버전만 Production 승격, Docker로 재현

**Day3 — Day2 서버에 드리프트 감지·자동 재학습 부착**
드리프트 주입 → fine-tuning(warm start, 10 epoch) → 자동 재배포 확인

세 Day의 산출물은 독립적이지 않고 **하나의 `serving_app/`** 위에 순서대로 쌓입니다
(`model_loader.py`가 Day1 로컬 모델 → Day2 MLflow Production 모델로 전환되는 지점이 그 연결고리입니다).
스케일러(`scaler.pkl`)는 Day1에서 한 번 fit한 뒤 Day1~3 내내 그대로 재사용됩니다 -
fine-tuning 시 다시 fit하면 이미 그 스케일로 학습된 기존 가중치와 어긋나기 때문입니다.

## 실행 명령어

### 실행 폴더와 접속 주소

새 터미널을 열 때마다 먼저 `project/` 폴더로 이동합니다.

```bash
cd /Users/iin-u/Desktop/workspace/AI/aiOps/project
```

| 대상 | 대시보드 또는 UI | API 문서 | 상태 확인 |
| --- | --- | --- | --- |
| 로컬 FastAPI | http://localhost:8077/ | http://localhost:8077/docs | http://localhost:8077/health |
| Docker FastAPI | http://localhost:8099/ | http://localhost:8099/docs | http://localhost:8099/health |
| 로컬 MLflow | http://localhost:5001/ | 해당 없음 | UI에서 실험과 등록 모델 확인 |

이미 `project/` 안에 있다면 Compose 경로는 `serving_app/docker-compose.yml`입니다.
`project/serving_app/docker-compose.yml`을 쓰면 `project/project/`를 찾게 되어 실패합니다.

### Docker로 바로 실행하거나 수정 코드 반영하기

Docker Desktop을 실행한 뒤 아래 명령을 사용합니다.

```bash
docker compose -p aiops-day2 -f serving_app/docker-compose.yml up -d --build
docker compose -p aiops-day2 -f serving_app/docker-compose.yml ps
docker compose -p aiops-day2 -f serving_app/docker-compose.yml logs -f --tail 100 serving-app
```

`--build`는 현재 코드를 이미지에 반영하고, `-d`는 서버를 백그라운드로 실행합니다.
빌드 중 샘플 CSV로 baseline과 MLflow 모델을 학습하고 Production 등록까지 진행하므로
시간이 걸릴 수 있습니다. 실행 로그에서 모델 로딩과 서버 기동을 확인한 뒤 대시보드를 엽니다.
로컬에서 모델을 먼저 학습할 필요는 없습니다.

로그 화면의 `Ctrl+C`는 로그 조회만 끝내고 서버는 계속 실행합니다.
서버 상태를 확인하려면 다음 요청을 사용합니다.

```bash
curl http://localhost:8099/health
```

코드를 수정한 뒤에도 `up -d --build`를 다시 실행합니다. 현재 Compose는 로컬 소스를
컨테이너에 연결하는 volume 설정이 없어서 `restart`만으로 수정 코드가 반영되지 않습니다.
화면에 TODO 관련 500 오류가 나오면 서버 로그의 실제 예외를 확인합니다.
`NameError: name '___' is not defined`라면 실행 중인 이미지에 예전 빈칸 코드가 남아 있는지도 확인합니다.

코드 변경 없이 기존 이미지를 실행할 때는 아래 명령을 사용합니다.

```bash
docker compose -p aiops-day2 -f serving_app/docker-compose.yml up -d
```

컨테이너를 종료할 때는 아래 명령을 사용합니다.

```bash
docker compose -p aiops-day2 -f serving_app/docker-compose.yml stop
```

다음 실습까지 데이터와 재학습 결과를 유지하려면 `stop` 후 `up -d`로 기존 컨테이너를
다시 실행합니다. 현재 업로드 CSV와 MLflow 결과는 컨테이너 내부에 저장되며,
volume으로 보존하지 않습니다. 컨테이너를 제거하거나 재생성하기 전에는 필요한 결과를 보관합니다.

### 로컬 환경 준비

Docker 밖에서 FastAPI, 학습 또는 시뮬레이션을 실행하려면 먼저 의존성을 설치합니다.
`uv`는 부모 폴더의 설정을 찾아 Python 3.11 환경을 사용하므로 가상환경을 별도로 활성화하지 않아도 됩니다.

```bash
uv sync --locked
uv run python --version
```

### Day1 로컬 모델과 Lazy Eager 비교

기본 Lazy 서버를 실행합니다. 아래 명령이 실행 중인 터미널은 그대로 둡니다.

```bash
MODEL_SOURCE=local LOADING_MODE=lazy uv run uvicorn serving_app.main:app --port 8077
```

기존 `haic_v1.keras`와 `scaler.pkl`이 있으면 바로 예측을 확인할 수 있습니다.
모델을 처음 만들려면 서버 실행 후 **다른 터미널의 `project/` 폴더에서** CSV를 업로드하고 학습합니다.
baseline 학습은 로컬 모델과 스케일러를 갱신하므로 Day2 이후에는 고정 스케일러를 유지합니다.

```bash
curl -F 'file=@data/sample_haic_prices.csv' http://localhost:8077/data/upload
uv run python scripts/train_baseline_v1.py
```

모델 학습이 끝난 뒤 단일 예측과 상태를 확인합니다. `/predict`는 종가와 거래량을 담은
정확히 20개의 시퀀스를 받으며, 아래 파일에는 실행 가능한 요청 예제가 들어 있습니다.

```bash
curl -H 'Content-Type: application/json' \
  --data-binary @data/predict_example.json http://localhost:8077/predict
curl http://localhost:8077/health
```

서버 터미널에서 `Ctrl+C`로 종료한 뒤 Eager로 다시 실행합니다.
로딩 시간을 비교할 때는 `--reload`를 사용하지 않습니다.

```bash
MODEL_SOURCE=local LOADING_MODE=eager uv run uvicorn serving_app.main:app --port 8077
```

### Day2 로컬 MLflow 학습과 Production 서빙

CSV 업로드와 Day1 스케일러 준비를 마친 뒤 학습합니다.
RMSE가 4.00 이하일 때만 `HAIC_Predictor`의 Production으로 승격합니다.

```bash
uv run python serving_app/train_and_register.py
```

MLflow UI는 **별도 터미널의 `project/` 폴더에서** 실행합니다.

```bash
uv run mlflow ui --backend-store-uri sqlite:///runtime/mlflow.db --port 5001
```

기존 8077 서버를 `Ctrl+C`로 종료한 뒤 Production 모델을 불러오는 서버를 실행합니다.
Day3 로컬 실습에서도 이 명령을 사용합니다.

```bash
MODEL_SOURCE=mlflow LOADING_MODE=eager uv run uvicorn serving_app.main:app --port 8077
```

로컬 학습, 서빙, UI는 같은 `runtime/mlflow.db`를 사용합니다.
Docker는 컨테이너 내부의 별도 MLflow 저장소를 사용하므로 로컬 UI의 기록과 구분합니다.

### Day3 시뮬레이션 실행

서버 기동과 `/health` 응답을 확인한 뒤 **다른 터미널의 `project/` 폴더에서** 실행합니다.
각 명령은 정상 배치와 드리프트 배치를 순서대로 전송합니다.

```bash
# Docker 서버 8099에 전송
uv run python scripts/simulate_drift.py --target container

# 로컬 Production 서버 8077에 전송
uv run python scripts/simulate_drift.py --target local

# 두 서버가 모두 실행 중일 때 같은 배치를 양쪽에 전송
uv run python scripts/simulate_drift.py --target both
```

위 세 명령은 목적에 맞게 하나를 선택합니다. 생성 가격은 실행마다 달라져 정상 배치가
감지 기준을 넘거나 드리프트 배치가 기준을 넘지 않을 수도 있습니다. 실제 응답과 로그로 판단합니다.
재학습은 각 서버에 가장 최근 업로드된 CSV의 마지막 41행을 사용하며,
시뮬레이션 가격을 자동으로 CSV에 저장하지 않습니다.

감지 시 `logs/aiops.log`에 경고와 재학습 로그를 남깁니다. 게이트를 통과한 경우에만
승격 성공 로그가 기록됩니다. 로컬 로그는 다음 명령으로 확인합니다.

```bash
tail -f logs/aiops.log
```

Production이 승격되어도 서버 캐시의 모델은 자동으로 교체되지 않습니다.
로컬은 서버를 `Ctrl+C`로 종료한 뒤 Production 실행 명령으로 다시 실행합니다.
Docker는 같은 컨테이너를 재시작해 내부에 저장된 최신 Production 모델을 로딩합니다.

```bash
docker compose -p aiops-day2 -f serving_app/docker-compose.yml restart serving-app
curl http://localhost:8099/health
```

재시작 후 로그에서 모델 로딩 완료를 확인합니다. 응답의 `model_version`은 `production`이라는
라벨이므로 정확한 숫자 버전은 해당 서버의 MLflow 레지스트리에서도 확인해야 합니다.

### 종료와 포트 확인

로컬 FastAPI, MLflow UI, `tail -f`는 각각 실행 중인 터미널에서 `Ctrl+C`로 종료합니다.
Docker 서버는 앞의 `docker compose ... stop` 명령으로 종료합니다.
포트 충돌이 발생하면 사용 중인 프로세스를 확인하고 해당 서버 터미널에서 종료합니다.

```bash
lsof -nP -iTCP:8077 -sTCP:LISTEN
lsof -nP -iTCP:8099 -sTCP:LISTEN
lsof -nP -iTCP:5001 -sTCP:LISTEN
```

## Day3 구현 상태

- [x] `batch_test()`: 20일 슬라이딩 윈도우 예측과 실제값 연결
- [x] `compute_rmse()`: 예측과 실제값의 제곱 오차 평균에 제곱근 적용
- [x] `is_drift()`: 최소 21건 확인 후 최근 21건의 RMSE가 4달러를 초과하는지 판단
- [x] `check_and_trigger()`: 최신 CSV의 마지막 41행으로 fine-tuning 후 승격 여부 확인
- [x] `send_batch()`: `/predict/batch-test`로 시뮬레이션 배치 전송

## 완료 기준

- [ ] `/data/upload`로 CSV를 올리면 업로드 완료로 표시되는가 (`/data/status`로도 확인 가능)
- [ ] 정상 배치와 드리프트 배치의 RMSE 및 감지 여부를 실제 결과로 비교했는가
- [ ] 감지 시 `logs/aiops.log`에 `[WARN] drift detected`와 `[INFO] retrain triggered`가 기록되는가
- [ ] 평가 RMSE가 4달러 이하일 때만 승격 성공 로그가 기록되고, 실패 시 기존 Production을 유지하는가
- [ ] 모델 재로딩 후 새 Production 버전의 예측을 확인했는가
