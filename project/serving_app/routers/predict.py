"""
[Day1 → Day3] 예측 API  —  serving_app/routers/predict.py
【실습용】 ___ (밑줄 3개)만 채우세요. 채울 곳은 [빈칸 N] 으로 표시되어 있습니다.
   ___ 가 남은 채 실행하면 "name '___' is not defined" 에러가 나며, 그 줄이 채울
   곳입니다.

■ 이 파일이 하는 일 (한 줄 요약)
   외부 요청을 받아 모델에게 전달하고, 결과를 돌려주는 "창구"입니다.
   계산은 직접 하지 않고, 모델(model_loader)과 감시 도구(retrain_trigger)에게 맡깁니다.

■ 엔드포인트
   [Day1] POST /predict : 20일치 데이터 → 다음날 종가 1개 (완성 — 읽고 흐름만
   이해하세요)
   [Day3] POST /predict/batch-test  : 긴 가격 목록 → 여러 번 예측 → 드리프트 검사

■ 이 파일의 빈칸 : [빈칸 6]  (batch_test 의 슬라이딩 윈도우)
"""

from data.features import SEQ_LEN  # = 20
from fastapi import APIRouter

from serving_app import model_loader
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.schemas import (
    BatchTestRequest,
    BatchTestResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()

# (Day3) 최근 예측 기록을 모아 두는 목록. 예: [{"predicted": 161.2, "actual": 163.0},
# ...]
# 드리프트 판단은 "최근 21건"(drift_detector.py 의 WINDOW_SIZE)만 보므로 21개까지만
# 유지합니다.
recent_predictions: list[dict] = []

# (Day3) 시뮬레이션은 종가만 보내므로, 거래량은 이 값으로 고정해서 채웁니다.
SIMULATED_VOLUME = 1_200_000


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    [Day1] 다음날 종가 예측  (완성)
    받는 것  : {"sequence": [{"close": 160.0, "volume": 1200000}, ... 20개]}
               20개가 아니면 schemas.py 가 알아서 422 에러를 돌려줍니다.
    돌려줄 것: {"predicted_close": 161.37, "model_version": "v1-local"}

    흐름: 모델 가져오기(get_model) → dict 목록으로 변환 → predict_one → 응답 포장
    핵심 계산은 모두 model_loader.predict_one() 안에 있습니다. ([빈칸 2], [빈칸 3])
    """
    model = model_loader.get_model()
    sequence = [p.model_dump() for p in req.sequence]
    predicted_close = model.predict_one(sequence)
    return PredictResponse(
        predicted_close=round(predicted_close, 2), model_version=model.version
    )


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    [Day3] 드리프트 시뮬레이션
    받는 것 : {"prices": [165.0, 166.2, ... 41개]} (scripts/simulate_drift.py 가 보냄)
    돌려줄 것: {"predictions": [예측값 21개], "drift_check": {"status": "ok"} 또는
    재학습 결과}

    ■ 핵심 아이디어: 슬라이딩 윈도우 (20칸짜리 창문을 한 칸씩 밀기)
      가격 41개가 들어오면, 20개씩 잘라 "그다음 날"을 예측하고 실제 값과 비교합니다.

        i=0 : [p0  ~ p19] → 예측   vs  실제 p20
        i=1 : [p1  ~ p20] → 예측   vs  실제 p21
        ...
        i=20: [p20 ~ p39] → 예측   vs  실제 p40
        → 총 41 - 20 = 21번 예측 = 드리프트 판단에 필요한 21건이 딱 채워집니다.

    확인 방법
      python scripts/simulate_drift.py
        [normal]          drift_check = {'status': 'ok'}
        [drift_injection] drift_check = {'status': 'retrain_triggered', 'promoted':
        True, ...}
      /docs 에서 직접 호출할 때는 predictions 가 (가격 개수 - 20)개인지 확인하세요.
    """
    model = model_loader.get_model()
    predictions: list[float] = []

    prices = req.prices
    for i in range(len(prices) - SEQ_LEN):
        # ════════════════════════════ [빈칸 6] ════════════════════════════
        # i번째 창문(window)의 시작·끝 위치와, 그 창문 바로 다음 날(actual)의 위치를
        # 채우세요. (i 와 SEQ_LEN 으로)
        # (위 docstring 의 그림에서 i=0 일 때 무엇이 창문이고 무엇이 실제 값인지 먼저
        # 확인)
        #
        #   생각해 볼 질문
        #     · 파이썬 슬라이싱 prices[a:b] 는 b 를 포함하나요?
        # · 실제 값을 한 칸 앞(창문의 마지막 날)으로 잡으면, 모델은 무엇을 "맞힌" 셈이
        # 될까요?
        # · 반대로 창문을 한 칸 더 길게 잡아서 실제 값이 창문 안에 들어가면 RMSE는
        # 어떻게 될까요?
        window = prices[i : i + SEQ_LEN]
        sequence = [{"close": p, "volume": SIMULATED_VOLUME} for p in window]
        pred = model.predict_one(sequence)
        actual = prices[i + SEQ_LEN]  # 창문 바로 다음 날
        predictions.append(pred)
        recent_predictions.append({"predicted": pred, "actual": actual})

    # 최근 21건만 남기기 — 오래된 기록까지 섞이면 "지금" 상태를 판단할 수 없습니다.
    # (recent_predictions = ... 로 쓰면 함수 안의 새 변수가 되므로, [:] 로 목록 내용을
    # 바꿉니다)
    recent_predictions[:] = recent_predictions[-21:]  # WINDOW_SIZE 유지

    # 드리프트 판단·재학습은 retrain_trigger.py 가 합니다. 여기서는 넘겨주기만!
    drift_check = check_and_trigger(recent_predictions)
    return BatchTestResponse(predictions=predictions, drift_check=drift_check)
