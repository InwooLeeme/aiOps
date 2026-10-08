# 태양광 발전량 예측 서빙 + AIOps 대시보드 (단일 컨테이너)
# 빌드는 miniProject 루트에서:  docker build -f serving_app/Dockerfile -t solar-aiops .
# 빌드 시점에 data/uploads/ 의 발전량 CSV 와 data/weather_jeju_forecast.csv 로 base 모델(v1)을 학습·등록한다.
# (tensorflow 때문에 이미지가 1GB대이고 빌드에 몇 분 걸린다)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY serving_app/ serving_app/
COPY scripts/ scripts/

RUN python serving_app/train_and_register.py

EXPOSE 8099
ENV MODEL_SOURCE=mlflow
ENV LOADING_MODE=eager
CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8099"]
