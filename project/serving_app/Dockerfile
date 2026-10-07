# 저장소 project/ 디렉터리를 build context로 사용합니다.
# 최초 실행 시 영속 런타임에 모델을 준비하고 이후에는 재사용합니다.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serving_app/ serving_app/
COPY data/ data/
COPY scripts/ scripts/
EXPOSE 8099
ENV MODEL_SOURCE=mlflow
ENV AUTO_PREPARE=1
ENV LOADING_MODE=eager
CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8099"]
