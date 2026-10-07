# 저장소 project/ 디렉터리를 build context로 사용합니다.
# 모델 학습은 명시적으로 실행하고, 검증된 태양광 번들을 마운트합니다.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serving_app/ serving_app/
COPY data/ data/
COPY scripts/ scripts/
EXPOSE 8099
ENV MODEL_SOURCE=local
ENV LOADING_MODE=lazy
CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8099"]
