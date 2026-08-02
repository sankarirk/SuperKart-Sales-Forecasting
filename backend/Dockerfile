FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY sales_prediction_api.py .
COPY superkart_sales_pipeline.joblib .

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --workers 2 --bind 0.0.0.0:${PORT:-10000} sales_prediction_api:app"]