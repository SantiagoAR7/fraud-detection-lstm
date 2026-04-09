FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY data/fraud_autoencoder.pt ./data/
COPY data/scaler.pkl ./data/
COPY data/feature_cols.pkl ./data/
COPY data/best_threshold.pkl ./data/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]