FROM python:3.11-slim

WORKDIR /app

# Install OS dependencies for scipy / pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    LOG_DIR=/app/logs \
    DB_PATH=/data/trading.db

VOLUME ["/data", "/app/logs"]

# Default: run the trading bot (override with docker run ... optimize / validate etc.)
CMD ["python", "main.py", "trade"]
