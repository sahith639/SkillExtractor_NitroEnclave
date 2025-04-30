FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libsndfile1 \
    wget curl git \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy==1.23.1 \
    && pip install --no-cache-dir torch==1.12.1 \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
