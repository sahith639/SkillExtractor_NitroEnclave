FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    wget curl git && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
