FROM python:3.8-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y gcc && \
    pip install --no-cache-dir pandas numpy scikit-learn

CMD ["sh", "-c", "python3 vsock_server.py || echo ' Python crashed'; sleep infinity"]

