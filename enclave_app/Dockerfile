FROM python:3.8-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y gcc && \
    pip install --no-cache-dir pandas numpy scikit-learn

CMD ["python3", "vsock_server.py"]
