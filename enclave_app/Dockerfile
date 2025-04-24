FROM python:3.8-slim

WORKDIR /app
COPY . .

# Install Python deps
RUN apt-get update && apt-get install -y gcc \
    libstdc++6 \
    libtinfo5 \
    libffi6 \
    libedit2 \
    && pip install --no-cache-dir pandas numpy scikit-learn

#  Enable enclave console support
RUN apt-get install -y socat

CMD ["sh", "-c", "python3 vsock_server.py || (echo 'Python crashed' && sleep 99999)"]
