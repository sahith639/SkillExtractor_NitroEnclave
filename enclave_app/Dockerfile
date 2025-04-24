FROM python:3.8-slim

WORKDIR /app
COPY ./vsock_server.py /app/vsock_server.py


# Install Python deps
RUN apt-get update && apt-get install -y \
    gcc \
    libstdc++6 \
    libtinfo5 \
    libffi-dev \
    libedit2 \
    socat \
    && pip install --no-cache-dir pandas numpy scikit-learn


#  Enable enclave console support
RUN apt-get install -y socat

CMD ["sh", "-c", "python3 vsock_server.py || (echo 'Python crashed' && sleep 99999)"]
