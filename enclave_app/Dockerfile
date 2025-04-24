FROM python:3.8-slim

WORKDIR /app

COPY vsock_server.py .

RUN pip install pandas numpy scikit-learn

CMD ["python", "vsock_server.py"]
