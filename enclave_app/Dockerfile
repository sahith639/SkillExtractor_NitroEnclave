# Use Alpine Linux for minimal footprint
FROM alpine:3.18

# Install dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-cryptography \
    py3-pandas \
    py3-numpy \
    py3-scikit-learn

# Copy enclave code
COPY vsock_server.py /vsock_server.py

# Create non-root user for security
RUN adduser -D nitrouser
USER nitrouser

# Start the VSOCK server on launch
CMD ["python3", "/vsock_server.py"]
