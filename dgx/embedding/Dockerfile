FROM nvcr.io/nvidia/cuda:12.9.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    fastapi uvicorn[standard] \
    torch==2.3.1 \
    sentence-transformers \
    numpy

WORKDIR /app
COPY server.py .

EXPOSE 8003
CMD ["python3", "server.py"]
