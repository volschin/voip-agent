FROM docker.io/nvidia/cuda:12.8.0-devel-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git curl ffmpeg libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip numpy

RUN pip install --no-cache-dir "qwen-asr[vllm]" fastapi uvicorn soundfile httpx prometheus-client

ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

COPY server.py /app/server.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=6 \
    CMD python3 -c "import urllib.request,json,sys; \
d=json.loads(urllib.request.urlopen('http://localhost:8000/health').read()); \
sys.exit(0 if d['model_ready'] else 1)"

ENTRYPOINT ["python3", "/app/server.py"]
CMD ["--host", "0.0.0.0", \
     "--port", "8000", \
     "--max-model-len", "4096", \
     "--kv-cache-dtype", "fp8"]
