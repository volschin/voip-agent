FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-venv sox \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN useradd -m appuser \
    && chown -R appuser:appuser /workspace
USER appuser

RUN python -m venv /workspace/venv --system-site-packages
ENV PATH=/workspace/venv/bin:$PATH

RUN pip install -U qwen-tts

# flash-attn: prebuilt wheel for x86_64, source build for arm64
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.6.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl; \
    else \
        pip install flash-attn --no-build-isolation; \
    fi

ENV HF_HOME=/workspace/hf \
    TRANSFORMERS_CACHE=/workspace/hf \
    HF_HUB_CACHE=/workspace/hf

COPY start_demo_app.py /workspace/start_demo_app.py

ENTRYPOINT ["/workspace/venv/bin/python", "/workspace/start_demo_app.py"]
CMD ["--custom-voice", "--host", "0.0.0.0", "--port", "8000"]
