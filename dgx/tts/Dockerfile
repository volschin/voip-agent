# Thin TTS sidecar — reuses the v3 vLLM image's Python/torch/audio stack
# (aarch64-native, has librosa/soundfile already). Adds qwen-tts + FastAPI.
FROM ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3@sha256:6506ebcb79b1bd0d48f8afca127984791f32345333be1be0fef334eaa5a9e23a

# Qwen audio I/O and every package added to the pinned base image are locked
# to the exact versions verified on the GX10 ARM64 target.
COPY tts/apt-packages-arm64.lock /tmp/apt-packages-arm64.lock
RUN apt-get update \
 && xargs -r apt-get install -y --no-install-recommends \
      < /tmp/apt-packages-arm64.lock \
 && rm -rf /var/lib/apt/lists/*

COPY tts/requirements-arm64.lock /tmp/requirements-arm64.lock

# flash-attn for max throughput. The qwen-asr README explicitly recommends this
# install style ("--no-build-isolation"). ARM64 compiles the hash-pinned source
# distribution (about 15 minutes on GX10) and failure must fail the image build.
# Cap MAX_JOBS to keep memory pressure down on Spark's unified RAM during the
# native build; 4 is conservative for a 20-core ARM with 128GB but keeps the
# build stable.
ENV MAX_JOBS=4 \
    FLASH_ATTN_CUDA_ARCHS=120
RUN pip install --no-cache-dir --break-system-packages --no-build-isolation \
      --require-hashes -r /tmp/requirements-arm64.lock

WORKDIR /app
COPY tts/server.py /app/dgx/tts/server.py
COPY tts/runtime.py /app/dgx/tts/runtime.py
COPY tts/profiles.py /app/dgx/tts/profiles.py
COPY tts/clone_runtime.py /app/dgx/tts/clone_runtime.py
COPY tts/api.py /app/dgx/tts/api.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8002
CMD ["python3", "-m", "dgx.tts.server"]
