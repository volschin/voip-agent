FROM python:3.14-slim-bookworm AS pjsip-builder

ARG PJSIP_COMMIT=5a457451fa2712ba18e12b01738e8ff3af2b26fd

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libssl-dev \
        pkg-config \
        swig \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir setuptools wheel

RUN python -m pip wheel \
    --no-cache-dir \
    --wheel-dir /tmp/python-wheels \
    "webrtcvad-wheels>=2.0.14"

WORKDIR /src/pjproject
RUN git init \
    && git remote add origin https://github.com/pjsip/pjproject.git \
    && git fetch --depth 1 origin "${PJSIP_COMMIT}" \
    && git checkout --detach FETCH_HEAD

RUN CFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC" \
        ./configure \
        --disable-sound \
        --disable-video \
        --disable-ffmpeg \
        --disable-v4l2 \
        --disable-sdl \
        --disable-libyuv \
        --disable-opus \
    && make dep \
    && make --jobs="$(nproc)" \
    && make --directory=pjsip-apps/src/swig/python wheel


FROM python:3.14-slim-bookworm AS runtime

ARG SOURCE_REVISION
LABEL org.opencontainers.image.source="https://github.com/volschin/voip-agent" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libssl3 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin voip-agent \
    && install -d -o voip-agent -g voip-agent /home/voip-agent/.cache/huggingface

COPY --from=pjsip-builder /src/pjproject/pjsip-apps/src/swig/python/dist/*.whl /tmp/
COPY --from=pjsip-builder /tmp/python-wheels/*.whl /tmp/

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY agent /app/agent

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install /tmp/*.whl . \
    && rm -f /tmp/*.whl \
    && python -c "import agent.main, pjsua2; print('voip-agent runtime imports OK')"

ENV HF_HOME=/opt/voip-model-cache

RUN python -c "from agent.turn_detector import TurnDetector; TurnDetector(model_repo='pipecat-ai/smart-turn-v3', model_filename='smart-turn-v3.2-cpu.onnx', model_revision='f766f81d3cfdf7737ac64aad813d91bbfd56bf93', providers=['CPUExecutionProvider'])" \
    && chmod -R a+rX /opt/voip-model-cache

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1

USER voip-agent

STOPSIGNAL SIGTERM
CMD ["python", "-m", "agent.main"]
