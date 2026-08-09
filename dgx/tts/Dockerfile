ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.3.1-devel-ubuntu26.04@sha256:da3989b0ea8e8b4b241711edd5823bc1cc83d05a01882258bddad84d7394c37e
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.3.1-base-ubuntu26.04@sha256:f65b4f0b65bbf2e0a2520cebaec3120bf4ed110aecc3e7dcab3b11cb508a0484

FROM ${CUDA_DEVEL_IMAGE} AS builder

COPY tts/apt-builder-packages-arm64.lock /tmp/apt-builder-packages-arm64.lock
RUN apt-get update \
 && xargs -r apt-get install -y --no-install-recommends \
      < /tmp/apt-builder-packages-arm64.lock \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/tts-venv
ENV PATH=/opt/tts-venv/bin:$PATH \
    MAX_JOBS=4 \
    FLASH_ATTN_CUDA_ARCHS=120

COPY tts/requirements-slim-arm64.lock /tmp/requirements-slim-arm64.lock
RUN pip install --no-cache-dir --require-hashes \
      -r /tmp/requirements-slim-arm64.lock

COPY tts/tts-packages-arm64.lock /tmp/tts-packages-arm64.lock
RUN pip install --no-cache-dir --require-hashes --no-deps \
      -r /tmp/tts-packages-arm64.lock

COPY tts/flash-attn-arm64.lock /tmp/flash-attn-arm64.lock
RUN pip install --no-cache-dir --require-hashes --no-deps --no-build-isolation \
      -r /tmp/flash-attn-arm64.lock

COPY tts/torchaudio-kaldi-compat-arm64.lock /tmp/torchaudio-kaldi-compat-arm64.lock
RUN pip download --no-cache-dir --require-hashes --no-deps \
      -r /tmp/torchaudio-kaldi-compat-arm64.lock \
      --dest /tmp/torchaudio-kaldi \
 && package_root="$(python3 -c \
      'from pathlib import Path; import qwen_tts; print(Path(qwen_tts.__file__).resolve().parent)')" \
 && kaldi_wheel="$(find /tmp/torchaudio-kaldi -name 'torchaudio-*.whl' -print -quit)" \
 && python3 -c "from pathlib import Path; from zipfile import ZipFile; import sys; wheel, root = sys.argv[1:]; source = ZipFile(wheel).read('torchaudio/compliance/kaldi.py').decode(); assert source.count('import torchaudio\\n') == 1; Path(root, 'core/tokenizer_25hz/vq/kaldi_compat.py').write_text(source.replace('import torchaudio\\n', ''), encoding='utf-8')" \
      "$kaldi_wheel" "$package_root" \
 && python3 -c "from pathlib import Path; import sys; path = Path(sys.argv[1]) / 'core/tokenizer_25hz/vq/speech_vq.py'; source = path.read_text(encoding='utf-8'); updated = source.replace('import torchaudio.compliance.kaldi as kaldi', 'from . import kaldi_compat as kaldi'); assert updated != source; path.write_text(updated, encoding='utf-8')" \
      "$package_root" \
 && rm -rf /tmp/torchaudio-kaldi

RUN python3 -c 'import faster_qwen3_tts, qwen_tts, torch; import qwen_tts.core; import qwen_tts.inference.qwen3_tts_model; from qwen_tts.core.tokenizer_25hz.vq import kaldi_compat; features = kaldi_compat.fbank(torch.ones(1, 400), num_mel_bins=80, dither=0, sample_frequency=16000); assert features.shape == (1, 80); assert torch.isfinite(features).all()'

# faster-qwen3-tts 0.2.6 has no cancellation hook in its stable decode loop.
# Apply a fail-closed, version-bound patch so barge-in can stop between codec
# steps instead of leaving the exclusive model lock held until full completion.
COPY tts/faster-qwen3-tts-0.2.6-cooperative-cancel.patch /tmp/
COPY tts/verify_cooperative_cancel_patch.py /tmp/
RUN package_root="$(python3 -c \
      'from pathlib import Path; import faster_qwen3_tts; print(Path(faster_qwen3_tts.__file__).resolve().parent.parent)')" \
 && cd "$package_root" \
 && patch --batch --forward --fuzz=0 -p1 \
      < /tmp/faster-qwen3-tts-0.2.6-cooperative-cancel.patch \
 && python3 /tmp/verify_cooperative_cancel_patch.py \
 && rm /tmp/faster-qwen3-tts-0.2.6-cooperative-cancel.patch \
      /tmp/verify_cooperative_cancel_patch.py

FROM ${CUDA_RUNTIME_IMAGE} AS runtime

COPY tts/apt-runtime-packages-arm64.lock /tmp/apt-runtime-packages-arm64.lock
RUN apt-get update \
 && xargs -r apt-get install -y --no-install-recommends \
      < /tmp/apt-runtime-packages-arm64.lock \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/tts-venv /opt/tts-venv
ENV PATH=/opt/tts-venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY tts/server.py /app/dgx/tts/server.py
COPY tts/runtime.py /app/dgx/tts/runtime.py
COPY tts/profiles.py /app/dgx/tts/profiles.py
COPY tts/clone_runtime.py /app/dgx/tts/clone_runtime.py
COPY tts/api.py /app/dgx/tts/api.py

EXPOSE 8002
CMD ["python3", "-m", "dgx.tts.server"]
