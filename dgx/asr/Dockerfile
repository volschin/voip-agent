FROM eugr/spark-vllm@sha256:1d861bef8a6c0851140cec2575ebd32342d55bc0fd28ad4c6ca178269e9d1cff

COPY asr/requirements-audio-arm64.lock /tmp/requirements-audio-arm64.lock
COPY asr/patch_vllm_transcription_contract.py /tmp/patch_vllm_transcription_contract.py
RUN python3 -m pip install --no-cache-dir --require-hashes --no-deps \
      -r /tmp/requirements-audio-arm64.lock \
 && python3 /tmp/patch_vllm_transcription_contract.py \
 && python3 -c "import importlib.metadata as m, importlib.util as u, torch; from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionResponse, TranscriptionUsageAudio; assert torch.__version__ == '2.11.0+cu130'; assert torch.version.cuda == '13.0'; assert m.version('vllm') == '0.26.1rc1.dev468+g6b5bec7be.d20260807'; assert m.version('soundfile') == '0.13.1'; assert m.version('av') == '17.0.1'; assert m.version('cffi') == '2.1.1'; assert u.find_spec('vllm.model_executor.models.qwen3_asr'); assert TranscriptionResponse(text='probe', usage=TranscriptionUsageAudio(seconds=1)).model_dump() == {'text': 'probe'}" \
 && rm -f /tmp/patch_vllm_transcription_contract.py

EXPOSE 8001
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python3 -c "import urllib.request,sys,json; r=urllib.request.urlopen('http://127.0.0.1:8001/v1/models',timeout=4); d=json.loads(r.read()); sys.exit(0 if any(model.get('id') == 'qwen3-asr' for model in d.get('data',[])) else 1)" || exit 1
