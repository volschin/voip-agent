ARG SPARK_BASE=dgx-spark-vllm:midpoint-v023
FROM ${SPARK_BASE}

COPY asr/requirements-audio-arm64.lock /tmp/requirements-audio-arm64.lock
RUN python3 -m pip install --no-cache-dir --require-hashes --no-deps \
      -r /tmp/requirements-audio-arm64.lock \
 && python3 -c "import hashlib, importlib.metadata as m, importlib.util as u, pathlib, torch; adapter = u.find_spec('vllm.model_executor.models.qwen3_asr'); assert adapter and adapter.origin; assert hashlib.sha256(pathlib.Path(adapter.origin).read_bytes()).hexdigest() == 'e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d'; assert torch.__version__ == '2.11.0+cu130'; assert torch.version.cuda == '13.0'; assert m.version('vllm') == '0.23.0'; assert m.version('transformers') == '5.12.1'; assert m.version('flashinfer-python') == '0.6.12'; assert m.version('soundfile') == '0.13.1'; assert m.version('av') == '17.0.1'"

EXPOSE 8001
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python3 -c "import urllib.request,sys,json; r=urllib.request.urlopen('http://127.0.0.1:8001/v1/models',timeout=4); d=json.loads(r.read()); sys.exit(0 if any(model.get('id') == 'qwen3-asr' for model in d.get('data',[])) else 1)" || exit 1
