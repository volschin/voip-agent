# DGX Spark Services

Run on the DGX Spark (GPU host).

```bash
cd dgx
cp .env.example .env
docker network inspect shared_ai_voice >/dev/null 2>&1 \
  || docker network create --internal shared_ai_voice
docker compose up -d
```

This Compose file remains the canonical owner of Qwen3-ASR and Qwen3-TTS. It
keeps the stack-private default network and additionally attaches those two
services to the pre-created neutral `shared_ai_voice` network consumed by
Traefik. The stack does not publish ports 8001 or 8002 after cutover.

ASR loads the already cached revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0` with Hugging Face and Transformers
offline. TTS refuses to start unless PyTorch reports CUDA on `NVIDIA GB10`;
there is no CPU fallback.

`qwen3-tts` builds locally from `./tts` (faster-qwen3-tts + the custom
`/v1/audio/speech/stream` endpoint). First `up` compiles flash-attn —
expect ~10–15 min on the Spark; later boots reuse the cached layer.

## Health checks

```bash
# ASR
docker compose exec -T qwen3-asr \
  python3 -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8001/v1/models')))"

# TTS
docker compose exec -T qwen3-tts \
  python3 -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8002/health')))"

# Embedding (text-embeddings-inference)
curl -s http://localhost:8003/health
```

External ASR/TTS requests use the exact authenticated Traefik paths on
`https://mate.olcon.de`; direct host access to 8001 and 8002 is intentionally
closed. The independently managed Gemma stack is reached through the same
Traefik origin.
