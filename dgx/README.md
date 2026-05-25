# DGX Spark Services

Run on the DGX Spark (GPU host).

```bash
cd dgx
cp .env.example .env
docker compose up -d
```

## Health checks

```bash
# ASR
curl -s http://localhost:8001/health

# TTS
curl -s http://localhost:8002/health

# Embedding (text-embeddings-inference)
curl -s http://localhost:8003/health
```

Nous Hermes via vLLM is assumed to already be running on its own port.
Set `LLM_BASE_URL` in the agent `.env` to point at it.
