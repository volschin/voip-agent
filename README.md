# voip-agent

German-language voice AI agent for inbound and outbound calls via Fritzbox SIP. Fully offline — all inference runs on a DGX Spark over LAN.

## Architecture

```
Fritzbox ──SIP──► Asterisk (NUC)
                      │ ARI WebSocket + ExternalMedia RTP
                      ▼
              Python asyncio agent
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    Qwen3-ASR   Nous Hermes   Qwen3-TTS
      (STT)     (LLM/vLLM)     (TTS)
                    │
              ┌─────┴─────┐
              ▼           ▼
          pgvector    MS Graph
           (RAG)     (Calendar)
```

**Voice turn:** aLaw RTP → VAD → 16 kHz PCM → Qwen3-ASR → LLM tool-call loop → Qwen3-TTS → 8 kHz aLaw RTP

The live turn is **streaming**: LLM tokens → German sentence segmenter → Qwen3-TTS `/v1/audio/speech/stream` → resample → aLaw chunks play while generation continues (compute and playback overlap). Audio starts on the first synthesized chunk rather than after the whole turn. Barge-in (caller speaks over the agent) cancels the in-flight turn mid-stream; RTP underruns are filled with comfort silence so the clock never stalls. The non-streaming whole-turn path is retained for the greeting and as a fallback.

**Turn detection (opt-in):** by default the caller's turn ends on a fixed 800 ms silence — a long thinking pause is misread as end-of-turn and the agent talks over them. With `TURN_DETECTION_ENABLED=true` the listen path drops the VAD floor to ~200 ms and confirms each candidate with an **in-process** [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) ONNX model (Whisper-Tiny encoder, 8 MB, ~12 ms CPU, 23 languages incl. German): a complete turn flushes immediately, an incomplete one keeps listening (bounded by `max_speech_ms`). The model is downloaded once (revision-pinned) on first start; inference runs off the event loop. Classifier error or hard cap degrades to the legacy silence flush (fail toward responding). Barge-in is unchanged. **On by default** (set `TURN_DETECTION_ENABLED=false` for the legacy 800 ms path); German verified offline at ~95% on a synthetic test split — a real-call smoke test on the live trunk is still recommended.

## Hardware

| Host | Role |
|------|------|
| ASUS NUC | Asterisk 20, Python agent |
| DGX Spark | Qwen3-ASR, Qwen3-TTS, multilingual-e5-large embedding, Nous Hermes via vLLM, pgvector |

## Prerequisites

- **NUC:** Asterisk 20 (`apt install asterisk`), Python 3.12+
- **DGX:** Docker with NVIDIA runtime, `docker compose`
- **Azure AD app** with `Calendars.ReadWrite` for MS Graph (optional — calendar tool disabled if unconfigured)

## Setup

### 1. DGX — start AI services

```bash
cd dgx
cp .env.example .env   # adjust model names if needed
docker compose up -d
# wait ~60s for model loads
docker compose logs -f qwen3-asr   # watch for "Uvicorn running"
```

Health checks:
```bash
curl -s http://dgx-spark:8001/health   # ASR
curl -s http://dgx-spark:8002/health   # TTS
curl -s http://dgx-spark:8003/health   # embedding
```

Nous Hermes via vLLM runs separately — set `LLM_BASE_URL` accordingly.

### 2. NUC — Asterisk config

```bash
# Fill in Fritzbox credentials
vim asterisk/pjsip.conf    # replace <FRITZBOX_IP>, <SIP_USER>, <SIP_PASSWORD>, <YOUR_PHONE_NUMBER>

sudo cp asterisk/pjsip.conf asterisk/extensions.conf asterisk/ari.conf /etc/asterisk/
sudo asterisk -rx "core reload"

# Verify Fritzbox registration
sudo asterisk -rx "pjsip show registrations"
# Expected: fritzbox   Registered

# Verify ARI
curl -s -u voip-agent:changeme http://localhost:8088/ari/applications
```

### 3. NUC — pgvector schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1024)
);
```

### 4. NUC — agent

```bash
git clone <repo> voip-agent && cd voip-agent
python -m venv venv && source venv/bin/activate
pip install -e .

cp .env.example .env
vim .env   # fill in all values

python -m agent.main
```

Expected log output:
```
INFO agent.ari Connecting to ARI at ws://localhost:8088/ari/events?...
```

### 5. Test call

Dial `9999` from any phone on the Fritzbox network.

Expected:
1. Log: `Call ch-xxx from <number> ready`
2. You hear the German greeting
3. Speak a question → German response within ~3 s

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARI_BASE_URL` | `http://localhost:8088` | Asterisk ARI HTTP base |
| `ARI_USERNAME` | `voip-agent` | ARI user (matches `asterisk/ari.conf`) |
| `ARI_PASSWORD` | `changeme` | ARI password |
| `RTP_PORT` | `5000` | First UDP port for ExternalMedia RTP (increments by 2 per call) |
| `STT_BASE_URL` | `http://dgx-spark:8001` | Qwen3-ASR |
| `TTS_BASE_URL` | `http://dgx-spark:8002` | Qwen3-TTS |
| `LLM_BASE_URL` | `http://dgx-spark:8000` | Nous Hermes via vLLM |
| `LLM_MODEL` | `nous-hermes` | Model name passed to `/v1/chat/completions` |
| `EMBEDDING_BASE_URL` | `http://dgx-spark:8003` | multilingual-e5-large |
| `TURN_DETECTION_ENABLED` | `true` | Smart Turn v3 in-process end-of-turn gating (on by default; set `false` for the legacy 800 ms path) |
| `TURN_COMPLETE_THRESHOLD` | `0.70` | `prob` ≥ this ⇒ turn complete (0.70 biases toward fewer cut-ins on telephony) |
| `TURN_VAD_SILENCE_MS` | `200` | Lowered VAD silence floor for the turn-end candidate |
| `TURN_MODEL_REPO` | `pipecat-ai/smart-turn-v3` | HF repo for the ONNX model |
| `TURN_MODEL_FILENAME` | `smart-turn-v3.2-cpu.onnx` | Model file (use `-gpu.onnx` with an OpenVINO/CUDA provider) |
| `TURN_MODEL_REVISION` | `f766f81…` | Pinned HF revision |
| `TURN_ONNX_PROVIDERS` | `CPUExecutionProvider` | Comma-separated onnxruntime execution providers |
| `DB_DSN` | — | asyncpg DSN for pgvector |
| `AZURE_TENANT_ID` | — | MS Graph auth |
| `AZURE_CLIENT_ID` | — | MS Graph auth |
| `AZURE_CLIENT_SECRET` | — | MS Graph auth |
| `CALENDAR_USER_EMAIL` | — | Calendar owner |
| `CALLER_ID` | `+49123456789` | CLI shown on outbound calls |
| `GREETING_TEXT` | `Hallo, wie kann ich Ihnen helfen?` | Spoken on answer |
| `LLM_SYSTEM_PROMPT` | German assistant prompt | Injected as system message |

## Development

```bash
# Install with dev deps
pip install -e ".[dev]"

# Run tests
pytest -v

# Run single module
pytest tests/test_pipeline.py -v
```

## Module overview

| Module | Responsibility |
|--------|---------------|
| `agent/config.py` | Typed settings from env via pydantic-settings |
| `agent/session.py` | Per-call state machine (ANSWER→LISTENING→PROCESSING→SPEAKING→ENDED) |
| `agent/audio.py` | G.711 aLaw codec, 8↔16/24 kHz resampling, WebRTC VAD buffer |
| `agent/stt.py` | Qwen3-ASR HTTP client (16 kHz WAV → transcript) |
| `agent/tts.py` | Qwen3-TTS HTTP client (text → 24 kHz PCM) |
| `agent/turn_detector.py` | In-process Smart Turn v3 ONNX detector (PCM → complete/incomplete) |
| `agent/llm.py` | OpenAI-compat chat + tool-call dispatch loop |
| `agent/tools/rag.py` | pgvector cosine search via asyncpg |
| `agent/tools/calendar.py` | MS Graph calendar (get/create events) |
| `agent/pipeline.py` | One voice turn: VAD flush → STT → LLM → TTS → aLaw |
| `agent/rtp.py` | asyncio UDP DatagramProtocol, RTP parse/build, paced streaming |
| `agent/ari.py` | ARI WebSocket events, ExternalMedia bridge, VAD-driven turns, optional Smart Turn v3 gating, barge-in |
| `agent/main.py` | Entry point: wire all components, start ARI event loop |

## Latency targets

Per-stage targets for the **non-streaming** whole-turn path (greeting / fallback):

| Stage | Target |
|-------|--------|
| STT (Qwen3-ASR-1.7B) | ~140 ms |
| LLM (Nous Hermes, no tool) | ~500 ms |
| TTS (Qwen3-TTS) | ~1500 ms |
| Total turn | ~2200 ms |

On the streaming live path these stages overlap, so the metric that matters is
**time-to-first-audio** (STT + first LLM tokens + first TTS chunk), not the
full-turn sum — the caller hears the start of the reply while the rest is still
generating. On-box streaming numbers are not yet measured (see below).

If TTS exceeds 3 s, switch to the 0.6B variant in `dgx/.env`.

> **Status:** the streaming pipeline (#3, faster-qwen3-tts) is implemented and
> unit-tested but **not yet verified on-box** — the DGX TTS server is built
> locally from `dgx/tts/` and its streaming endpoint has not been wire-tested
> against the live agent.
