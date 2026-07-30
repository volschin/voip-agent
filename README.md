# voip-agent

German-language voice AI answering agent for inbound calls via FRITZ!Box SIP. All inference
runs on a DGX Spark over LAN.

## Architecture

```
Fritzbox ──SIP/RTP──► PJSUA2 + Python asyncio agent (Docker/NUC)
                              │
                              ▼
                 authenticated HTTPS / Traefik
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              Qwen3-ASR   Gemma/vLLM  Qwen3-TTS
                (STT)       (LLM)       (TTS)
                    │
              ┌─────┴─────┐
              ▼           ▼
          pgvector    MS Graph
           (RAG)     (Calendar)
```

**Voice turn:** negotiated RTP codec → PJSUA2 16 kHz PCM → VAD → Qwen3-ASR → LLM
tool-call loop → Qwen3-TTS → PJSUA2 PCM/RTP

The live turn streams LLM tokens into the German sentence segmenter. Each
completed sentence is synthesized sequentially through the stable
`/v1/audio/speech` whole-WAV endpoint, converted once from 24 to 16 kHz PCM,
then fed through the 300 ms PCM prebuffer to PJSUA2. Up to two completed
sentence segments may wait ahead, within the existing two-second maximum-ahead
playback bound; generation and playback overlap across sentences. The
experimental `/v1/audio/speech/stream` codec path is retained for diagnostics
but is not used for VoIP responses.

Barge-in (caller speaks over the agent) cancels the in-flight producers and
clears buffered PCM. The stable TTS decode loop observes cancellation between
codec steps, releases the exclusive model lock, and lets the replacement turn
start without waiting for the cancelled sentence to finish. The non-streaming
whole-turn path is retained for the greeting and as a fallback.
An interrupted assistant turn is not persisted as conversation history. The
next LLM request receives a one-time system context that the previous response
was not completely delivered and should not be continued unless requested.

**Turn detection:** with `TURN_DETECTION_ENABLED=true` the listen path drops the VAD floor to
~200 ms and confirms each candidate with an **in-process** [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3)
ONNX model. A complete turn flushes immediately; an incomplete one keeps listening. The feature
is on by default; set it to `false` for the legacy fixed 800 ms silence path.

## Hardware

| Host | Role |
|------|------|
| ASUS NUC | Containerized PJSUA2/Python agent |
| DGX Spark | Traefik, Qwen3-ASR, Qwen3-TTS, Gemma via vLLM, ComfyUI; optional embedding/pgvector |

## Prerequisites

- **NUC:** Docker Engine with Compose; no host Asterisk or Python installation
- **DGX:** Docker with NVIDIA runtime, `docker compose`
- **Azure AD app** with `Calendars.ReadWrite` for MS Graph (optional — calendar tool disabled if unconfigured)

## Setup

### 1. DGX — start independently owned AI stacks

The `voice`, `companion-llm`, and `proxy` stacks remain separate owners.
Traefik joins their neutral internal networks and publishes only exact
authenticated routes on the configured `AI_ORIGIN` (default
`https://mate.olcon.de`). Direct ASR/TTS ports 8001
and 8002 stay closed. See [`dgx/README.md`](dgx/README.md) for the voice-stack
contract. The optional embedding service remains outside this Traefik
cutover.

### 2. FRITZ!Box — internal IP telephone

Under **Telephony → Telephony Devices**, create a **LAN/Wi-Fi (IP telephone)**. Assign the
normal incoming number so the handsets and agent ring in parallel. Put its exact username and
password into `.env.pjsip-poc`; this is a local registration, not an external SIP forwarding
target.

### 3. NUC — pgvector schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1024)
);
```

### 4. NUC — agent container

```bash
git clone <repo> voip-agent && cd voip-agent
cp .env.example .env
cp .env.pjsip-poc.example .env.pjsip-poc
# Fill AI/integration values in .env and FRITZ!Box credentials in .env.pjsip-poc.

install -d -m 0700 /home/volsch/voip-agent/secrets
# Install these three files out of band, owned by UID/GID 10001 and mode 0400:
# shared_ai_password, mate_ca.crt, voice_priority_token

docker compose -f compose.pjsip-poc.yml down  # stop the signalling-only PoC
docker compose up --detach --build
docker compose logs --follow voip-agent
```

Expected log output:
```
INFO agent.pjsip SIP registration active=True status=200 OK
```

The three credential/trust mounts must be regular files, never symlinks.
Password and priority-token files may have no group/other permissions. The CA
may be group/world readable but not group/world writable. `DGX_HOST_IP`
controls both the `dgx-spark` and `AI_ORIGIN_HOST` host-gateway entries; set
`AI_ORIGIN`/`AI_ORIGIN_HOST` together when moving the AI boundary to another
hostname.

### 5. Test call

Call the public number assigned to both the handsets and the IP telephone.

Expected:
1. Answer on a handset before the deadline: the FRITZ!Box cancels the agent leg.
2. Leave the next call unanswered: PJSIP accepts it after `ANSWER_DELAY_SECONDS`.
3. You hear the German greeting and can speak to the existing AI pipeline.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRITZBOX_HOST` | `fritz.box` | FRITZ!Box LAN address or hostname |
| `FRITZBOX_SIP_USERNAME` | — | Dedicated LAN/WLAN IP-telephone username |
| `FRITZBOX_SIP_PASSWORD` | — | Dedicated IP-telephone password |
| `PJSIP_LOCAL_PORT` | `5062` | Local SIP port on the Docker host |
| `ANSWER_DELAY_SECONDS` | `20` | Delay before accepting an unanswered call |
| `MAX_CALL_SECONDS` | `900` | Maximum accepted-call duration |
| `AI_ORIGIN` | `https://mate.olcon.de` | Authenticated Traefik origin for STT/TTS/LLM/voice priority (https, no path) |
| `AI_ORIGIN_HOST` | `mate.olcon.de` | Hostname part of `AI_ORIGIN`, mapped to `DGX_HOST_IP` by `compose.yml` |
| `STT_BASE_URL` | `AI_ORIGIN` | Exact authenticated Qwen3-ASR route |
| `TTS_BASE_URL` | `AI_ORIGIN` | Exact authenticated Qwen3-TTS routes |
| `TTS_VOICE_PROFILE` | `shared-female-de-v1` | Server-owned private voice profile ID |
| `LLM_BASE_URL` | `AI_ORIGIN` | Exact authenticated Gemma chat route |
| `LLM_MODEL` | `companion-gemma` | Model name passed to `/v1/chat/completions` |
| `AI_PROXY_USERNAME` | `voip-agent` | Dedicated Traefik BasicAuth account |
| `AI_PROXY_PASSWORD_FILE` | `/run/secrets/shared_ai_password` | Protected client-password file |
| `AI_PROXY_CA_FILE` | `/run/secrets/mate_ca.crt` | Private-CA trust file |
| `VOICE_PRIORITY_TOKEN_FILE` | `/run/secrets/voice_priority_token` | Protected lease-token file |
| `VOICE_PRIORITY_BASE_URL` | `AI_ORIGIN` | Companion voice-priority API |
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
| `GREETING_TEXT` | `Hallo, wie kann ich Ihnen helfen?` | Spoken on answer |
| `LLM_SYSTEM_PROMPT` | German assistant prompt | Injected as system message |

Embedding and pgvector are optional and remain outside the Traefik cutover. If
their independently managed services are unavailable, startup disables RAG
with a warning; ordinary conversation stays available and data tools remain
fail-closed.

## Development

```bash
# Install with dev deps
pip install -e ".[dev]"

# Run tests
pytest -v

# Run single module
pytest tests/test_pipeline.py -v
```

### FRITZ!Box answering-machine signalling PoC

The isolated `compose.pjsip-poc.yml` remains available for SIP-only diagnostics. The production
`compose.yml` uses the same validated delayed-answer behavior and additionally bridges PJSIP
audio into the voice pipeline. See [`docs/pjsip-poc.md`](docs/pjsip-poc.md).
The production cutover is described in
[`docs/pjsip-migration.md`](docs/pjsip-migration.md).

## Module overview

| Module | Responsibility |
|--------|---------------|
| `agent/config.py` | Typed settings from env via pydantic-settings |
| `agent/session.py` | Per-call state machine (ANSWER→LISTENING→PROCESSING→SPEAKING→ENDED) |
| `agent/audio.py` | Stateful PCM resampling, legacy G.711 aLaw codec, WebRTC VAD buffer |
| `agent/stt.py` | Qwen3-ASR HTTP client (16 kHz WAV → transcript) |
| `agent/tts.py` | Qwen3-TTS HTTP client (text → 24 kHz PCM) |
| `agent/turn_detector.py` | In-process Smart Turn v3 ONNX detector (PCM → complete/incomplete) |
| `agent/llm.py` | OpenAI-compat chat + tool-call dispatch loop |
| `agent/tools/rag.py` | pgvector cosine search via asyncpg |
| `agent/tools/calendar.py` | MS Graph calendar (get/create events) |
| `agent/pipeline.py` | One voice turn: VAD flush → STT → LLM → TTS → 16 kHz PCM |
| `agent/conversation.py` | Transport-neutral VAD, turn detection, streaming, and barge-in lifecycle |
| `agent/pjsip.py` | Direct FRITZ!Box SIP/RTP and PJSUA2 PCM media ports |
| `agent/ari.py`, `agent/rtp.py` | Legacy Asterisk rollback adapter; converts pipeline PCM to 8 kHz aLaw |
| `agent/main.py` | Entry point: wire all components and start direct PJSIP transport |

## Latency targets

Per-stage targets for the **non-streaming** whole-turn path (greeting / fallback):

| Stage | Target |
|-------|--------|
| STT (Qwen3-ASR-1.7B) | ~140 ms |
| LLM (Gemma, no tool) | ~500 ms |
| TTS (Qwen3-TTS) | ~1500 ms |
| Total turn | ~2200 ms |

On the live response path, time-to-first-audio includes stable synthesis of the
first completed sentence. Subsequent LLM segmentation, sentence synthesis, and
playback overlap while preserving sentence order. Intelligibility and
once-only ordering take precedence over raw codec-stream time-to-first-chunk.

The 2026-07-29 authenticated on-box cutover measured 282 ms ASR, 861 ms
non-streaming TTS, and 609 ms to the first diagnostic codec-stream chunk for
short German probes. Longer real inference correlated with 94% ASR and 96% TTS GPU
utilization on the NVIDIA GB10. These are acceptance probes, not long-running
percentile benchmarks; there is no cloud, direct-port, or CPU fallback.
