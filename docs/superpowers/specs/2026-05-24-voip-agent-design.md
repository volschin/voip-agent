# VoIP Agent — Design Spec
*2026-05-24*

## Overview

General-purpose voice AI agent handling inbound and outbound calls via a Fritzbox SIP trunk. Fully local, German-first. LLM-driven conversation with RAG knowledge retrieval and calendar access. No cloud dependencies.

---

## Architecture

### Machine Split

**DGX Spark (GPU host)**
| Service | Model | Port |
|---|---|---|
| `qwen3-asr-server` | Qwen3-ASR-1.7B via vLLM | 8001 |
| `qwen3-tts-server` | Qwen3-TTS via FastAPI | 8002 |
| vLLM (Nous Hermes) | Nous Hermes (OpenAI-compat.) | configurable |
| pgvector | PostgreSQL + pgvector | 5432 |
| embedding sidecar | multilingual-e5-large | 8003 |

**ASUS NUC (agent host)**
| Service | Role |
|---|---|
| Asterisk | SIP registration with Fritzbox, RTP handling, ARI |
| Python agent | Asyncio orchestrator, VAD, pipeline coordination |

### Call Flow

```
Inbound:
  Phone → Fritzbox → SIP → Asterisk ──ARI websocket──► Python Agent
                                                            │
                                               VAD detects end-of-speech
                                                            │
                                            POST → Qwen3-ASR :8001  (German STT)
                                                            │ text
                                            POST → Nous Hermes vLLM  (LLM + tools)
                                                  ├─ RAG tool → pgvector
                                                  └─ Calendar tool → MS Graph API
                                                            │ response text
                                            POST → Qwen3-TTS :8002  (German TTS)
                                                            │ PCM audio
                                    Agent → ARI → Asterisk → Fritzbox → Phone

Outbound:
  Agent decides to call → ARI originate → Asterisk → Fritzbox → PSTN
  → connected → same pipeline as inbound
```

---

## Components

### Asterisk (NUC)
- PJSIP registers to Fritzbox using phone number credentials
- Codec: `alaw` (G.711 A-law — standard German PSTN)
- ARI application named `voip-agent`, websocket on `ws://localhost:8088/ari/events`
- Inbound dialplan: `[from-fritzbox]` → `Stasis(voip-agent)`
- Outbound: ARI `POST /ari/channels` with `endpoint=PJSIP/fritzbox`

### Python Agent (NUC)
- `asyncio`-based; one coroutine per active call
- Key libraries: `ari-py`, `webrtcvad`, `httpx`, `asyncpg`, `msal` (MS Graph)
- Audio in: G.711 aLaw → decode → 16kHz PCM → VAD → speech buffer → ASR
- Audio out: TTS PCM (24kHz) → resample to 8kHz → G.711 aLaw → ARI play

### VAD
- `webrtcvad`, frame size 20ms, aggressiveness level 2
- Trigger ASR after 800ms silence or 15s max buffer
- Suppress input frames while agent is speaking (echo prevention)

### STT
- `POST http://dgx:8001/v1/audio/transcriptions`
- Body: WAV chunk, `language=de`
- Response: `{ "text": "..." }`
- Model: Qwen3-ASR-1.7B (~142ms for 2s clip)

### LLM
- `POST http://dgx:<port>/v1/chat/completions` (OpenAI-compatible vLLM)
- Model: Nous Hermes
- Tools: `rag_lookup(query)`, `calendar_get_events(date_range)`, `calendar_create_event(...)`
- Conversation history maintained per session in process memory

### TTS
- `POST http://dgx:8002/v1/audio/speech`
- Body: `{ "text": "...", "instruct": "Eine warme, natürliche deutsche Stimme." }`
- Response: PCM audio (24kHz, mono)

### RAG
- pgvector on DGX Spark (existing PostgreSQL instance)
- Embedding via `multilingual-e5-large` HTTP sidecar on DGX Spark
- `rag_lookup(query)` → embed query → cosine similarity search → top-k chunks → returned to LLM as tool result

### Calendar
- Backend interface `CalendarBackend` with methods: `get_events(start, end)`, `create_event(...)`
- Current implementation: **Microsoft Graph API** via `msal` + `httpx`
- Future: swap to CalDAV (Nextcloud/Radicale) by replacing implementation class only

---

## Session State

```python
@dataclass
class CallSession:
    call_id: str          # Asterisk channel ID
    caller_id: str        # PSTN number
    history: list[dict]   # OpenAI message format
    created_at: datetime
    state: Literal["listening", "processing", "speaking", "ended"]
```

Sessions live in process memory. No persistence (calls are ephemeral).

### Call Answer
On `StasisStart` (call connected), agent immediately synthesizes and plays a configurable greeting (e.g., "Hallo, wie kann ich Ihnen helfen?") before entering LISTENING state.

### Turn State Machine
```
ANSWER → (play greeting) → LISTENING
LISTENING → (end-of-speech detected) → PROCESSING
PROCESSING → (TTS audio ready) → SPEAKING
SPEAKING → (playback complete) → LISTENING
SPEAKING → (new speech detected) → PROCESSING  ← interruption: cancel playback
any state → (ARI StasisEnd) → ENDED → cleanup
```

---

## Error Handling

| Failure | Behavior |
|---|---|
| ASR timeout / empty result | Play "Ich habe Sie leider nicht verstanden" → retry once → hangup |
| LLM error / timeout | Play "Technischer Fehler, bitte später erneut anrufen" → hangup |
| TTS failure | Play pre-recorded fallback clip → hangup |
| SIP registration lost | Asterisk PJSIP auto-re-registers (retry interval configured) |
| Call dropped mid-session | `StasisEnd` event → session cleanup |
| Tool error (RAG / Calendar) | Tool result contains error string → LLM continues without it |

---

## Outbound Calls

Outbound calls are initiated by the agent itself based on its own reasoning (e.g., triggered via the agent's communication channels). No external REST trigger. The agent calls ARI `originate` with `endpoint=PJSIP/fritzbox`, `callerId`, and `extension`. Once the call is answered, the same inbound pipeline handles the conversation.

---

## Testing

- **Unit**: each HTTP client (ASR, TTS, LLM, RAG, Calendar) tested with `pytest` + `respx` (async HTTP mocking)
- **Integration**: Asterisk loopback test extension — call it from LAN, audio echoes through full pipeline
- **E2E**: `linphone-cli` or `SIPp` dials Asterisk test extension from LAN, validates full voice turn

---

## Technology Summary

| Layer | Technology |
|---|---|
| SIP/RTP | Asterisk + PJSIP, registered to Fritzbox |
| Audio transport | ARI websocket, G.711 aLaw |
| VAD | webrtcvad |
| STT | Qwen3-ASR-1.7B on DGX Spark |
| LLM | Nous Hermes via vLLM on DGX Spark |
| TTS | Qwen3-TTS on DGX Spark |
| RAG store | pgvector (existing PostgreSQL) |
| Embeddings | multilingual-e5-large on DGX Spark |
| Calendar | MS Graph API → CalDAV (future) |
| Agent runtime | Python asyncio on ASUS NUC |
