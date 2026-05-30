# voip-agent — Claude context

German-language voice AI agent. Inbound/outbound SIP calls via Fritzbox → Asterisk ARI → Python asyncio agent → Qwen3-ASR/TTS + Nous Hermes on DGX Spark.

## Commands

```bash
# Run tests
venv/bin/pytest -v

# Single module
venv/bin/pytest tests/test_pipeline.py -v

# Type check (if mypy added)
venv/bin/mypy agent/

# Import sanity check
venv/bin/python -c "from agent.main import main; print('OK')"

# Start agent
venv/bin/python -m agent.main
```

## Project structure

```
agent/
  config.py       # pydantic-settings, all env vars
  session.py      # CallSession dataclass + SessionState FSM
  audio.py        # alaw_encode/decode, resample_*, VadBuffer
  stt.py          # SttClient → Qwen3-ASR /v1/audio/transcriptions
  tts.py          # TtsClient → Qwen3-TTS /v1/audio/speech
  llm.py          # LlmClient → OpenAI-compat /v1/chat/completions + tool dispatch
  pipeline.py     # VoicePipeline: process_turn + synthesize_alaw
  rtp.py          # RtpServer (asyncio.DatagramProtocol) + build/parse helpers
  ari.py          # AriClient: WebSocket event loop + ExternalMedia bridge
  main.py         # Entry point: wire everything, asyncio.run(main())
  tools/
    rag.py        # RagTool: embed → pgvector cosine search
    calendar.py   # MSGraphCalendar + CalendarBackend protocol

asterisk/         # Config files to copy to /etc/asterisk/
dgx/              # Docker Compose for DGX Spark AI services
tests/            # pytest, asyncio_mode=auto, respx for HTTP mocking
```

## Architecture invariants

- **State machine:** greeting path `ANSWER → SPEAKING → LISTENING`, then loop `LISTENING → PROCESSING → SPEAKING → LISTENING`. Any state → `ENDED`. `PROCESSING → LISTENING` allowed (error fallback). `SPEAKING` is reachable only from `ANSWER` (greeting) or `PROCESSING` — never from `LISTENING`. Transitions enforced by `session.transition()` — raises ValueError on invalid.
- **State ownership:** `pipeline.process_turn` owns `PROCESSING` only — it runs STT+LLM+synth and returns audio with the session still in `PROCESSING`. `AriClient._play_audio` drives `PROCESSING → SPEAKING → LISTENING` (and `ANSWER → SPEAKING` for the greeting) around actual RTP playback.
- **Audio path in:** Asterisk ExternalMedia → UDP RTP → `RtpServer.datagram_received` → `on_audio` callback → `VadBuffer.add_frame` → on speech end → `pipeline.process_turn`
- **Audio path out:** `pipeline.synthesize_alaw` → `rtp_server.stream_audio` (paced 20 ms frames) → RTP UDP → Asterisk
- **Barge-in:** Detected when VAD fires during `SPEAKING`. Cancels the active `_playback_tasks[channel_id]` asyncio.Task.
- **Greeting:** Non-blocking — launched via `asyncio.create_task(_play_audio(...))` so the ARI WebSocket reader is never blocked.
- **VoicePipeline callables:** Constructor takes `stt`, `llm`, `tts` as bare callables (not objects). In main.py: `stt=stt_client.transcribe`, `llm=llm_client.complete`, `tts=tts_client.synthesize`.

## Tech stack

- Python 3.12+ (venv at `venv/`; currently running 3.14)
- `pydantic-settings` — config
- `httpx` — async HTTP for all AI service calls
- `websockets` — ARI WebSocket
- `asyncpg` — pgvector queries
- `webrtcvad` — VAD (aggressiveness=2)
- `scipy` / `numpy` — resampling
- `audioop-lts` — G.711 aLaw codec (Python 3.13+ compat shim)
- `msal` — MS Graph auth (blocking, wrapped in `asyncio.to_thread`)
- `respx` — HTTP mock in tests
- `pytest-asyncio` (asyncio_mode=auto)

## Key design decisions

- **RTP port allocation:** Starts at `rtp_port` (default 5000), increments by 2 per call, wraps at 65534.
- **VAD frame size:** 20 ms at 16 kHz = 320 samples. Hard cap at `max_speech_ms` (default 15 s) to prevent unbounded buffer growth.
- **TTS output:** server returns a 24 kHz PCM_16 **WAV** → `pipeline._decode_wav` (strips the RIFF header) → `resample_24k_to_8k` → aLaw encode → RTP. STT input: 8 kHz aLaw → `alaw_decode` → `resample_8k_to_16k` → WAV → Qwen3-ASR.
- **TTS voice field:** the client sends `voice` (free-form German voice description); the server maps `voice` → qwen-tts `instruct`. Do not send `instruct` — the server ignores unknown fields.
- **Embedding API:** RAG calls `POST /v1/embeddings` with `{"input": text}` and reads `data[0].embedding` (OpenAI-shaped). Not `/embed`.
- **MS Graph timezone:** `Prefer: outlook.timezone="Europe/Berlin"` header on calendarView requests. `create_event` uses `timeZone: Europe/Berlin`.
- **LLM tools:** `rag_lookup`, `calendar_get_events`, `calendar_create_event` defined in `agent/llm.py:TOOLS`.
- **Tool authorization (fail closed):** tools are offered to the LLM only when the caller number is in `trusted_callers` (`TRUSTED_CALLERS`). Empty allowlist = tools off for everyone. `process_turn` passes `session.caller_id` to `llm.complete(messages, caller_id)`. Unknown callers can still converse, but get no RAG/calendar access (closes read-side exfiltration). Writes need `calendar_write_enabled=True` AND a `confirmed=true` tool arg on top. Note: `confirmed` is model-set so it is a cooperative-call correctness guard, not an adversarial control — the real write boundary is the disabled-by-default flag. The tool loop is capped at `max_tool_rounds`.

## Testing conventions

- No real network/DB calls in tests — mock with `respx` (HTTP) or `unittest.mock.AsyncMock` (callables)
- `asyncio_mode = "auto"` in pyproject.toml — no `@pytest.mark.asyncio` needed
- `tests/conftest.py` provides `settings` fixture with all fields filled
- All async tests are plain `async def` functions

## Gotchas

- `audioop` removed in Python 3.13 — project uses `audioop-lts` backport
- MSAL `acquire_token_for_client` is sync/blocking — always wrap in `asyncio.to_thread`
- `VoicePipeline` calls `self._stt(bytes)` not `self._stt.transcribe(bytes)` — pass bound methods
- `RtpServer._transport` is private — use `rtp.close()` from outside
