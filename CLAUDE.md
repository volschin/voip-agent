# voip-agent — Claude context

German-language voice AI agent. Inbound SIP calls use direct FRITZ!Box PJSUA2 media through
the Python asyncio agent and authenticated Qwen3-ASR/TTS plus Gemma services.

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
  pipeline.py     # VoicePipeline: stable sentence synthesis to 16 kHz PCM
  conversation.py # Transport-neutral VAD, turn, playback, and barge-in lifecycle
  pjsip.py        # Production direct FRITZ!Box PJSUA2 transport
  rtp.py          # Legacy ARI rollback RTP helpers
  ari.py          # Legacy Asterisk rollback adapter
  main.py         # Entry point: wire everything, asyncio.run(main())
  tools/
    rag.py        # RagTool: embed → pgvector cosine search
    calendar.py   # MSGraphCalendar + CalendarBackend protocol

asterisk/         # Legacy rollback templates
dgx/              # Docker Compose for DGX Spark AI services
tests/            # pytest, asyncio_mode=auto, respx for HTTP mocking
```

## Architecture invariants

- **State machine:** greeting path `ANSWER → SPEAKING → LISTENING`, then loop `LISTENING → PROCESSING → SPEAKING → LISTENING`. Any state → `ENDED`. `PROCESSING → LISTENING` allowed (error fallback). `SPEAKING` is reachable only from `ANSWER` (greeting) or `PROCESSING` — never from `LISTENING`. Transitions enforced by `session.transition()` — raises ValueError on invalid.
- **State ownership (greeting):** `ConversationManager.start_call` synthesizes the greeting as
  16 kHz PCM, then `_play_pcm16` drives `ANSWER → SPEAKING → LISTENING` through the PJSIP sink.
- **State ownership (live turns):** `pipeline.process_turn_stream` owns the `PROCESSING` entry.
  `ConversationManager._play_stream` moves to `SPEAKING` on the first emitted PCM sentence and
  returns to `LISTENING` after the producer/sink `TaskGroup` finishes. A zero-output turn also
  returns to `LISTENING`.
- **Audio path in:** PJSUA2 terminates negotiated RTP into 16 kHz mono PCM and schedules
  `ConversationManager.enqueue_pcm`; its single bounded consumer owns VAD and turn dispatch.
- **Audio path out:** LLM tokens are segmented, then each completed sentence uses stable
  `/v1/audio/speech` whole-WAV synthesis, one 24-to-16 kHz conversion, and the bounded
  `PjsipAudioSink`. `/v1/audio/speech/stream` remains diagnostic-only and is not used for
  production responses. The legacy ARI/RTP adapter alone converts output to 8 kHz A-law.
- **Barge-in:** Detected during `{LISTENING, SPEAKING, PROCESSING}`. Under the per-call lock,
  `ConversationManager` advances the generation, cancels the current playback task, clears the
  sink, and dispatches the replacement turn. Stable TTS checks cancellation between codec steps
  and releases its exclusive model lock; the stale-generation guard still discards any late
  output at the agent boundary. The cancelled assistant turn is not persisted; a session flag
  injects one transient system context into the next LLM request and is then cleared.
- **Turn detection (Smart Turn v3, on by default):** When `turn_detection_enabled` (default **True**) and a `TurnDetector` is injected (`_turn_active()`), the LISTENING turn-end runs a lowered VAD floor (`turn_vad_silence_ms`, default 200 ms) via `VadBuffer.add_frame_candidate` (returns a candidate **without** resetting) and confirms with a classify call in `_gate_turn_end`. `complete` (or hard cap `at_cap`, or candidate < `_MIN_CLASSIFY_SAMPLES`, or a classify error → **degrade**) flushes and dispatches; `incomplete` calls `continue_speech()` and keeps listening (bounded by `max_speech_ms`). Barge-in is **unchanged**: SPEAKING/PROCESSING use a separate `_bargein_buffers` VadBuffer at the legacy 800 ms floor. Flag off → single VadBuffer at 800 ms for every state (legacy path verbatim). `classify` runs **before** the turn lock; if state leaves LISTENING during the await, the verdict is discarded. The detector is **in-process**: an 8 MB Smart Turn v3 ONNX model (Whisper-Tiny encoder) downloaded once on enable (revision-pinned), run via `onnxruntime` in `asyncio.to_thread` with `inter_op_num_threads=1` so it never starves RTP pacing; Whisper log-mel features via `transformers.WhisperFeatureExtractor`. Constructed only when enabled (else `None`).
- **Greeting:** Playback is asynchronous after the priority lease and stable whole-WAV synthesis
  succeed, so the PJSIP event loop remains responsive.
- **VoicePipeline callables:** Constructor takes `stt`, `llm`, `tts` as bare callables plus
  optional `llm_stream`/`tts_stream`. Production responses use `llm_stream` and the stable `tts`
  callable per completed sentence. `tts_stream` stays wired for compatibility/diagnostics but is
  never invoked by the VoIP response pipeline.

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

- **RTP port allocation:** Starts at `rtp_port` (default 5000), increments by 2 per call, wraps at 65534. `_bind_rtp_server` retries the next port on `OSError` (bind collision) instead of trusting the counter; gives up after `RTP_BIND_ATTEMPTS`.
- **RTP bind vs advertise:** `rtp_bind_host` is the local socket bind (may be `0.0.0.0`). `rtp_advertise_host` is what `_create_external_media` puts in ExternalMedia's `external_host` — the address Asterisk dials RTP *to*. It must be routable from Asterisk; the validator rejects `0.0.0.0`/empty (fail closed — a wrong value is a silent-call bug). Default `127.0.0.1` assumes the agent is co-located with Asterisk; set the agent's LAN IP if separated.
- **Resource lifecycle:** One `httpx.AsyncClient` is created in `main()` and injected into `SttClient`/`TtsClient`/`LlmClient`/`RagTool` (each closes only a client it *owns* — see `_owns_client`); `AriClient` reuses a single internal client via `_client()`. `main()` closes the shared client, the ARI client, and the pg pool in a `finally`. The ARI websocket reconnects with capped exponential backoff (`run()` loop).
- **VAD frame size:** 20 ms at 16 kHz = 320 samples. Hard cap at `max_speech_ms` (default 15 s) to prevent unbounded buffer growth.
- **TTS output:** server returns a 24 kHz PCM_16 **WAV** →
  `pipeline._decode_wav` (strips the RIFF header) → 16 kHz PCM → PJSIP. Only
  the legacy ARI rollback path resamples to 8 kHz and A-law.
- **TTS voice field:** the client sends `voice` (free-form German voice description); the server maps `voice` → qwen-tts `instruct`. Do not send `instruct` — the server ignores unknown fields.
- **Embedding API:** RAG calls `POST /v1/embeddings` with `{"input": text}` and reads `data[0].embedding` (OpenAI-shaped). Not `/embed`.
- **MS Graph timezone:** `Prefer: outlook.timezone="Europe/Berlin"` header on calendarView requests. `create_event` uses `timeZone: Europe/Berlin`.
- **LLM tools:** `rag_lookup`, `calendar_get_events`, `calendar_create_event` defined in `agent/llm.py:TOOLS`.
- **Tool authorization (fail closed):** tools are offered to the LLM only when the caller number is in `trusted_callers` (`TRUSTED_CALLERS`). Empty allowlist = tools off for everyone. `process_turn` passes `session.caller_id` to `llm.complete(messages, caller_id)`. Unknown callers can still converse, but get no RAG/calendar access (closes read-side exfiltration). The tool loop is capped at `max_tool_rounds`. Caveat: caller-ID authorization trusts the SIP CLI, which is spoofable at the telephony layer — a stronger control (verified CLI / spoken PIN) belongs in Asterisk/Fritzbox config, not the agent.
- **Calendar write gate (deterministic, fail closed):** `calendar_create_event` no longer trusts the model-set `confirmed` arg as the boundary. A write commits only when all hold: (1) `calendar_write_enabled=True`; (2) a **prior** turn proposed the *exact same* event (server-side per-caller pending state in `LlmClient._pending_writes` — so the model can't one-shot a write); (3) the conversation has **advanced to a strictly later user turn** than the proposal (the caller actually got to answer the read-back); (4) that new turn matches `_AFFIRMATIVE`. The load-bearing gate is (3) "conversation advanced" — an injected tool result can't manufacture a user turn. The affirmative regex is a secondary signal, not "verified consent." A correction (different params) re-proposes rather than committing the stale event. `_dispatch` is threaded with `(caller_id, user_turns, last_user)` from **both** `complete` and `complete_stream`.
- **Smart Turn v3 on by default, fail-fast:** turn detection ships enabled (`turn_detection_enabled=True`). The in-process ONNX model is auto-downloaded from HF (revision-pinned) at startup, no DGX service; if it can't load the agent fails fast rather than silently degrading. German verified **offline** at ~95% on pipecat's synthetic test split (see `docs/research/2026-06-14-smart-turn-german-accuracy.md`); a real-call smoke test on the live trunk is still recommended (telephony 8 kHz aLaw runs ~92–93%, and the test split is synthetic). Default threshold 0.70 biases toward fewer cut-ins. Tests mock the session/feature-extractor boundary (no model download in CI).

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
