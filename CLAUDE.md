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
- **State ownership (greeting):** the greeting is non-streaming — `pipeline.synthesize_alaw` produces the audio and `AriClient._play_audio` drives `ANSWER → SPEAKING → LISTENING` around RTP playback. (`pipeline.process_turn` is a retained whole-turn API — it owns `PROCESSING` and returns audio with the session still in `PROCESSING` for `_play_audio` to drive — but is **no longer wired** in production: live turns use `process_turn_stream` and its in-stream fallback. Kept for tests / non-streaming fallback.)
- **State ownership (streaming turns, #3):** `PROCESSING` and `SPEAKING` now **overlap** — compute continues while audio plays. `pipeline.process_turn_stream` owns the `PROCESSING` entry (and stays there through STT + first LLM tokens); `AriClient._play_stream`'s `produce()` flips `PROCESSING → SPEAKING` on the **first emitted aLaw chunk** while generation is still ongoing, then `_play_stream` drives `SPEAKING → LISTENING` at end. The tail recovers `LISTENING` from `SPEAKING` *or* `PROCESSING`, so a turn that yields zero chunks never strands the FSM. The producer chain (LLM stream → segmenter → TTS stream → RTP drain) is owned by a per-turn `asyncio.TaskGroup`; a single barge-in `.cancel()` tears down producer + consumer together.
- **Audio path in:** Asterisk ExternalMedia → UDP RTP → `RtpServer.datagram_received` → `parse_rtp_payload` (validates RFC 3550 header: version, CSRC, extension, padding; drops malformed) → `on_audio` callback → `AriClient._enqueue_audio` (bounded per-call `asyncio.Queue`, drops on overflow) → single `_audio_consumer` task → `_on_audio` → `VadBuffer.add_frame` → on speech end → `pipeline.process_turn`. The consumer is the **only** caller of `_on_audio`, so per-packet work is serialized (no task-per-datagram).
- **StasisStart filtering + transactional setup:** the ExternalMedia leg we create (`channelId = _EXT_PREFIX + channel_id`, i.e. `ext-*`) re-enters the same Stasis app, firing a second StasisStart. `_handle_event` drops any channel whose id starts with `_EXT_PREFIX` — otherwise each call recursively spawns another session/RTP/bridge. Caller number is read defensively (`ch.get("caller", {}).get("number", "")`) since ext/originated channels may omit it. `_setup_call` is **transactional**: its whole body is wrapped in try/except and any failure calls `_teardown_call(channel_id)` to free the session/RTP socket/queue/consumer it had already registered (it runs in a detached task, so nothing else would observe the exception). Local teardown does *not* hang up the Asterisk-side channels — StasisEnd drives that. Asterisk-side integration of the ext-channel filter remains **unverified** (needs a live-Asterisk test).
- **Audio path out (greeting/fallback):** `pipeline.synthesize_alaw` → `rtp_server.stream_audio` (20 ms frames paced against an **absolute monotonic clock**, `start + n*20ms`, not `sleep(0.02)`, so per-frame work doesn't drift) → RTP UDP → Asterisk
- **Audio path out (streaming turns, #3):** `pipeline.process_turn_stream` yields aLaw blobs → `_play_stream` feeds a bounded `asyncio.Queue` → `rtp_server.stream_audio_chunks` (same absolute-clock pacing; prebuffers a few frames, fills underruns with comfort silence so the RTP clock never stalls, `None` sentinel = producer done) → RTP UDP → Asterisk. Server (`/v1/audio/speech/stream` via faster-qwen3-tts) streams 24 kHz int16 PCM chunks → `TtsClient.synthesize_stream` → `resample_24k_to_8k` → aLaw.
- **Barge-in:** Detected when VAD fires during any state in `_INTERRUPTIBLE_STATES` = `{LISTENING, SPEAKING, PROCESSING}`. `PROCESSING` is interruptible because in streaming the `PROCESSING → first-chunk` window lasts seconds — the caller must be able to cut in mid-generation. Guarded by a per-call `asyncio.Lock` + monotonic generation id (`_generation[channel_id]`). `_on_audio` bumps the generation under the lock, cancels the in-flight `_playback_tasks[channel_id]` (the streaming `TaskGroup`), moves `SPEAKING`/`PROCESSING → LISTENING`, then dispatches the new turn via `_play_stream(channel_id, session, gen, speech)`. `_play_stream` no-ops if `gen` is stale, and `produce()` breaks if the generation changes mid-stream — a superseded turn can never emit audio or move the FSM.
- **Turn detection (Smart Turn v3, opt-in):** When `turn_detection_enabled` and a `TurnDetector` is injected (`_turn_active()`), the LISTENING turn-end runs a lowered VAD floor (`turn_vad_silence_ms`, default 200 ms) via `VadBuffer.add_frame_candidate` (returns a candidate **without** resetting) and confirms with a classify call in `_gate_turn_end`. `complete` (or hard cap `at_cap`, or candidate < `_MIN_CLASSIFY_SAMPLES`, or a classify error → **degrade**) flushes and dispatches; `incomplete` calls `continue_speech()` and keeps listening (bounded by `max_speech_ms`). Barge-in is **unchanged**: SPEAKING/PROCESSING use a separate `_bargein_buffers` VadBuffer at the legacy 800 ms floor. Flag off → single VadBuffer at 800 ms for every state (legacy path verbatim). `classify` runs **before** the turn lock; if state leaves LISTENING during the await, the verdict is discarded. The detector is **in-process**: an 8 MB Smart Turn v3 ONNX model (Whisper-Tiny encoder) downloaded once on enable (revision-pinned), run via `onnxruntime` in `asyncio.to_thread` with `inter_op_num_threads=1` so it never starves RTP pacing; Whisper log-mel features via `transformers.WhisperFeatureExtractor`. Constructed only when enabled (else `None`).
- **Greeting:** Non-blocking — launched via `asyncio.create_task(_play_audio(...))` so the ARI WebSocket reader is never blocked.
- **VoicePipeline callables:** Constructor takes `stt`, `llm`, `tts` as bare callables (not objects), plus optional `llm_stream`/`tts_stream` for the streaming path. In main.py: `stt=stt_client.transcribe`, `llm=llm_client.complete`, `tts=tts_client.synthesize`, `llm_stream=llm_client.complete_stream`, `tts_stream=tts_client.synthesize_stream`. `complete_stream` reuses `complete`'s auth/cap/`_dispatch_safe` exactly (tools offered only to trusted callers, below `max_tool_rounds`); it fires `on_tool_round` so the pipeline can play a filler utterance while a tool round resolves (tool turns can't stream tokens).

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
- **TTS output:** server returns a 24 kHz PCM_16 **WAV** → `pipeline._decode_wav` (strips the RIFF header) → `resample_24k_to_8k` → aLaw encode → RTP. STT input: 8 kHz aLaw → `alaw_decode` → `resample_8k_to_16k` → WAV → Qwen3-ASR.
- **TTS voice field:** the client sends `voice` (free-form German voice description); the server maps `voice` → qwen-tts `instruct`. Do not send `instruct` — the server ignores unknown fields.
- **Embedding API:** RAG calls `POST /v1/embeddings` with `{"input": text}` and reads `data[0].embedding` (OpenAI-shaped). Not `/embed`.
- **MS Graph timezone:** `Prefer: outlook.timezone="Europe/Berlin"` header on calendarView requests. `create_event` uses `timeZone: Europe/Berlin`.
- **LLM tools:** `rag_lookup`, `calendar_get_events`, `calendar_create_event` defined in `agent/llm.py:TOOLS`.
- **Tool authorization (fail closed):** tools are offered to the LLM only when the caller number is in `trusted_callers` (`TRUSTED_CALLERS`). Empty allowlist = tools off for everyone. `process_turn` passes `session.caller_id` to `llm.complete(messages, caller_id)`. Unknown callers can still converse, but get no RAG/calendar access (closes read-side exfiltration). The tool loop is capped at `max_tool_rounds`. Caveat: caller-ID authorization trusts the SIP CLI, which is spoofable at the telephony layer — a stronger control (verified CLI / spoken PIN) belongs in Asterisk/Fritzbox config, not the agent.
- **Calendar write gate (deterministic, fail closed):** `calendar_create_event` no longer trusts the model-set `confirmed` arg as the boundary. A write commits only when all hold: (1) `calendar_write_enabled=True`; (2) a **prior** turn proposed the *exact same* event (server-side per-caller pending state in `LlmClient._pending_writes` — so the model can't one-shot a write); (3) the conversation has **advanced to a strictly later user turn** than the proposal (the caller actually got to answer the read-back); (4) that new turn matches `_AFFIRMATIVE`. The load-bearing gate is (3) "conversation advanced" — an injected tool result can't manufacture a user turn. The affirmative regex is a secondary signal, not "verified consent." A correction (different params) re-proposes rather than committing the stale event. `_dispatch` is threaded with `(caller_id, user_turns, last_user)` from **both** `complete` and `complete_stream`.
- **Smart Turn v3 fail-closed:** turn detection is off by default. German precision is unverified (multilingual ≠ verified-for-German — see the TTS language lesson); verify live before enabling. The detector is in-process — the ONNX model is auto-downloaded from HF (revision-pinned) on enable, no DGX service. Tests mock the session/feature-extractor boundary (no model download in CI), so spot-check the real model live before flipping the flag.

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
