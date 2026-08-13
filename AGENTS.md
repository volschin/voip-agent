# voip-agent repository instructions

German telephone assistant on Python 3.12+. Production uses direct FRITZ!Box
PJSUA2 media; `ari.py`, `rtp.py` and `asterisk/` are rollback paths. AI services
are authenticated OpenAI-compatible Qwen3-ASR, Gemma and Qwen3-TTS endpoints.

## Commands

```bash
venv/bin/pytest -v
venv/bin/pytest tests/test_pipeline.py -v
venv/bin/ruff check agent/ tests/
venv/bin/ruff format --check agent/ tests/
venv/bin/python -c "from agent.main import main; print('OK')"
```

Tests must not contact real AI services, databases or Microsoft Graph. Use
`respx`/`AsyncMock`; report live PJSIP or DGX validation separately.

## Call and media invariants

- Preserve the session FSM: greeting `ANSWER -> SPEAKING -> LISTENING`; live turn
  `LISTENING -> PROCESSING -> SPEAKING -> LISTENING`; any state may end.
  `session.transition()` owns validation.
- PJSUA2 input/output is 16 kHz mono PCM. Production TTS returns whole 24 kHz
  PCM16 WAVs, then converts once to 16 kHz. Only the legacy ARI path uses 8 kHz
  A-law. `/v1/audio/speech/stream` is diagnostic-only.
- Keep blocking model/SDK work off the event loop. PJSIP pacing, bounded queues,
  cancellation and the exclusive TTS model lock are latency-critical.
- Barge-in advances the generation, cancels playback, clears the sink and drops
  stale/unfinished assistant output. Do not persist the cancelled turn.
- Smart Turn v3 is an in-process, revision-pinned ONNX model and defaults on.
  Startup fails if it cannot load. Classify only LISTENING candidates; the
  SPEAKING/PROCESSING barge-in VAD path remains separate.

## Security and external contracts

- Configuration stays in `agent/config.py`; never commit `.env`, credentials,
  caller numbers, voice references or database data.
- Tools are offered only to normalized callers in `TRUSTED_CALLERS`; an empty
  allowlist disables them. Unknown callers may converse but receive no
  RAG/calendar access.
- Calendar writes require the feature flag, an exact prior proposal, a later
  user turn and an affirmative reply. Preserve the server-side pending state and
  pass `(caller_id, user_turns, last_user)` through both completion paths.
- RAG uses `POST /v1/embeddings` and reads `data[0].embedding`.
- TTS sends the `voice` field, not `instruct`.
- Microsoft Graph uses `Europe/Berlin`; wrap blocking MSAL calls in
  `asyncio.to_thread`.
- Keep DGX services, model data and Compose deployment in `dgx/`. A model/runtime
  change requires isolated qualification before production adoption.

Preserve unrelated live services and rollback artifacts. Update README/design
docs when wire contracts, environment variables or deployment steps change.
