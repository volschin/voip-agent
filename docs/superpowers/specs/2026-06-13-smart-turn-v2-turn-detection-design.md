# Smart Turn v2 Turn Detection — Design

**Date:** 2026-06-13
**Status:** Approved for planning
**Branch:** TBD (feature branch off `main`)

## Problem

The agent ends a caller's turn purely on acoustic silence: `VadBuffer` flushes the
speech buffer after a fixed **800 ms** of silence (`agent/audio.py`, `silence_threshold_ms`).
A natural thinking pause longer than 800 ms ("Moment, ich überlege ...") is
misread as end-of-turn, so the agent starts talking over the caller. Pure VAD has
no language/prosody understanding — it only knows *whether* someone is speaking,
not *whether they are done*.

## Goal

Add semantic, prosody-aware end-of-turn detection so the agent waits through
mid-utterance pauses but still responds promptly on genuine turn-ends — without
regressing barge-in and without freezing turn-taking if the model is slow/unavailable.

Out of scope: changing the ASR/LLM/TTS pipeline; fixing barge-in latency;
building the DGX inference container (separate task — endpoint contract pinned below).

## Solution Overview

Use **Smart Turn v2** (`pipecat-ai/smart-turn-v2`, wav2vec2-based, multilingual incl.
German) as an audio-waveform classifier: given the buffered speech, predict
`complete` vs `incomplete`. It runs as a **DGX HTTP service** alongside Qwen3-ASR/TTS
and Hermes; the agent calls it through a new `TurnDetectorClient`.

Timing model (chosen during brainstorm): **lower the VAD silence floor and let
Smart Turn v2 confirm**. VAD proposes an endpoint candidate fast (~200 ms silence);
the classifier decides flush-now vs keep-listening. Snappy on real endings, patient
on pauses.

### Decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Topology | DGX HTTP service (consistent with STT/TTS/Hermes; GPU; keeps agent box lean) |
| Gating | Lower VAD floor (~200 ms) → Smart Turn v2 confirms `complete`/`incomplete` |
| Fallback | Degrade to legacy silence flush on error/timeout; hard cap via `max_speech_ms` (15 s); fail toward responding |
| Barge-in | Unchanged — Smart Turn v2 gates LISTENING turn-end only, not barge-in |
| Rollout | Feature flag `turn_detection_enabled`, default `False` |

## Architecture

### Topology comparison (record of decision)

| | DGX HTTP service (**chosen**) | In-process |
|---|---|---|
| Pattern | new endpoint, like existing AI services | torch/onnx in agent via `asyncio.to_thread` |
| Latency | LAN hop ~1–5 ms + GPU inference | no hop, CPU inference contends with event loop |
| Deps | agent stays httpx-only | heavy ML dep + weights in agent runtime |
| Lifecycle | fits injected `httpx.AsyncClient` / `_owns_client` | new model load/warmup in agent |

LAN hop is negligible against the 150 ms latency budget; GPU and a lean agent
runtime win. Chosen: **DGX service**.

## Components

### a) `TurnDetectorClient` (`agent/turn_detector.py`)

Mirrors `SttClient`. Takes an injected `httpx.AsyncClient` (closes only a client it
owns — `_owns_client`).

```python
async def classify(self, pcm_16k: np.ndarray) -> bool:
    """True = caller's turn is complete. Raises on timeout/HTTP error."""
```

- `POST {turn_detector_url}/v1/turn/classify` with 16 kHz int16 PCM.
- Reads `prob`, returns `prob >= turn_complete_threshold` (default 0.5).
- Per-call timeout `turn_classify_timeout_ms` (150). Timeout / non-2xx /
  connection error → raise; the policy in `_on_audio` catches and falls back to a
  legacy flush.

### b) `VadBuffer` — two instances per call (state-disjoint)

Barge-in must stay unchanged (800 ms) while turn-end wants a low floor (200 ms).
LISTENING and SPEAKING/PROCESSING never overlap, so use two buffers selected by
`session.state` rather than mutating one buffer's threshold:

- `_turn_vad` — `silence_threshold_ms = turn_vad_silence_ms` (200). Used in LISTENING.
  Emits endpoint candidates fast.
- `_bargein_vad` — `silence_threshold_ms = 800`. Used in SPEAKING/PROCESSING. Exactly
  today's behavior.

`VadBuffer`'s internal semantics are untouched (keeps it simple and already tested);
the state→buffer selection and the gating policy live in `AriClient._on_audio`.

When `turn_detection_enabled = False`: a single `VadBuffer` at 800 ms, today's path
verbatim (regression guard).

### c) New method `VadBuffer.continue_speech()`

On an `incomplete` verdict, keep accumulating: reset the silence counter, **retain**
the buffered frames (do not flush, do not clear). Distinct from `reset()` (clears
everything) and `_flush()` (returns + resets).

```python
def continue_speech(self) -> None:
    self._silence_count = 0
    # _speech_frames retained; _in_speech stays True
```

### d) Config (`agent/config.py`)

| Field | Default | Purpose |
|---|---|---|
| `turn_detection_enabled` | `False` | Feature flag; off = legacy single-buffer 800 ms path |
| `turn_detector_url` | — | DGX Smart Turn service base URL |
| `turn_complete_threshold` | `0.5` | `prob` cutoff for `complete` |
| `turn_classify_timeout_ms` | `150` | latency budget; exceed → degrade |
| `turn_vad_silence_ms` | `200` | endpoint-candidate floor for `_turn_vad` |

## Data Flow (LISTENING turn-end)

```
frame → _on_audio
  state == LISTENING → candidate = _turn_vad.add_frame(frame)
     candidate is None → return                          # still speaking / <200ms silence
     candidate not None:                                 # 200ms silence = endpoint candidate
        if not turn_detection_enabled → flush (legacy)
        if len(buffer) >= max_speech_frames → flush       # HARD CAP (15s), always ends
        if candidate too short (< min frames) → flush     # noise; skip inference
        try: complete = await classify(candidate)         # 150ms budget, before turn lock
        except (timeout / error): flush                   # DEGRADE → respond
        if state changed during await (not LISTENING) → discard
        if complete → flush → enter existing lock + generation + process_turn_stream
        else → _turn_vad.continue_speech(); return        # thinking pause: keep listening

  state in (SPEAKING, PROCESSING) → _bargein_vad.add_frame(frame)   # UNCHANGED
     candidate not None → existing barge-in teardown + new turn
```

Key ordering: `classify` runs **before** the per-call turn lock, so a slow verdict
never holds the lock. On `complete`, the existing lock + monotonic-generation +
`process_turn_stream` / `_play_stream` path is entered unchanged.

No datagram race: `_audio_consumer` serializes `_on_audio` calls, so the next frame
isn't processed until the current `await classify` returns.

## Error Handling / Edge Cases

- STv2 timeout / 5xx / connection error → legacy flush, WARN log, call continues.
- Candidate below `min frames` → flush directly, no inference (don't burn a call on noise).
- State leaves LISTENING during `await classify` (e.g. hangup → ENDED) → discard verdict.
- `continue_speech` loop is bounded every iteration by `max_speech_frames` → guaranteed
  to terminate; STv2 can never stall a turn indefinitely.

## DGX Service (separate task)

New `smart-turn` container in `dgx/` loading `pipecat-ai/smart-turn-v2`, exposing
`POST /v1/turn/classify` (int16 16 kHz PCM body → `{"complete": bool, "prob": float}`),
with a health check, compose entry analogous to TTS. Endpoint contract is pinned here;
container build is tracked separately and is **not** part of the agent spec.

## Testing (`tests/`, respx + `unittest.mock.AsyncMock`)

- `classify` → complete → immediate flush, turn dispatched.
- `classify` → incomplete → `continue_speech`, no turn; next candidate complete →
  flush with the **full** buffer (both speech segments concatenated).
- timeout/error → degrade flush.
- cap reached while still `incomplete` → flush.
- `turn_detection_enabled = False` → single VadBuffer, 800 ms, today's behavior
  (regression guard).
- barge-in path (SPEAKING / PROCESSING) never calls STv2 (`classify` `assert_not_awaited`).
- state change during `await classify` → verdict discarded, no turn.

## Risks / Caveats

- **German precision unverified.** Smart Turn v2 is multilingual but DE accuracy is
  not measured. Requires a live eval before flipping `turn_detection_enabled = True`
  (cf. the TTS `language` full-names lesson — multilingual ≠ verified for German).
- Caller-ID/security: none — this path handles audio framing only.
- Tests mock the client/server boundary; the DGX endpoint contract (PCM format,
  `prob` field) must be wire-verified against the real container before enabling.
