# Streaming voice pipeline (#3) — design

**Date:** 2026-05-31
**Status:** Design — not yet implemented
**Tracked as:** TODO.md #3 (streaming pipeline) + faster-qwen3-tts adoption
**Supersedes the "blocked on TTS server" note** in TODO.md with a concrete plan.

## Problem

The turn loop is fully serialized and non-streaming. VAD waits 800ms of
silence, then STT → LLM → TTS run end-to-end before a single audio sample
plays. The caller hears dead air for the whole compute window. A phone target
of ~300–800ms to first audio is impossible under this model. On a phone, dead
air reads as a dropped call.

Two latency sources stack:

1. **Pipeline serialization** — STT, LLM, and TTS run sequentially to
   completion before playback starts.
2. **TTS time-to-first-audio (TTFA)** — the current model takes ~567–661ms to
   first audio even once it starts.

Fixing TTS alone does not help: serialization (source 1) alone blows the
budget. The fix is to stream the whole pipeline *and* swap to a TTS runtime
that can yield audio incrementally.

## Decision: adopt faster-qwen3-tts, bundled with the streaming rewrite

### Why the swap (go/no-go resolved GO)

- **VoiceDesign is preserved.** faster-qwen3-tts runs the *same weights*
  (`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`) and exposes both
  `generate_voice_design` and `generate_voice_design_streaming`. The free-form
  German voice instruct — a hard requirement — survives unchanged. The swap
  replaces the *runtime*, not the voice model.
- **Only the swap unlocks audio-output streaming.** The current server hardcodes
  `non_streaming_mode=True`, but that flag controls **text input feeding**
  (all-at-once vs progressive), not audio output. Flipping it still returns one
  complete WAV. Incremental audio-chunk yield is faster-qwen3-tts's own
  contribution (CUDA-graph static-cache path): *"audio chunks are yielded during
  generation with the same per-step performance as non-streaming."* So a
  flag-flip on the current `generate_voice_design()` call cannot feed RTP
  incrementally — the swap is required.
- **~2x TTFA** via CUDA graphs (567→280ms on 0.6B, 661→400ms on 1.7B per the
  upstream benchmarks), with `chunk_size` tunable.
- **Telephony masks the quality risk.** The CUDA-graph static-cache path "can
  differ numerically" from the upstream dynamic-cache path, but output is
  resampled 24kHz → 8kHz aLaw narrowband before it reaches the caller. Those
  numerical differences are below the audible floor of the phone codec. This is
  a point *for* swapping in this specific context.

### Risks accepted

- New dependency, single-maintainer community package (andimarafioti — HF staff,
  not Alibaba-official).
- CUDA graphs pin tensor shapes and are PyTorch-version-sensitive (a
  compatibility note is flagged in the README); first-synth warmup cost.
- Necessary-but-not-sufficient: without the streaming pipeline rewrite, the swap
  delivers little. The two ship together.

### Rejected alternatives

- **Option A — flip `non_streaming_mode` on the current server, no swap.**
  Mirage: the flag is text-input feeding, not audio output. Still returns a full
  buffer; cannot stream into RTP. Off the table.
- **Option C — status quo.** Phone latency target stays impossible.

## Architecture

### Current (serialized)

```
VAD end-of-speech
  → pipeline.process_turn  [owns PROCESSING]
      STT (full) → LLM (full, +tool rounds) → TTS (full WAV)
  → returns complete aLaw buffer
  → ari._play_audio  [PROCESSING → SPEAKING → LISTENING]
      rtp.stream_audio  (20ms frames, monotonic clock)
```

Clean invariant: compute fully precedes playback. `pipeline` owns PROCESSING,
`ari` owns SPEAKING, the two never overlap.

### Target (streaming)

```
VAD end-of-speech
  → STT (full — STT stays whole-turn; see Scope)
  → LLM token stream
      → sentence/clause segmenter (German-aware buffer)
          → TTS generate_voice_design_streaming(text=sentence, chunk_size=N)
              → 24kHz PCM chunk
                  → resample 24k→8k → aLaw encode
                      → output chunk queue
                          → RTP 20ms-frame drain (monotonic clock)
```

Compute and playback **overlap**: TTS is still generating while RTP is already
draining. This is the central change and the source of every hard part below.

### Components (new / changed)

| Unit | Responsibility | Depends on |
|---|---|---|
| `TtsClient.synthesize_stream` (changed) | New async-generator method: POST text, yield 24kHz PCM chunks as they arrive (HTTP chunked transfer or a streaming endpoint on the server). Keeps the existing whole-buffer `synthesize` for the greeting. | streaming TTS server |
| `dgx/tts/server.py` (changed) | New streaming endpoint (e.g. `/v1/audio/speech/stream`) backed by `generate_voice_design_streaming`, yielding PCM chunks via `StreamingResponse`. Keeps `/v1/audio/speech` for non-streaming callers. | faster-qwen3-tts |
| `SentenceSegmenter` (new) | Buffer LLM token stream; emit on clause/sentence boundary so TTS gets coherent prosodic units, not raw tokens. German punctuation + abbreviation handling. | — |
| `pipeline.process_turn_stream` (new) | Orchestrate STT → LLM-stream → segmenter → TTS-stream → aLaw chunk queue. Async generator yielding aLaw chunks. | STT, LLM (streaming), TtsClient, SentenceSegmenter |
| `rtp.stream_audio_chunks` (new) | Drain an async chunk queue as 20ms frames on the monotonic clock, with prebuffer + underrun handling. Generalizes the existing `stream_audio`. | — |
| `AriClient` playback (changed) | Drive the overlapped FSM, own the producer-chain teardown on barge-in. | pipeline, rtp |

## Design decisions for the hard parts

### 1. Barge-in tears down a live producer chain

Keep the existing per-call turn lock + monotonic generation id. The cancel
target grows from one `_playback_task` to the whole producer chain (LLM stream,
TTS stream, RTP drain). On barge-in:

1. Bump generation under the lock.
2. Cancel the chain top-down (LLM stream task → which cascades to segmenter →
   TTS stream → RTP drain), or cancel a single supervising task that owns the
   chain via a `TaskGroup`/`gather`. **Decision: one supervising task per turn
   owning the chain via `asyncio.TaskGroup`**, so a single `.cancel()` tears down
   the whole chain and the generation-guard semantics are unchanged.
3. Barge-in may fire *mid-generation* (before any audio emitted). The guard
   already no-ops stale generations; the new case is cancelling a chain that has
   not yet produced a chunk — covered by the same supervising-task cancel.

### 2. FSM overlap

The documented invariant ("`pipeline` returns audio with session in
PROCESSING"; "SPEAKING only after PROCESSING completes") **no longer holds** —
compute and playback overlap.

**Decision:** enter SPEAKING on the **first emitted aLaw chunk** while compute
continues. PROCESSING means "computing, nothing played yet"; SPEAKING means "at
least one chunk has hit RTP." Transition `PROCESSING → SPEAKING` fires once, on
first chunk. `ari` still owns the SPEAKING transition; `pipeline` still owns
PROCESSING entry. No new state — the existing states are reinterpreted as
"first-chunk" boundaries rather than "compute-complete" boundaries. CLAUDE.md's
state-ownership section must be updated to document the overlap.

### 3. Tool-call turns cannot stream

When the LLM calls a tool (RAG/calendar), the final answer is unknown until tool
rounds resolve (up to `max_tool_rounds`). The response cannot stream early.

**Decision:** detect tool use on the first LLM stream event. If the model emits a
tool call, fall back to the existing non-streaming path for that turn **and**
emit a short filler utterance ("Einen Moment, ich schaue nach…") via the
non-streaming greeting path so the caller hears something during the tool round.
Tool-free turns get the streaming win; tool turns keep full latency but mask it
with the filler. The filler is fixed text — pre-synthesizable/cacheable.

### 4. Output underrun

RTP drains 20ms frames at fixed wall-clock; TTS yields faster than realtime
(RTF>1) normally, but GPU contention can drop it below realtime mid-utterance,
underrunning RTP → audible gap/click.

**Decision:** prebuffer a small fixed number of chunks (target ~100–200ms of
audio) before starting RTP drain, trading a little first-audio latency for
underrun protection. On underrun (queue empty but turn not done), hold the last
frame / emit comfort noise rather than stopping. A full jitter buffer + PLC stays
deferred; the prebuffer is the minimum viable guard. `chunk_size` and prebuffer
depth are tuned together on real calls.

### 5. Mid-stream failure

A TTS/LLM failure can now occur after audio already played, leaving a
half-spoken sentence.

**Decision:** on mid-stream error, stop cleanly at the last good frame, log, and
transition to a recovery prompt (fixed text via the non-streaming path:
"Entschuldigung, da ist etwas schiefgelaufen.") then back to LISTENING. Never
leave RTP draining a dead queue.

## Tuning knobs (set in implementation, tune on real calls)

- **`chunk_size`** — latency vs prosody seams. Start at the README default
  (8 steps ≈ 667ms/chunk); reduce only if first-audio latency demands it.
- **Sentence-aware feeding** — clause/sentence boundary, not raw tokens. German
  segmentation with abbreviation handling.
- **Prebuffer depth** — ~100–200ms; balances first-audio latency against
  underrun risk.
- **Barge-in / VAD sensitivity** — responses are now longer-lived and
  interruptible; a backchannel "mhm" can falsely trigger barge-in. Revisit VAD
  aggressiveness and the barge-in threshold.

## Scope

### In scope

- TTS server streaming endpoint + faster-qwen3-tts adoption.
- `TtsClient.synthesize_stream`, `SentenceSegmenter`,
  `pipeline.process_turn_stream`, `rtp.stream_audio_chunks`.
- LLM streaming completion (token stream from the chat endpoint).
- Overlapped FSM (first-chunk SPEAKING transition).
- Producer-chain barge-in teardown.
- Tool-turn fallback + filler.
- Output prebuffer + underrun handling.
- Mid-stream failure recovery.

### Out of scope (deferred, separate work)

- **Streaming STT.** STT stays whole-turn — the model already streams but the
  added complexity (partial transcripts, re-decoding) is a separate lever and
  not on the critical first-audio path the way LLM→TTS is. Revisit after the
  LLM→TTS stream lands.
- **Input-side endpointing.** The 800ms VAD silence still gates turn start.
  Reducing it is a separate latency lever, not part of this swap.
- **Full jitter buffer + PLC.** The prebuffer is the minimum guard; a real
  jitter buffer is deferred (matches the #4 tail note).
- **Parakeet ASR.** Not switching — see TODO.md.

## Testing

Per the `tests-mock-client-server-boundary` lesson — assert on the actual
streaming contract, not a mock that passes green while broken.

- **TtsClient stream:** mock a chunked HTTP response; assert
  `synthesize_stream` yields decoded PCM chunks in order and closes cleanly.
- **SentenceSegmenter:** token-by-token feed → assert emission only on
  boundaries, German abbreviations don't false-split, trailing partial flushes
  at stream end.
- **process_turn_stream:** mock STT/LLM/TTS callables; assert aLaw chunks yielded
  incrementally (first chunk before LLM stream completes), tool-call path falls
  back + emits filler.
- **FSM overlap:** assert `PROCESSING → SPEAKING` fires on first chunk, not after
  compute; assert it fires exactly once.
- **Barge-in mid-stream:** start a turn, fire VAD during generation, assert the
  whole producer chain cancels, generation bumps, no stale chunk reaches RTP,
  superseded turn never moves the FSM.
- **Underrun:** feed a chunk queue slower than realtime; assert RTP holds rather
  than stopping; assert prebuffer delays drain start by the configured depth.
- **Mid-stream failure:** inject a TTS error after N chunks; assert clean stop +
  recovery prompt + LISTENING.

## Build sequence (rough — detailed plan via writing-plans)

1. TTS server streaming endpoint + faster-qwen3-tts on the DGX side; verify
   VoiceDesign output + TTFA on the box.
2. `TtsClient.synthesize_stream` against the new endpoint.
3. `SentenceSegmenter` (pure, unit-testable).
4. LLM streaming completion.
5. `pipeline.process_turn_stream` wiring STT → LLM-stream → segmenter →
   TTS-stream → aLaw queue.
6. `rtp.stream_audio_chunks` with prebuffer + underrun.
7. `AriClient` overlapped FSM + producer-chain barge-in teardown.
8. Tool-turn fallback + filler; mid-stream failure recovery.
9. Tune `chunk_size` / prebuffer / barge-in threshold on real calls.

## Open questions for implementation

- Does the TTS server stream PCM via `StreamingResponse` (raw chunks) or a
  framed protocol? Raw 24kHz PCM chunks are simplest; the client resamples.
- Does the LLM endpoint (`/v1/chat/completions`) support SSE token streaming in
  the current deployment? If not, that's a prerequisite.
- Confirm faster-qwen3-tts PyTorch version against the DGX image's torch.
