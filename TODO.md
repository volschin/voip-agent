# TODO — deferred hardening

Tracked from an adversarial codex review (2026-05-30). The contract bugs
(#1/#2/#10) and security findings (#11 read+write, #13 network/creds) are fixed
on branch `fix/contract-and-fsm-bugs`. The real-time-correctness and
concurrency/lifecycle items (#4/#5/#8/#9/#12) are fixed on branch
`feat/realtime-hardening`. The LLM-streaming rewrite and faster-qwen3-tts
service are also complete; production response playback now uses stable
whole-WAV sentence synthesis, while the raw codec stream remains available
only for diagnostics. Numbering matches the original review.

## Live findings (open)

Observed on a real FRITZ!Box call on 2026-07-31, 18:27 UTC (caller `**613`,
internal handset). The call itself completed normally; one response turn was
lost silently.

- [x] **TTS 503 drops a response turn without recovery.** `/v1/audio/speech`
  returned `503 Service Unavailable` mid-conversation; `TtsClient.synthesize`
  raised through `raise_for_status`, `VoicePipeline.synthesize_pcm16` logged
  `TTS failed for text: 'Das tut mir leid, das war nicht meine Absicht.'`, and
  `process_turn_stream` aborted with `RuntimeError: TTS returned no audio`.
  The caller heard nothing for that turn and got no spoken error. The next
  `/v1/audio/speech` call 3 s later returned 200, so the outage was brief.
  Fixed: `TtsClient.synthesize` now retries the transient classes
  (`RETRY_STATUS` = 502/503/504, plus `httpx.TransportError`) on the
  `RETRY_BACKOFF_S` schedule (0.3 s, 0.9 s — 3 attempts, ≤1.2 s added worst
  case) and logs each retry at WARNING so recovered outages stay visible. A
  4xx or the request-body 500 is not retried. The backoff sleeps are
  cancellation points, so barge-in still cuts a retrying turn.
- [ ] **Root-cause the server-side 503.** Determine why the DGX
  `/v1/audio/speech` route rejects a request under conversational load —
  exclusive model lock contention, worker restart, or Traefik-level shedding.
  The agent-side retry above treats the symptom; this is the cause. Decide
  from the finding whether the retry needs a backoff longer than one turn.
- [x] **Caller ID format confirmed live.** Lifecycle logging in
  `agent/answer_policy.py` is verified for both an internal FRITZ!Box extension
  (`**613`, 18:27 UTC) and an external mobile call (`015172420641`, 18:33 UTC).
- [ ] **`TRUSTED_CALLERS` must use the national format the FRITZ!Box sends.**
  The external call logged `015172420641`, not E.164 `+4915172420641`.
  `LlmClient._is_trusted` (`agent/llm.py:184`) does an exact string match
  against the parsed allowlist (`agent/config.py:184`), so an E.164 entry
  silently fails closed — the caller converses but gets no RAG/calendar access
  and nothing indicates why. Either document the exact-format requirement in
  `.env.example` next to `TRUSTED_CALLERS`, or normalize both sides before
  comparing. Decide before the allowlist is first populated; it is empty today.

## Real-time correctness (highest value for call quality)

- [x] **#3 Streaming pipeline.** LLM streams tokens → German-aware
  `SentenceSegmenter` batches sentences → stable `tts.synthesize` returns one
  whole 24 kHz WAV per sentence → 16 kHz PCM → bounded PJSIP playback.
  `PROCESSING`/`SPEAKING` overlap; SPEAKING enters on the first sentence;
  barge-in (also interruptible during PROCESSING) cancels the per-turn
  `TaskGroup`. STT stays whole-turn (Qwen3-ASR is fine offline).
  Spec: `docs/superpowers/specs/2026-05-31-streaming-pipeline-design.md`,
  plan: `docs/superpowers/plans/2026-05-31-streaming-pipeline.md`.
  `agent/{segmenter,tts,llm,rtp,pipeline,ari,main}.py`.
- [x] **#4 RTP hardening (header validation).** `parse_rtp_payload` now
  validates the RFC 3550 header — version check, CSRC list, header extension,
  and padding — and returns `b""` on anything malformed so a bad datagram can
  no longer feed garbage into the VAD buffer. (Still deferred: jitter buffer +
  PLC + sequence/timestamp tracking — those need a real-time receive path.)
  `agent/rtp.py`.
- [x] **#5 Playback clock drift.** `stream_audio` now paces each frame against
  an absolute monotonic schedule (`start + n*20ms`) instead of
  `sleep(0.02)`-after-work, so per-frame work and event-loop jitter no longer
  accumulate drift. `agent/rtp.py`.

## Concurrency / lifecycle

- [x] **#8 Barge-in turn race.** Added a per-call turn lock + monotonic
  generation id. `_on_audio` runs the cancel-playback / process-turn /
  start-playback section under the lock; both the turn and `_play_audio` carry
  the generation they were started for and no-op if a newer barge-in has
  superseded them, so a cancelled turn can never emit stale audio.
  `agent/ari.py`.
- [x] **#9 Per-packet task churn.** The datagram callback now only enqueues onto
  a bounded per-call `asyncio.Queue` (drops on overflow); a single consumer
  task drains it and runs `_on_audio` serially, surfacing handler exceptions in
  one place instead of spawning a task per packet. `agent/ari.py`.
- [x] **#12 Resource lifecycle.** One shared `httpx.AsyncClient` per process
  (injected into STT/TTS/LLM/RAG and reused inside ARI); the pg pool, the
  shared client, and the ARI client are closed in `main()`'s `finally`; the ARI
  websocket reconnects with capped exponential backoff; RTP bind now retries
  the next port on collision instead of trusting the counter.
  `agent/stt.py`, `agent/tts.py`, `agent/llm.py`, `agent/tools/rag.py`,
  `agent/main.py`, `agent/ari.py`. (Still deferred: jitter buffer.)

## Models

- [x] **faster-qwen3-tts (stable named clone production path).** Production
  uses the private `shared-female-de-v1` profile with the pinned Qwen Base
  clone runtime. Each response sentence uses stable `/v1/audio/speech`
  whole-WAV synthesis at 24 kHz PCM16; diagnostic
  `/v1/audio/speech/stream` remains outside the VoIP response path.
  Cooperative cancellation now stops active synthesis between codec steps,
  synchronizes CUDA before releasing the exclusive model lock, and lets the
  following request proceed without waiting for the cancelled sentence.
  GX10 verification on 2026-07-30 covered the image-build patch verifier,
  healthy zero-restart deployment, authenticated stable WAV synthesis,
  cancellation/recovery, and real GPU activity.
  Raw diagnostic-stream wire shape and vLLM streamed `delta.tool_calls`
  fragments remain unit-fixture assumptions; verify those paths live before
  treating them as production-proven contracts.
- [ ] **Parakeet ASR: do NOT switch yet.** Qwen3-ASR already streams; the code
  just uses it offline/batch. Parakeet-TDT v3 is also offline per NVIDIA NIM
  docs. If NVIDIA is ever wanted, use Parakeet RNNT Multilingual (streaming) via
  NIM/Riva — but fix the streaming path (#3) first.

## Residual security (lower priority)

- [x] **Scope `NVIDIA_VISIBLE_DEVICES`** off `all` in production. Env knob
  `${NVIDIA_VISIBLE_DEVICES:-all}` on all three DGX services + documented in
  `dgx/.env.example` (set `NVIDIA_VISIBLE_DEVICES=0`). Actual scoping is a
  deploy-time value, not a code change.
- [x] **Bind SIP / inference ports** to the LAN interface + firewall. Env knobs
  in place: `DGX_BIND_IP` (`dgx/.env.example`); `asterisk/pjsip.conf` documents
  the LAN bind. Firewalling is deploy-time.
- [x] **`external_host` advertised a bind address, not a routable media
  destination** for Asterisk ExternalMedia (#7). Split bind from advertise: new
  `rtp_advertise_host` config (default `127.0.0.1`, validator rejects `0.0.0.0`
  and empty); `_create_external_media` now sends the advertise host. Tests
  assert the POST advertises the routable host and the validator fails closed.
  `agent/config.py`, `agent/ari.py`.
