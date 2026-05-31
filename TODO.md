# TODO — deferred hardening

Tracked from an adversarial codex review (2026-05-30). The contract bugs
(#1/#2/#10) and security findings (#11 read+write, #13 network/creds) are fixed
on branch `fix/contract-and-fsm-bugs`. The real-time-correctness and
concurrency/lifecycle items (#4/#5/#8/#9/#12) are fixed on branch
`feat/realtime-hardening`. What remains below is the streaming rewrite (#3 +
faster-qwen3-tts) — blocked on the TTS server, which is non-streaming
(`dgx/tts/server.py` runs `non_streaming_mode=True`) — and the residual
security config. Numbering matches the original review.

## Real-time correctness (highest value for call quality)

- [ ] **#3 Streaming pipeline.** Turn loop is fully serialized and non-streaming:
  VAD waits 800 ms of silence, then STT → LLM → TTS run end to end before any
  audio plays. Phone target (~300–800 ms to first audio) is impossible. Stream
  STT partials, stream LLM tokens into TTS, stream TTS first-chunk into RTP.
  Pairs with the faster-qwen3-tts adoption below. `agent/audio.py:29`,
  `agent/pipeline.py`.
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

- [ ] **faster-qwen3-tts (with streaming).** TTFA on DGX Spark 567→280 ms (0.6B),
  661→400 ms (1.7B) via CUDA graphs; supports streaming chunks. Worth adopting,
  but only alongside #3 — change the TTS contract from "POST text → WAV" to
  "stream text → yield 24 kHz PCM chunks → downsample/RTP immediately". A
  non-streaming drop-in leaves the latency bug intact.
  https://github.com/andimarafioti/faster-qwen3-tts
- [ ] **Parakeet ASR: do NOT switch yet.** Qwen3-ASR already streams; the code
  just uses it offline/batch. Parakeet-TDT v3 is also offline per NVIDIA NIM
  docs. If NVIDIA is ever wanted, use Parakeet RNNT Multilingual (streaming) via
  NIM/Riva — but fix the streaming path (#3) first.

## Residual security (lower priority)

- [ ] **Scope `NVIDIA_VISIBLE_DEVICES`** off `all` in production (set per service).
- [ ] **Bind SIP / inference ports** to the LAN interface + firewall (env knobs
  added: `DGX_BIND_IP`, `dgx/.env.example`; `asterisk/pjsip.conf` documented).
- [ ] **`external_host=0.0.0.0` is a bind address, not a routable media
  destination** for Asterisk ExternalMedia (#7). Confirm the agent advertises a
  reachable IP. `agent/config.py:19`, `agent/ari.py:153`.
