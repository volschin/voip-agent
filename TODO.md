# TODO — deferred hardening

Tracked from an adversarial codex review (2026-05-30). The contract bugs
(#1/#2/#10) and security findings (#11 read+write, #13 network/creds) are fixed
on branch `fix/contract-and-fsm-bugs`. The items below were deferred from that
review — they need real-time/architecture work, not one-line fixes. Numbering
matches the original review.

## Real-time correctness (highest value for call quality)

- [ ] **#3 Streaming pipeline.** Turn loop is fully serialized and non-streaming:
  VAD waits 800 ms of silence, then STT → LLM → TTS run end to end before any
  audio plays. Phone target (~300–800 ms to first audio) is impossible. Stream
  STT partials, stream LLM tokens into TTS, stream TTS first-chunk into RTP.
  Pairs with the faster-qwen3-tts adoption below. `agent/audio.py:29`,
  `agent/pipeline.py`.
- [ ] **#4 RTP hardening.** `agent/rtp.py` strips 12 bytes and calls it RTP — no
  version/PT/SSRC validation, no CSRC/extension/padding handling, no
  sequence/timestamp tracking, no jitter buffer, no packet-loss concealment, no
  payload-length check. A malformed packet can poison VAD or kill an unobserved
  task. Validate headers; add a jitter buffer + PLC. `agent/rtp.py:9,50`.
- [ ] **#5 Playback clock drift.** `await asyncio.sleep(0.02)` paces frames
  relative to work done, not an absolute clock; event-loop delay accumulates
  while RTP timestamps advance as if perfect. Pace against a monotonic clock.
  `agent/rtp.py:70`.

## Concurrency / lifecycle

- [ ] **#8 Barge-in turn race.** `_on_audio` accepts audio during SPEAKING and a
  second invocation can start `process_turn` while the first is still
  synthesizing. No per-call turn lock or playback generation id. Add a per-call
  lock + generation token so a cancelled turn cannot emit stale audio.
  `agent/ari.py:119,134`, `agent/pipeline.py`.
- [ ] **#9 Per-packet task churn.** Every inbound RTP packet does
  `loop.create_task(_on_audio(...))` (~50/s/call) with no backpressure and
  unobserved exceptions. Use a per-call queue/consumer; surface task errors.
  `agent/ari.py:72`.
- [ ] **#12 Resource lifecycle.** `httpx.AsyncClient` is recreated per
  STT/TTS/LLM/RAG/ARI request; asyncpg pool is never closed; ARI websocket has
  no reconnect loop; RTP port allocation wraps without checking live binds.
  Share long-lived clients, close the pool on shutdown, add ws reconnect + bind
  collision handling. `agent/stt.py:23`, `agent/llm.py`, `agent/main.py:25`,
  `agent/ari.py:37,63`.

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
