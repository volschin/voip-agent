# Smart Turn v2 — DGX service contract

Consumer: `agent/turn_detector.py` (`TurnDetectorClient`). Model:
`pipecat-ai/smart-turn-v2` (wav2vec2-based end-of-turn classifier, multilingual
incl. German). Build/deploy of this container is tracked separately; this file
pins the wire contract the agent depends on.

## Endpoint

`POST /v1/turn/classify`

- Request body: raw little-endian **int16 PCM, mono, 16 kHz** (no WAV header).
  Content-Type `application/octet-stream`. Represents the caller's buffered
  speech since their last endpoint candidate.
- Response: `200` JSON `{"complete": bool, "prob": float}` where `prob` is the
  probability in `[0,1]` that the turn is complete. The client compares `prob`
  against `turn_complete_threshold` (default 0.5); `complete` is advisory.
- Latency: must answer within the client budget (`turn_classify_timeout_ms`,
  default 150 ms) or the client times out and degrades to a silence flush.

## Default port

`8004` (see `turn_detector_url` default `http://dgx-spark:8004`). Add the
container to `dgx/docker-compose.yml` analogous to the TTS service, with a
health check.

## Rollout gate

`turn_detection_enabled` stays `False` until German precision is verified live
against this service (multilingual != verified for German — cf. the TTS
`language` full-names lesson).
