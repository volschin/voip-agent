# Direct PJSIP Migration

The production entry point now registers directly as a FRITZ!Box LAN/Wi-Fi IP telephone.
Asterisk, ARI, and ExternalMedia are no longer in the runtime path. The former `agent/ari.py`,
`agent/rtp.py`, and `asterisk/` files remain temporarily as a rollback reference.

## Runtime Flow

1. PJSUA2 registers with the FRITZ!Box over local SIP/UDP.
2. An incoming call receives `180 Ringing` while the normal handsets continue ringing.
3. A handset pickup produces `CANCEL`; otherwise the agent answers after
   `ANSWER_DELAY_SECONDS`.
4. PJSUA2 terminates the negotiated RTP codec into 16 kHz mono PCM through a custom audio port.
5. `ConversationManager` runs VAD, turn detection, barge-in, STT, LLM, and TTS.
6. Each completed LLM sentence is synthesized through `/v1/audio/speech`,
   WAV-decoded, converted once from 24 to 16 kHz PCM, and written to the
   existing bounded playback queue. `/v1/audio/speech/stream` is not used by
   production VoIP playback.
7. PJSIP starts playback after a 300 ms (9600-byte) prebuffer and pulls the
   pipeline PCM directly; there is no A-law round trip in production.

Completed LLM sentences use a bounded two-entry prefetch queue. TTS generation
remains sequential, with the existing two-second maximum-ahead playback bound,
and barge-in cancels the producers, clears prebuffered PCM, and stops the
stable TTS decode between codec steps so the replacement turn does not wait on
the cancelled sentence's model lock. The incomplete assistant turn is omitted
from persistent history; the next LLM request receives one transient
interruption context. The legacy ARI/RTP rollback path alone converts pipeline
PCM to 8 kHz G.711 A-law at its transport boundary.

## Deployment

Keep AI settings, integration settings, and the dedicated IP-telephone secret together in
`.env` — the production compose file reads no other env file. `.env.pjsip-poc` is scoped to
the signalling-only `compose.pjsip-poc.yml`. Both files are ignored by Git.

```bash
docker compose -f compose.pjsip-poc.yml down
docker compose up --detach --build
docker compose logs --follow voip-agent
```

The service uses host networking for SIP/RTP, runs as UID 10001 with all
capabilities dropped, and keeps its root filesystem read-only. It mounts the
agent model-cache volume plus three read-only files:
`shared_ai_password`, `mate_ca.crt`, and `voice_priority_token`. Password and
token files must be regular, non-symlink files owned for UID/GID 10001 with
mode `0400` (no group/other permissions); the CA must be a regular non-symlink
file and must not be group/world writable.

## Acceptance Checks

- Registration reaches `200 OK` and renews after five minutes.
- Human pickup cancels the agent leg before the answer timer.
- An unanswered call produces `PJSIP audio bridge active` and plays the greeting.
- Caller speech reaches STT; a response is played and can be interrupted.
- Hangup releases the conversation and accepts a subsequent call.

Use `PJSIP_LOG_LEVEL=2` in normal operation. Higher levels are intended only for
short-lived SIP diagnostics because raw SIP and media details can contain caller
numbers and headers beyond the number itself.

`agent.answer_policy` deliberately logs the caller number at INFO for every
lifecycle step — offer, answer, duration limit, cancel, and hangup — so a call
can be traced end to end. Only the SIP user extracted by `caller_id_from_uri`
is recorded, never the raw `remoteUri`; an unavailable number logs as
`unknown`. Application logs therefore contain caller numbers regardless of
`PJSIP_LOG_LEVEL`; retain and ship them accordingly.
