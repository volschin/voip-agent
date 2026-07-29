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
6. Each Qwen TTS utterance is converted continuously from 24 to 16 kHz PCM with
   resampler state retained across HTTP chunks.
7. PJSIP starts playback after a 300 ms (9600-byte) prebuffer and pulls the
   pipeline PCM directly; there is no A-law round trip in production.

Completed LLM sentences use a bounded two-entry prefetch queue. TTS generation
remains sequential, and barge-in cancels the producers and clears prebuffered
PCM. The legacy ARI/RTP rollback path alone converts pipeline PCM to 8 kHz
G.711 A-law at its transport boundary.

## Deployment

Keep AI and integration settings in `.env` and the dedicated IP-telephone secret in
`.env.pjsip-poc`. Both files are ignored by Git.

```bash
docker compose -f compose.pjsip-poc.yml down
docker compose up --detach --build
docker compose logs --follow voip-agent
```

The service uses host networking for SIP/RTP, runs as UID 10001 with all capabilities dropped,
and mounts only a model-cache volume into its read-only filesystem.

## Acceptance Checks

- Registration reaches `200 OK` and renews after five minutes.
- Human pickup cancels the agent leg before the answer timer.
- An unanswered call produces `PJSIP audio bridge active` and plays the greeting.
- Caller speech reaches STT; a response is played and can be interrupted.
- Hangup releases the conversation and accepts a subsequent call.

Use `PJSIP_LOG_LEVEL=2` in normal operation. Higher levels are intended only for
short-lived SIP diagnostics because SIP and media details can contain caller numbers.
