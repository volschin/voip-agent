# Stable Sentence Synthesis Design

## Problem and evidence

The 2026-07-29 real-call acceptance disproved the assumption that continuous
PCM conversion alone would make conversational output clean. The greeting was
clean, while every generated response remained chopped and contained repeated
or reordered fragments.

That comparison isolates the differing TTS source path:

- the greeting uses `POST /v1/audio/speech`, which calls Qwen voice cloning in
  stable full-utterance mode;
- response sentences use `POST /v1/audio/speech/stream`, which calls the
  experimental codec streaming generator;
- both paths use the same 24-to-16 kHz conversion, PCM prebuffer, PJSIP media
  port, negotiated telephone codec, and FRITZ!Box connection.

Controlled ASR and sample-boundary probes did not reproduce the audible defect,
so automated transcription is insufficient as an acceptance oracle. The clean
greeting through the common output chain is the strongest boundary test.

## Selected repair

Keep LLM token streaming and sentence segmentation. For every completed
sentence, call the stable full-utterance TTS operation already used by the
greeting, decode its WAV, resample it once to 16 kHz PCM16, and yield the
complete sentence to the existing bounded playback queue.

The response still plays incrementally by sentence. Sentence order remains
owned by the existing single consumer, and the PJSIP sink retains its 300 ms
prebuffer and two-second maximum-ahead bound. No model requests run
concurrently.

The experimental `/v1/audio/speech/stream` endpoint and
`TtsClient.synthesize_stream()` remain available for diagnostics and other
consumers, but the VoIP response pipeline no longer uses them.

## Alternatives considered

### Repair the experimental codec stream

This preserves the lowest possible time to first audio, but the controlled
probes did not expose a deterministic boundary defect to repair. Changing its
hybrid codec decoder without a reproducer would be speculative and would keep
the broken component in the production path.

### Generate a full sentence inside the server's streaming endpoint

This should match stable quality, but it changes the shared server contract
without improving cancellation: model generation still finishes before audio
can be returned. Keeping the repair in the VoIP pipeline is smaller and uses an
already proven endpoint.

## Latency and cancellation

Playback begins only after the first sentence has been fully synthesized.
Expected time to first audio therefore increases by the remaining synthesis
time for that sentence. This is accepted in favor of intelligible, correctly
ordered speech.

Cancelling the HTTP client during full synthesis cannot necessarily stop the
synchronous model call already running on the server. The local turn and
playback tasks must still cancel immediately and discard the late response.
The next TTS request may wait for the current sentence generation to release
the server's model lock. This bounded server-side completion delay is accepted
for the repair and must be recorded in live barge-in validation.

## Code and contract changes

- `VoicePipeline._tts_pcm16_chunks(text)` calls `synthesize_pcm16(text)` and
  yields its non-empty result once.
- `VoicePipeline.process_turn_stream(...)` keeps sentence prefetch, strict
  sequential consumption, history behavior, recovery speech, and cancellation.
- `TtsClient`, the shared Qwen service, audio adapters, PJSIP sink, ARI rollback
  path, configuration, and voice profile remain unchanged.
- Documentation identifies stable per-sentence synthesis as the VoIP response
  path and the raw streaming endpoint as non-production for VoIP.

## Verification

Automated tests must prove:

1. Response sentences use the stable whole-WAV callable and never invoke the
   experimental TTS stream callable.
2. Multiple sentence segments are synthesized and emitted exactly once in
   source order.
3. Empty stable TTS output emits no fabricated audio.
4. TTS failure still emits the stable recovery prompt and leaves no partial
   assistant history entry.
5. Cancellation promptly closes the local producer and playback chain.
6. The full unit, lint, format, and Compose checks remain green.

Live acceptance must prove:

- the deployed container is built from the exact repair commit;
- SIP remains registered with zero container restarts and no playback, queue,
  or media errors;
- the first sentence uses `/v1/audio/speech`, not
  `/v1/audio/speech/stream`;
- a multi-sentence response is intelligible, ordered, and free from repeated
  fragments;
- barge-in clears local playback promptly and the following turn recovers;
- the greeting and clean hangup remain intact.

Auditory confirmation from the real call is the final acceptance gate.
