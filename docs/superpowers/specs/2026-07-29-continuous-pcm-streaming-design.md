# Continuous PCM Streaming Design

## Problem and evidence

The production PJSIP path already exposes a 16 kHz signed-16-bit mono media
port, but the voice pipeline still emits the legacy Asterisk transport format:

```text
TTS 24 kHz PCM
  -> independent scipy 24->8 kHz resample for every HTTP chunk
  -> G.711 A-law
  -> independent scipy 8->16 kHz resample for every playback chunk
  -> PJSIP PCM port
```

The 2026-07-29 live calls were SIP-stable and logged no queue drops, media
errors, or streaming exceptions. A representative TTS request produced 9.84
seconds of audio in 5.61 seconds, so sustained generation was faster than
playback. The same request arrived as 51 small chunks. A second continuity
probe found that 35 of 38 chunk lengths were not divisible by the 24-to-8 kHz
ratio. Restarting both polyphase filters at those arbitrary boundaries added
samples, reset filter state, and produced severe discontinuities. Five
sequential short sentences also exposed 222 ms of playback underrun.

The repair must fix the media contract and timing behavior, not hide the
artifacts by increasing TTS chunk size.

## Goals

- Preserve one continuous resampler state across every HTTP chunk belonging to
  one TTS utterance.
- Deliver signed-16-bit little-endian mono 16 kHz PCM directly to the production
  PJSIP sink.
- Remove the 8 kHz A-law encode/decode round trip from production playback.
- Start playback with a bounded 300 ms PCM prebuffer.
- Keep up to two completed LLM sentence segments queued while TTS/playback
  proceeds.
- Preserve cancellation, barge-in, state-machine transitions, fallback speech,
  priority leases, and bounded queues.
- Preserve the legacy ARI/RTP rollback path by converting 16 kHz pipeline PCM to
  8 kHz A-law only inside the ARI transport adapter.

## Non-goals

- Changing the selected voice profile or Qwen model.
- Changing inbound caller audio, VAD, Smart Turn, STT, or SIP registration.
- Adding a new resampling dependency.
- Building a general jitter buffer or packet-loss concealment system.
- Running more than one TTS model generation concurrently.

## Considered approaches

### 1. Stateful direct PCM streaming (selected)

Use `audioop.ratecv` with retained state for 24-to-16 kHz and, only in the
legacy ARI adapter, 16-to-8 kHz. The repository already depends on
`audioop-lts`. A controlled 440 Hz probe proved that arbitrary chunking produces
byte-identical output to one continuous `ratecv` call. Its RMS difference from
the current one-shot polyphase reference was 14.8 against a signal RMS of
8491.

This approach preserves low time-to-first-audio, removes the lossy A-law round
trip from PJSIP, and has the smallest stateful implementation surface.

### 2. Buffer each sentence and resample once

This avoids intra-sentence filter resets but discards audio-output streaming and
adds the full sentence synthesis time to latency. It does not meet the latency
goal.

### 3. Increase or align HTTP chunks

Rechunking to multiples of the sample-rate ratio reduces rounding errors but
still restarts the FIR filter and remains dependent on HTTP framing. Network
chunks are not an audio contract, so this is rejected.

## Audio contracts

### Stateful resampler

`agent.audio.StreamingPcm16Resampler` owns:

- input and output sample rates;
- `audioop.ratecv` state;
- validation that input is one-dimensional signed-16-bit PCM;
- `process(samples: numpy.ndarray) -> bytes`;
- no implicit silence insertion.

One instance belongs to one logical utterance. The TTS adapter creates it
before consuming `/v1/audio/speech/stream` and discards it after the stream
ends or is cancelled. Arbitrary chunk sizes, including one-sample chunks, must
produce exactly the same bytes as a one-shot call through the same resampler.

### Voice pipeline

The production-neutral pipeline contract becomes signed-16-bit little-endian
mono 16 kHz PCM:

- `synthesize_pcm16(text: str) -> bytes`;
- `process_turn(...) -> bytes`;
- `process_turn_stream(...) -> AsyncIterator[bytes]`.

The pipeline no longer imports or calls A-law encoding for outbound speech.
Fallback prompts use the same PCM contract. Empty output remains a bounded
failure, never a fabricated audio frame.

### PJSIP sink

`PjsipAudioSink` accepts 16 kHz PCM bytes and writes them directly to
`PcmPlaybackBuffer`. It must reject odd byte counts. The PJSUA2 port continues
to request 20 ms, 16 kHz, mono, 16-bit frames and remains responsible for codec
negotiation with the FRITZ!Box.

### Legacy ARI adapter

`AriClient` adapts pipeline PCM at its transport boundary:

- complete PCM responses use one 16-to-8 kHz conversion followed by A-law
  encoding;
- streamed responses retain one 16-to-8 kHz resampler state for the complete
  turn, then A-law-encode each converted result;
- `RtpServer` and its payload/pacing contract remain unchanged.

This keeps rollback functional without leaking A-law back into the production
pipeline.

## Prebuffer and sentence prefetch

`PjsipAudioSink.play_pcm16_chunks` accumulates up to 300 ms (9600 bytes) before
the first write. If the entire response is shorter, end-of-stream releases it
immediately. After playback starts, the existing two-second maximum-ahead
backpressure remains in force.

LLM token segmentation moves into its producer task. Completed sentences enter
a bounded queue of depth two while a TTS worker consumes them in order. This
does not run simultaneous model generations; it ensures the next sentence is
already identified and waiting when the current TTS stream releases the model.
Tool-round filler has priority only once, as today.

The TTS worker emits all PCM chunks into the existing bounded playback queue.
On barge-in or call teardown, cancellation must close the LLM producer, segment
queue, current TTS iterator, PCM queue, and sink playback without leaving a
queued model request.

## Error handling

- Odd-length PCM from any boundary is rejected with a bounded error.
- A resampler may not be reused after its utterance is complete.
- LLM or TTS failure keeps the existing recovery-prompt behavior.
- Producer failure always terminates both bounded queues.
- A cancelled turn clears prebuffered and already-buffered PCM immediately.
- No exception may strand the session in `PROCESSING` or `SPEAKING`.

## Verification

Automated tests must prove:

1. Stateful 24-to-16 conversion is invariant under arbitrary chunk splits.
2. A non-multiple-of-three split produces no extra or missing output samples.
3. The PJSIP sink writes PCM unchanged and never invokes A-law conversion.
4. Playback does not begin before 300 ms is ready, except for a shorter
   end-of-stream response.
5. Barge-in clears the prebuffer and cancels all producer tasks.
6. Sentence segmentation can run ahead by at most two entries while output
   ordering remains exact.
7. The ARI adapter still produces paced 8 kHz A-law RTP.
8. Existing state-machine, priority, TTS, PJSIP, ARI, and cancellation tests
   remain green.

Live acceptance must prove:

- the deployed container uses the exact implementation commit;
- SIP registration is `200 OK`, restart count is zero, and no queue/drop/media
  errors occur;
- a representative long stream has zero predicted PCM underruns and its
  chunked conversion is identical to one-shot stateful conversion;
- five short sentences produce no predicted playback gap after the initial
  prebuffer;
- authenticated TTS remains healthy and first playable PCM stays below
  1.3 seconds;
- a real call confirms smooth greeting, multi-sentence reply, barge-in, and
  clean hangup.

The real call is a mandatory final gate because signal metrics cannot replace
auditory acceptance.
