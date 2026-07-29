# Stable Synthetic German Voice Clone Design

**Status:** Approved on 2026-07-29

## Problem

The production VoIP agent and Companion currently send a natural-language voice
description to `Qwen3-TTS-12Hz-1.7B-VoiceDesign`. Each greeting and each
sentence-sized streaming segment triggers an independent VoiceDesign
generation. The deployed faster-qwen3-tts runtime samples by default, so those
requests can produce different speaker identities and can alternate between
female and male voices during one call.

The service needs one stable, synthetic, adult female German speaker. The VoIP
agent and Mira may share that speaker initially, but voice selection must use a
named profile so Mira can receive a different profile later without changing
the TTS wire contract.

## Goals

- Create a wholly synthetic reference voice rather than cloning a real person.
- Keep one recognizably stable speaker across greetings, sentence segments,
  turns, calls, and Companion speech.
- Preserve non-streaming WAV and streaming raw 24-kHz signed-16-bit PCM
  endpoints.
- Preserve sentence-level streaming and current time-to-first-audio behavior.
- Make the selected voice an explicit profile named
  `shared-female-de-v1`.
- Keep the voice stack CUDA-only on NVIDIA GB10 and fail closed on missing
  models, profiles, or CUDA.
- Allow Mira to move to a future `mira-female-de-v1` profile by configuration.

## Non-goals

- Cloning a real person's voice or importing Common Voice, LibriVox,
  M-AILABS, Piper, or another third-party speaker.
- Training or fine-tuning a TTS model.
- Changing ASR, LLM, SIP, RTP, VAD, sentence segmentation, barge-in, or voice
  priority behavior.
- Loading VoiceDesign and Base models concurrently in production.
- Creating Mira's final distinct voice in this change.

## Selected Approach

Use Qwen's documented **Voice Design then Clone** workflow:

1. Generate five reference candidates with the existing deployed
   `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` model at immutable revision
   `5ecdb67327fd37bb2e042aab12ff7391903235d3`.
2. Select one candidate and retain it as a versioned synthetic reference.
3. Replace the production VoiceDesign model with
   `Qwen/Qwen3-TTS-12Hz-1.7B-Base` at immutable revision
   `fd4b254389122332181a7c3db7f27e918eec64e3`.
4. Build and cache the voice-clone prompt from the selected reference, then use
   `generate_voice_clone` and `generate_voice_clone_streaming` for every
   request.

This is preferred over the `Serena` CustomVoice preset because Serena's native
language is Chinese and the voice would not be unique. Disabling sampling on
VoiceDesign is also rejected: it does not create a reusable speaker identity
and does not address the per-request redesign boundary.

## Synthetic Reference Creation

All candidates use this exact transcript:

> Guten Tag, hier ist Ihre digitale Assistentin. Ich höre Ihnen aufmerksam zu
> und helfe Ihnen ruhig, klar und zuverlässig weiter.

All candidates use this exact design instruction:

> Eine erwachsene Frau zwischen etwa dreißig und vierzig Jahren mit warmer,
> natürlicher und vertrauenswürdiger Stimme. Klares neutrales Hochdeutsch ohne
> erkennbaren Regionalakzent, ruhiges mittleres Sprechtempo, deutliche, aber
> nicht überbetonte Artikulation und sanfte lebendige Intonation. Professionell
> und freundlich, weder kindlich noch werblich, nicht hauchig und nicht
> flirtend.

Candidate selection is autonomous and uses both listening and objective gates:

- mono 24-kHz PCM WAV with no malformed samples;
- exact or semantically identical German ASR round trip;
- no clipping above 0.1 percent of samples;
- no audible noise, glitches, code-switching, duplicated words, or truncated
  phonemes;
- adult-female fundamental-frequency distribution, using a median target of
  165 through 240 Hz as a heuristic rather than the sole decision;
- calm, natural pace and clear Standard German;
- warm and trustworthy delivery without sounding childish, flirtatious, or
  like an advertisement.

The selected artifact is stored at:

`dgx/tts/profiles/shared-female-de-v1.wav`

`dgx/tts/profiles/profiles.json` records:

- profile ID;
- relative reference-audio path;
- exact reference transcript and language;
- source type `synthetic-qwen-voice-design`;
- source model and immutable revision;
- SHA-256 of the WAV;
- generation and selection date;
- the approved design instruction.

The generated WAV and manifest are committed. No downloaded human speech is
stored or used.

## Runtime Architecture

The independently owned GX10 `voice` Portainer stack remains the owner of
Qwen3-ASR and Qwen3-TTS. Its stack identity, internal networks, GPU device
reservation, health policy, and lack of host-published ports remain unchanged.

The TTS image contains the profile manifest and reference WAV. At startup the
server:

1. requires CUDA and verifies NVIDIA GB10;
2. loads the revision-pinned Base model from the local Hugging Face cache;
3. validates the manifest schema, profile ID, reference transcript, WAV format,
   and SHA-256;
4. creates or warms faster-qwen3-tts's clone-prompt cache from the reference;
5. reports healthy only after `shared-female-de-v1` is usable.

Only the Base model remains loaded after rollout. VoiceDesign is used to create
the committed reference before the model switch and is not retained as a
production dependency.

Both endpoints resolve a named profile and call the matching clone operation:

- `POST /v1/audio/speech` calls `generate_voice_clone` and returns WAV;
- `POST /v1/audio/speech/stream` calls
  `generate_voice_clone_streaming` and returns raw 24-kHz signed-16-bit
  little-endian PCM.

The reference path and transcript stay constant, allowing faster-qwen3-tts's
internal prompt cache to reuse extracted reference features across requests.
Sentence segmentation remains unchanged; every sentence is independently
synthesized against the same clone prompt rather than independently designing
a new speaker.

## API and Configuration

The existing OpenAI-style `voice` request field becomes a profile selector.
The canonical value is `shared-female-de-v1`.

During the coordinated rollout, the server temporarily recognizes the two
currently deployed natural-language descriptions as aliases for
`shared-female-de-v1`. This prevents a partial rollout from restoring
VoiceDesign behavior or breaking a caller. Unknown non-empty values return a
client error; they never fall back to free-form voice design.

The compatibility aliases are exactly:

- `Eine warme, natürliche deutsche Stimme mit angemessenem Sprechtempo.`
- `Eine warme, natürliche, erwachsene weibliche deutsche Stimme mit klarer
  Aussprache, lebendiger Intonation und ruhigem Sprechtempo.`

An unknown profile returns HTTP 422 with the bounded detail
`unsupported voice profile`.

The VoIP agent gains `TTS_VOICE_PROFILE`, defaulting to
`shared-female-de-v1`. Its non-streaming and streaming clients send the same
profile.

Companion Core gains `COMPANION_VOICE_PROFILE`, also defaulting to
`shared-female-de-v1`. Its trusted voice adapter sends that profile instead of
the natural-language instruction. A later Mira voice change only adds a second
validated server profile and changes this Companion setting; the VoIP setting
remains unchanged.

## Failure Handling

- Missing Base-model snapshot: startup fails while offline; no network fallback.
- Missing, malformed, or hash-mismatched profile asset: startup fails.
- CPU-only execution or a non-GB10 CUDA device: startup fails.
- Unknown profile ID: request fails with a bounded client error.
- Clone-prompt construction failure: health remains unhealthy.
- Empty, odd-byte, malformed, or failed TTS output: the current callers retain
  their existing fail-closed/fallback behavior.
- Cancellation propagates through the streaming generator exactly as it does
  today; no detached synthesis continues after barge-in or priority
  cancellation.

No transcript, generated call audio, or caller audio is persisted by the
runtime. Only the synthetic reference asset is persistent.

## Repository Boundaries

The voice server, synthetic reference, model pin, deployment contract, and
VoIP profile setting belong to the existing isolated
`voip-agent` branch `feat/shared-ai-traefik`.

Companion's profile setting and trusted adapter change belong to an isolated
`ai-companion` feature branch. The repositories retain their independent
ownership; deployment coordinates their compatible commits without copying
secrets or merging stack ownership.

## Tests

### Unit and contract tests

- Profile-manifest parsing accepts the committed profile and rejects missing,
  duplicate, malformed, path-escaping, and hash-mismatched entries.
- Both server endpoints resolve the same profile and invoke clone generation,
  not VoiceDesign generation.
- Non-streaming output stays a valid 24-kHz PCM WAV.
- Streaming output stays even-length raw signed-16-bit PCM.
- Unknown profiles fail closed.
- Health requires CUDA, the Base model, and a prepared default profile.
- VoIP sends `shared-female-de-v1` for greeting and streaming turns.
- Companion sends its configured profile and retains its fixed German language.
- Deployment tests pin the Base snapshot, mount or copy the profile assets, keep
  NVIDIA device reservations, and publish no backend ports.

### Local verification

- Complete VoIP pytest suite on Python 3.12 environment.
- Complete Companion pytest suite.
- Ruff check and format check for both repositories.
- Docker Compose configuration validation.
- TTS image build on the GX10 architecture.

## Live Acceptance

The rollout is accepted only when fresh evidence proves:

1. The existing Portainer `voice` stack identity is preserved and its ASR/TTS
   containers are healthy with restart count zero.
2. The TTS container reports the pinned Base revision, loaded
   `shared-female-de-v1`, NVIDIA GB10 CUDA, and no CPU fallback.
3. Non-streaming and streaming endpoints return valid, non-empty audio through
   authenticated Traefik.
4. At least five varied German utterances synthesized in separate requests have
   one stable female speaker by listening and speaker-embedding similarity.
   The verification-only model is `microsoft/wavlm-base-plus-sv` at revision
   `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`. Reference-to-output cosine
   similarity must have a median of at least 0.80 and a minimum of at least
   0.70; pairwise output similarity must have a median of at least 0.85.
5. ASR round trips remain intelligible and do not introduce code-switching.
6. Warm first-audio latency remains within the existing operational budget.
7. GPU activity is time-correlated with real TTS work.
8. Companion cancellation and VoIP priority still interrupt or reject work as
   specified.
9. A real inbound call proves the greeting, multiple sentence segments, and
   later turns retain the same speaker until clean hangup.

Image-level CUDA availability, HTTP 200 responses alone, or synthetic unit
fixtures do not satisfy live acceptance.

## Rollback

Before deployment, retain the current `voice` stack Compose, TTS image ID, and
VoiceDesign snapshot. If Base-model loading, latency, voice quality, priority,
or call acceptance fails, restore the previous image and Compose without
changing Traefik, ASR, or the clients' network topology. Rollback is reported as
a failed clone-profile rollout, not as successful completion.
