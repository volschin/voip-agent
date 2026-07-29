# Stable Private German Voice Clone Design

**Status:** Revised and approved on 2026-07-29

## Problem

The production VoIP agent and Companion currently send a natural-language voice
description to `Qwen3-TTS-12Hz-1.7B-VoiceDesign`. Each greeting and each
sentence-sized streaming segment triggers an independent VoiceDesign
generation. The deployed faster-qwen3-tts runtime samples by default, so those
requests can produce different speaker identities and can alternate between
female and male voices during one call.

The service needs one stable adult female German speaker. The selected voice may
come from Qwen VoiceDesign or from a user-authorized professional-speaker sample
licensed for this private TTS use. The two sources must be compared through
generated production speech rather than by comparing their raw references. The
VoIP agent and Mira may share the winner initially, but voice selection must use
a named profile so Mira can receive a different profile later without changing
the TTS wire contract.

## Goals

- Select the better production voice through a controlled, blinded comparison
  between synthetic and authorized human-reference clone candidates.
- Restrict the authorized speaker sample and derived references to this private
  deployment; never publish them in Git, an image, logs, or build artifacts.
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

- Using any human voice without explicit TTS/voice-cloning rights for this
  private deployment.
- Redistributing, publishing, or reusing the authorized speaker sample outside
  this private assistant.
- Importing Common Voice, LibriVox, M-AILABS, Piper, or any other unapproved
  third-party speaker.
- Training or fine-tuning a TTS model.
- Changing ASR, LLM, SIP, RTP, VAD, sentence segmentation, barge-in, or voice
  priority behavior.
- Loading VoiceDesign and Base models concurrently in production.
- Creating Mira's final distinct voice in this change.

## Selected Approach

Use Qwen's documented voice-cloning workflow in a controlled A/B evaluation:

1. Generate five reference candidates with the existing deployed
   `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` model at immutable revision
   `5ecdb67327fd37bb2e042aab12ff7391903235d3`.
2. Privately derive two or three calm, clean, exactly transcribed reference
   excerpts from the authorized professional-speaker sample.
3. Replace the evaluation runtime's VoiceDesign model with
   `Qwen/Qwen3-TTS-12Hz-1.7B-Base` at immutable revision
   `fd4b254389122332181a7c3db7f27e918eec64e3`.
4. Generate the same five varied German evaluation utterances from the best
   synthetic reference and each authorized human-reference excerpt.
5. Select the better generated production voice through blinded listening and
   objective quality, stability, intelligibility, and latency gates.
6. Build and cache the voice-clone prompt from the winner, then use
   `generate_voice_clone` and `generate_voice_clone_streaming` for every
   request.

The authorized sample is not automatically preferred merely because it comes
from a professional speaker: the clone can inherit dramatic delivery or clone
less consistently than a purpose-generated neutral reference. Conversely, the
synthetic candidate is not automatically preferred for portability. The winner
must be the better generated phone-assistant voice. If final weighted scores
differ by less than three points out of 100, the synthetic candidate wins the
tie because it has fewer license and portability constraints.

The `Serena` CustomVoice preset remains rejected because Serena's native
language is Chinese and the voice would not be unique. Disabling sampling on
VoiceDesign is also rejected: it does not create a reusable speaker identity
and does not address the per-request redesign boundary.

## Candidate Creation and Evaluation

### Synthetic candidates

All five synthetic candidates use this exact transcript:

> Guten Tag, hier ist Ihre digitale Assistentin. Ich höre Ihnen aufmerksam zu
> und helfe Ihnen ruhig, klar und zuverlässig weiter.

They use this exact design instruction:

> Eine erwachsene Frau zwischen etwa dreißig und vierzig Jahren mit warmer,
> natürlicher und vertrauenswürdiger Stimme. Klares neutrales Hochdeutsch ohne
> erkennbaren Regionalakzent, ruhiges mittleres Sprechtempo, deutliche, aber
> nicht überbetonte Artikulation und sanfte lebendige Intonation. Professionell
> und freundlich, weder kindlich noch werblich, nicht hauchig und nicht
> flirtend.

### Authorized human-reference candidates

The source is the professional-speaker sample supplied by the user and
explicitly authorized by the user for private TTS/voice-cloning use. Its URL,
audio bytes, speaker identity, and derived excerpts are private deployment data
and are not recorded in this public specification.

The source is decoded without pitch shifting, time stretching, denoising, or
voice conversion. Silence boundaries and signal quality identify two or three
calm excerpts of 5 through 12 seconds. Each excerpt must contain one continuous
speaker passage without music, overlap, abrupt cuts, or strong dramatic
delivery. Qwen3-ASR supplies a draft transcript; the transcript is then checked
word-for-word against the audio before it is used as `ref_text`.

Every derived reference is converted to mono 24-kHz signed-16-bit PCM WAV.
Temporary candidates live only in a mode-0700 working directory and are deleted
after selection.

### Controlled comparison

The best synthetic reference and each authorized human-reference excerpt are
loaded into the same immutable Base model. Each profile generates the same five
fixed evaluation texts in separately seeded requests:

1. `Guten Morgen. Wie kann ich Ihnen heute helfen?`
2. `Ihr Termin ist am Donnerstag, den siebzehnten September, um vierzehn Uhr dreißig.`
3. `Einen Augenblick bitte, ich prüfe das für Sie.`
4. `Die Außentemperatur beträgt minus drei Komma fünf Grad.`
5. `Ich habe Sie nicht ganz verstanden. Möchten Sie den letzten Satz wiederholen?`

Candidate selection is autonomous, blinded by randomized profile labels, and
uses both listening and objective gates:

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

All surviving profiles receive a score from zero through 100:

- 40 points for blinded listening: naturalness 15, trust/telephone fit 10,
  clarity 10, and appropriate prosody 5;
- 25 points for speaker consistency across the five outputs;
- 20 points for German intelligibility and transcript accuracy;
- 10 points for warm first-audio latency;
- 5 points for absence of noise, glitches, reference-tail leakage, duplicated
  words, truncation, and code-switching.

The comparison rejects a profile rather than scoring it if any hard gate fails.
In particular, generated output must not begin by repeating the end of the
reference transcript.

## Private Reference Storage

The winning reference and manifest are stored only on the GX10 host:

`/home/volsch/voice-private/profiles/`

The directory is mode 0700. The `voice` stack mounts it read-only at
`/run/voice-profiles`; the TTS container and image never contain the licensed or
synthetic evaluation audio.

The private `profiles.json` records:

- profile ID;
- relative reference-audio path;
- exact reference transcript and language;
- source type `synthetic-qwen-voice-design` or
  `licensed-human-reference-private`;
- for a synthetic winner, source model and immutable VoiceDesign revision;
- for a licensed winner, a private provenance label and source/derived hashes,
  but no public URL or speaker identity;
- SHA-256 of the WAV;
- generation and selection date;
- the approved design instruction for a synthetic winner;
- private-use-only scope;
- evaluation scores and the selected anonymous candidate ID.

Git contains the loader, schema, tests, and a non-audio example manifest only.
It never contains the source sample, derived excerpts, winner WAV, live
manifest, or a retrievable source location.

## Runtime Architecture

The independently owned GX10 `voice` Portainer stack remains the owner of
Qwen3-ASR and Qwen3-TTS. Its stack identity, internal networks, GPU device
reservation, health policy, and lack of host-published ports remain unchanged.

The TTS image contains the profile loader but no reference audio. At startup
the server:

1. requires CUDA and verifies NVIDIA GB10;
2. loads the revision-pinned Base model from the local Hugging Face cache;
3. reads the read-only private profile mount and validates its permissions,
   manifest schema, profile ID, private-use scope, reference transcript, WAV
   format, and SHA-256;
4. creates or warms faster-qwen3-tts's clone-prompt cache from the reference;
5. reports healthy only after `shared-female-de-v1` is usable.

Only the Base model remains loaded after rollout. VoiceDesign is used to create
temporary synthetic candidates before the model switch and is not retained as
a production dependency.

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
- Missing private-use scope or a writable in-container profile mount: startup
  fails.
- CPU-only execution or a non-GB10 CUDA device: startup fails.
- Unknown profile ID: request fails with a bounded client error.
- Clone-prompt construction failure: health remains unhealthy.
- Reference-tail leakage, code-switching, repeated words, or truncated
  phonemes during acceptance: rollout fails and rolls back.
- Empty, odd-byte, malformed, or failed TTS output: the current callers retain
  their existing fail-closed/fallback behavior.
- Cancellation propagates through the streaming generator exactly as it does
  today; no detached synthesis continues after barge-in or priority
  cancellation.

No evaluation output, transcript, generated call audio, or caller audio is
persisted by the runtime. Only the private winning reference and manifest are
persistent.

## Repository Boundaries

The voice server, profile schema, model pin, deployment contract, and VoIP
profile setting belong to the existing isolated `voip-agent` branch
`feat/shared-ai-traefik`. The private source and winning reference belong to the
GX10 deployment, not either Git repository.

Companion's profile setting and trusted adapter change belong to an isolated
`ai-companion` feature branch. The repositories retain their independent
ownership; deployment coordinates their compatible commits without copying
secrets or merging stack ownership.

## Tests

### Unit and contract tests

- Profile-manifest parsing accepts a mounted private profile and rejects
  missing, duplicate, malformed, path-escaping, hash-mismatched, wrong-scope,
  and writable-mount entries.
- Both server endpoints resolve the same profile and invoke clone generation,
  not VoiceDesign generation.
- Non-streaming output stays a valid 24-kHz PCM WAV.
- Streaming output stays even-length raw signed-16-bit PCM.
- Unknown profiles fail closed.
- Health requires CUDA, the Base model, and a prepared default profile.
- VoIP sends `shared-female-de-v1` for greeting and streaming turns.
- Companion sends its configured profile and retains its fixed German language.
- Deployment tests pin the Base snapshot, require the read-only external
  profile mount, keep NVIDIA device reservations, and publish no backend ports.

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
4. The blinded evaluation report proves that the chosen candidate passed every
   hard gate and beat the other source, or won the under-three-point tie by the
   declared synthetic tie-break. At least five varied German utterances
   synthesized in separate requests have one stable female speaker by listening
   and speaker-embedding similarity.
   The verification-only model is `microsoft/wavlm-base-plus-sv` at revision
   `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`. Reference-to-output cosine
   similarity must have a median of at least 0.80 and a minimum of at least
   0.70; pairwise output similarity must have a median of at least 0.85.
5. ASR round trips remain intelligible and do not introduce code-switching,
   duplicated words, or any suffix from the reference transcript.
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
VoiceDesign snapshot. Back up the private winning bundle locally without
redistributing it. If Base-model loading, profile-mount validation, latency,
voice quality, priority, or call acceptance fails, restore the previous image
and Compose without changing Traefik, ASR, or the clients' network topology.
Rollback is reported as a failed clone-profile rollout, not as successful
completion.
