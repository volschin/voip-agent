# Stable Private Voice Clone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select the better of a synthetic German female voice and an
authorized private professional-speaker clone, then serve that winner as one
stable named profile to VoIP and Companion.

**Architecture:** The current VoiceDesign service first creates synthetic
references. A revision-pinned Qwen Base service then evaluates synthetic and
authorized human-reference candidates through the same clone path and target
texts. Only a private, read-only, hash-validated winner bundle is mounted into
the existing GX10 `voice` stack; public Git contains code and schemas but no
reference audio, source URL, live manifest, or retrievable speaker identity.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, faster-qwen3-tts, Qwen3-TTS
VoiceDesign/Base, NumPy, SoundFile, WavLM speaker verification, Qwen3-ASR,
pytest, Ruff, Docker Compose, Portainer, NVIDIA GB10 CUDA.

## Global Constraints

- The production profile ID is exactly `shared-female-de-v1`.
- VoiceDesign revision:
  `5ecdb67327fd37bb2e042aab12ff7391903235d3`.
- Base revision:
  `fd4b254389122332181a7c3db7f27e918eec64e3`.
- WavLM verification revision:
  `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`.
- The authorized sample is restricted to this user's private assistant.
- Never commit or publish the source URL, source bytes, derived human excerpts,
  winner WAV, live manifest, speaker identity, credentials, or generated
  evaluation/call audio.
- Private profile host directory:
  `/home/volsch/voice-private/profiles`, mode 0700.
- Container profile mount: `/run/voice-profiles`, read-only.
- Keep `voice` as the only owner of ASR/TTS; preserve its Portainer identity,
  internal networks, GPU reservation, and lack of ASR/TTS host ports.
- Production is offline, CUDA-only, NVIDIA-GB10-only, and Base-only.
- Preserve `POST /v1/audio/speech` as 24-kHz PCM WAV and
  `POST /v1/audio/speech/stream` as raw 24-kHz signed-16-bit little-endian PCM.
- Unknown profiles return HTTP 422 with exactly
  `unsupported voice profile`.
- Do not push branches or open PRs without explicit user authorization.

---

### Task 1: Private profile bundle contract

**Files:**
- Create: `dgx/tts/profiles.py`
- Create: `dgx/tts/profiles.example.json`
- Create: `tests/test_tts_profiles.py`

**Interfaces:**
- Produces frozen `VoiceProfile`.
- Produces
  `load_profiles(profile_dir: Path, *, mountinfo_path: Path = Path("/proc/self/mountinfo")) -> dict[str, VoiceProfile]`.
- Produces
  `resolve_profile(value: str | None, profiles: Mapping[str, VoiceProfile], default_id: str) -> VoiceProfile`.
- Produces bounded `ProfileError`.

- [ ] **Step 1: Write failing profile tests**

  Build test WAVs with `wave`, never binary fixtures. Assert acceptance of:

  ```json
  {
    "schema_version": 1,
    "usage_scope": "private-user-assistant-only",
    "profiles": [{
      "id": "shared-female-de-v1",
      "audio": "shared-female-de-v1.wav",
      "reference_text": "Guten Tag.",
      "language": "german",
      "source_type": "licensed-human-reference-private",
      "source_revision": "private-sha256:0123456789abcdef",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "selected_at": "2026-07-29T12:00:00Z"
    }]
  }
  ```

  Reject missing/extra fields, wrong schema/scope, duplicate IDs, blank text,
  non-lowercase or malformed hashes, absolute/path-escaping/symlink audio,
  absent files, hash mismatch, non-WAV input, stereo, non-24-kHz, non-16-bit,
  compressed, or empty WAV. Reject a mount whose matching `/proc/self/mountinfo`
  entry lacks `ro`.

- [ ] **Step 2: Prove the red state**

  Run:

  ```bash
  venv/bin/pytest tests/test_tts_profiles.py -v
  ```

  Expected: collection fails because `dgx.tts.profiles` is absent.

- [ ] **Step 3: Implement the loader**

  `VoiceProfile.audio_path` is a resolved `Path`; require it to remain directly
  below the resolved profile directory. Parse JSON with duplicate-key rejection
  through `object_pairs_hook`. Validate the nearest matching mount point from
  `/proc/self/mountinfo` and require `ro` in its mount options.

  Validate WAV with:

  ```python
  channels == 1
  sample_width == 2
  sample_rate == 24_000
  frame_count > 0
  compression_type == "NONE"
  ```

  Temporary compatibility aliases resolve to the default ID:

  ```text
  Eine warme, natürliche deutsche Stimme mit angemessenem Sprechtempo.
  Eine warme, natürliche, erwachsene weibliche deutsche Stimme mit klarer Aussprache, lebendiger Intonation und ruhigem Sprechtempo.
  ```

  Blank or unknown selectors raise
  `ProfileError("unsupported voice profile")`.

- [ ] **Step 4: Add the safe example**

  `profiles.example.json` contains the schema with an intentionally
  non-deployable all-zero demonstration hash but no source URL, real hash,
  speaker name, transcript from the licensed sample, or usable reference path.

- [ ] **Step 5: Verify and commit**

  Run:

  ```bash
  venv/bin/pytest tests/test_tts_profiles.py -v
  venv/bin/ruff check dgx/tts/profiles.py tests/test_tts_profiles.py
  venv/bin/ruff format --check dgx/tts/profiles.py tests/test_tts_profiles.py
  git diff --check
  ```

  Then:

  ```bash
  git add dgx/tts/profiles.py dgx/tts/profiles.example.json \
    tests/test_tts_profiles.py
  git commit -m "feat(tts): validate private voice profiles"
  ```

### Task 2: Testable clone service and fail-closed API

**Files:**
- Create: `dgx/tts/clone_runtime.py`
- Create: `dgx/tts/api.py`
- Modify: `dgx/tts/server.py`
- Modify: `dgx/tts/Dockerfile`
- Modify: `dgx/docker-compose.yml`
- Modify: `pyproject.toml`
- Create: `tests/test_tts_clone_runtime.py`
- Create: `tests/test_tts_api.py`
- Modify: `tests/test_dgx_deployment.py`

**Interfaces:**
- Consumes Task 1 profiles.
- Produces
  `CloneRuntime(model: Any, profiles: Mapping[str, VoiceProfile], default_profile_id: str)`.
- Produces `warm()`, `synthesize(text, voice, language)`, and
  `stream(text, voice, language)`.
- Produces frozen `HealthMetadata` with model revision, default profile, loaded
  profiles, and device name.
- Produces
  `create_app(runtime: CloneRuntime, health: HealthMetadata) -> FastAPI`.

- [ ] **Step 1: Write failing clone-runtime tests**

  A fake model records calls. Assert non-streaming calls:

  ```python
  model.generate_voice_clone(
      text="Hallo",
      language="german",
      ref_audio=profile.audio_path,
      ref_text=profile.reference_text,
      non_streaming_mode=True,
  )
  ```

  Streaming uses the same profile and `chunk_size=8`. `warm()` synthesizes
  `Bereit.` once per profile and rejects empty or non-24-kHz results.

- [ ] **Step 2: Write failing API tests**

  Add `fastapi`, `soundfile`, and `asgi-lifespan` to the dev extra only. Test
  through `httpx.ASGITransport`:

  - profile selection and both aliases;
  - bounded 422 for unknown/blank profile;
  - valid WAV media/type/rate/shape;
  - even-length raw PCM streaming;
  - float audio clipped/scaled to int16 without silence;
  - model errors mapped to bounded HTTP 500;
  - closing a client stream closes the underlying generator.

- [ ] **Step 3: Prove both red states**

  Run:

  ```bash
  venv/bin/pip install -e ".[dev]"
  venv/bin/pytest tests/test_tts_clone_runtime.py tests/test_tts_api.py -v
  ```

  Expected: collection fails for the absent modules.

- [ ] **Step 4: Implement `CloneRuntime`**

  Keep it free of FastAPI and Torch imports. Resolve the profile before calling
  the model and reuse its stable path/text so faster-qwen3-tts reuses
  `_voice_prompt_cache`. Do not expose `instruct`, reference fields, sampling,
  `xvec_only`, prompt objects, or model paths.

- [ ] **Step 5: Extract the HTTP API**

  Move request/response encoding into `dgx/tts/api.py`. `SpeechRequest.voice`
  is a profile selector. Preserve WAV/FLAC/MP3 handling and the raw streaming
  contract. In the streaming generator, close the model iterator in `finally`
  so cancellation cannot leave detached synthesis.

- [ ] **Step 6: Convert `server.py` to Base-only startup**

  Startup order:

  1. `require_gb10_cuda(torch)`;
  2. `load_profiles(Path("/run/voice-profiles"))`;
  3. load the exact offline Base snapshot;
  4. construct and warm `CloneRuntime`;
  5. create the API and report healthy.

  Health must expose only:

  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "model_revision": "fd4b254389122332181a7c3db7f27e918eec64e3",
    "default_profile": "shared-female-de-v1",
    "profiles_loaded": ["shared-female-de-v1"],
    "device": "NVIDIA GB10"
  }
  ```

- [ ] **Step 7: Pin deployment and private mount**

  In `dgx/docker-compose.yml`, require:

  ```yaml
  environment:
    - HF_HUB_OFFLINE=1
    - TRANSFORMERS_OFFLINE=1
    - QWEN_TTS_MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3
    - QWEN_TTS_DEFAULT_PROFILE=shared-female-de-v1
  volumes:
    - ${HOME}/.cache/huggingface:/root/.cache/huggingface:ro
    - /home/volsch/voice-private/profiles:/run/voice-profiles:ro
  ```

  Copy code only into the image. Deployment tests reject `VoiceDesign`,
  reference audio `COPY`, writable profile mounts, host ports, missing offline
  flags, mutable model IDs, or changed external networks.

- [ ] **Step 8: Focused verification and commit**

  Run:

  ```bash
  venv/bin/pytest tests/test_tts_profiles.py tests/test_tts_clone_runtime.py \
    tests/test_tts_api.py tests/test_dgx_deployment.py -v
  venv/bin/ruff check dgx/tts tests/test_tts_profiles.py \
    tests/test_tts_clone_runtime.py tests/test_tts_api.py \
    tests/test_dgx_deployment.py
  venv/bin/ruff format --check dgx/tts tests/test_tts_profiles.py \
    tests/test_tts_clone_runtime.py tests/test_tts_api.py \
    tests/test_dgx_deployment.py
  docker compose -f dgx/docker-compose.yml config --quiet
  ```

  Then:

  ```bash
  git add dgx/tts dgx/docker-compose.yml pyproject.toml \
    tests/test_tts_clone_runtime.py tests/test_tts_api.py \
    tests/test_dgx_deployment.py
  git commit -m "feat(tts): serve private cloned voice profile"
  ```

### Task 3: VoIP named-profile client

**Files:**
- Modify: `agent/config.py`
- Modify: `agent/tts.py`
- Modify: `agent/main.py`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/conftest.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_tts.py`

**Interfaces:**
- Produces `Settings.tts_voice_profile: str`.
- Changes `TtsClient.__init__` to accept keyword-only
  `voice_profile: str = "shared-female-de-v1"`.

- [ ] **Step 1: Write failing configuration/client tests**

  Assert the default, configured override, and blank rejection. Assert greeting
  and streaming requests both send:

  ```json
  {
    "input": "Test",
    "voice": "shared-female-de-v1",
    "language": "german"
  }
  ```

  Patch `agent.main.TtsClient` and assert the setting is passed explicitly.

- [ ] **Step 2: Prove the red state**

  Run:

  ```bash
  venv/bin/pytest tests/test_config.py tests/test_tts.py -v
  ```

  Expected: failures for the absent setting and old free-form instruction.

- [ ] **Step 3: Implement the profile selector**

  Add:

  ```python
  tts_voice_profile: str = "shared-female-de-v1"

  @field_validator("tts_voice_profile")
  @classmethod
  def _require_tts_voice_profile(cls, value: str) -> str:
      if not value.strip():
          raise ValueError("tts_voice_profile must not be blank")
      return value.strip()
  ```

  Store the constructor value and use it on both endpoints. Remove
  `VOICE_INSTRUCT`; retain fixed language `german`.

- [ ] **Step 4: Document, verify, and commit**

  Add `TTS_VOICE_PROFILE=shared-female-de-v1` to `.env.example`. Explain that
  `voice` is a server-owned ID, never a caller-controlled description.

  Run:

  ```bash
  venv/bin/pytest tests/test_config.py tests/test_tts.py -v
  venv/bin/ruff check agent tests/test_config.py tests/test_tts.py
  venv/bin/ruff format --check agent tests/test_config.py tests/test_tts.py
  git diff --check
  ```

  Then:

  ```bash
  git add agent/config.py agent/tts.py agent/main.py .env.example README.md \
    tests/conftest.py tests/test_config.py tests/test_tts.py
  git commit -m "feat(tts): select stable voice profile"
  ```

### Task 4: Private candidate preparation

**Files:**
- Private only: `/home/volsch/voice-private/evaluation/source.mp3`
- Private only: `/home/volsch/voice-private/evaluation/human-*.wav`
- Private only: `/home/volsch/voice-private/evaluation/synthetic-*.wav`
- Private only: `/home/volsch/voice-private/evaluation/candidates.json`

**Interfaces:**
- Produces validated temporary reference profiles for Task 5.
- Does not modify Git.

- [ ] **Step 1: Preserve live rollback evidence**

  Resolve the current Portainer `voice` stack by name and copy its current
  stack ID, Compose, TTS image ID, health, restart count, and VoiceDesign
  revision into a secret-free local acceptance ledger. Do not print or store
  environment secrets.

- [ ] **Step 2: Create the private workspace**

  On the GX10:

  ```bash
  install -d -m 0700 \
    /home/volsch/voice-private/evaluation \
    /home/volsch/voice-private/profiles
  ```

  Transfer the authorized sample over SSH stdin. Record its SHA-256 privately;
  do not persist its URL or speaker name.

- [ ] **Step 3: Generate five synthetic references**

  Before replacing VoiceDesign, submit five independent requests using:

  ```text
  Guten Tag, hier ist Ihre digitale Assistentin. Ich höre Ihnen aufmerksam zu
  und helfe Ihnen ruhig, klar und zuverlässig weiter.
  ```

  and:

  ```text
  Eine erwachsene Frau zwischen etwa dreißig und vierzig Jahren mit warmer,
  natürlicher und vertrauenswürdiger Stimme. Klares neutrales Hochdeutsch ohne
  erkennbaren Regionalakzent, ruhiges mittleres Sprechtempo, deutliche, aber
  nicht überbetonte Artikulation und sanfte lebendige Intonation.
  Professionell und freundlich, weder kindlich noch werblich, nicht hauchig
  und nicht flirtend.
  ```

  Require valid mono 24-kHz PCM WAV, clipping fraction `<= 0.001`, clean German
  ASR, and no noise, duplicated words, or truncation.

- [ ] **Step 4: Derive human reference excerpts**

  Decode the authorized MP3 losslessly to mono 24-kHz PCM16. Use silence
  boundaries to propose 5–12-second passages. Audition them and retain two or
  three calm single-speaker excerpts without music, overlap, abrupt cuts, or
  strong dramatic delivery. Obtain Qwen ASR drafts and correct every
  `ref_text` word against the audio.

- [ ] **Step 5: Choose the best synthetic reference**

  Randomize the five synthetic labels. Score raw reference naturalness,
  telephone fit, clarity, prosody, ASR, signal quality, and F0 heuristic. Keep
  only the best synthetic reference for Base-model comparison; delete rejected
  synthetic files after recording bounded metrics.

### Task 5: Base-model A/B evaluation and winner bundle

**Files:**
- Private only:
  `/home/volsch/voice-private/evaluation/output/candidate-a/utterance-01.wav`
- Private only: `/home/volsch/voice-private/evaluation/report.json`
- Private only: `/home/volsch/voice-private/profiles/shared-female-de-v1.wav`
- Private only: `/home/volsch/voice-private/profiles/profiles.json`

**Interfaces:**
- Consumes Task 4 references and Task 2 clone service.
- Produces the only production profile bundle.

- [ ] **Step 1: Download exact offline snapshots**

  Use `hugging-face:hf-cli`. Download Base revision
  `fd4b254389122332181a7c3db7f27e918eec64e3` and WavLM revision
  `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` into the GX10 cache without
  printing a token. Prove both resolve with `HF_HUB_OFFLINE=1` before changing
  the live service.

- [ ] **Step 2: Deploy the candidate Base service**

  Build the exact source commit on ARM64. Update the existing Portainer `voice`
  stack, mounting a private candidate manifest read-only. Stop VoiceDesign
  before Base loads; never load both concurrently. Keep ASR, Traefik, networks,
  priority, and stack identity unchanged.

- [ ] **Step 3: Generate the fixed evaluation set**

  For the best synthetic reference and every retained human excerpt, submit the
  five exact evaluation sentences from the design in independently seeded
  requests. Capture non-streaming WAV and streaming first-byte latency.
  Anonymous profile labels must hide source type during listening.

- [ ] **Step 4: Apply hard gates**

  Reject any profile with invalid audio, clipping over 0.1 percent, failed or
  semantically wrong German ASR, code-switching, reference-tail leakage,
  repeated words, truncated phonemes, noise, glitches, unstable sex/identity,
  or warm first-audio latency over 3.0 seconds.

- [ ] **Step 5: Score surviving profiles**

  Score 0–100 exactly as the design specifies: listening 40, stability 25,
  intelligibility 20, latency 10, defects 5. Use WavLM embeddings to require:

  ```text
  reference-to-output cosine median >= 0.80
  reference-to-output cosine minimum >= 0.70
  pairwise output cosine median >= 0.85
  ```

  Listen in randomized order. If scores differ by fewer than three points,
  select synthetic; otherwise select the higher score.

- [ ] **Step 6: Build the winner bundle**

  Copy only the winning reference as
  `profiles/shared-female-de-v1.wav`. Write the strict live manifest with
  source type, private provenance label, source/derived hashes, exact
  transcript, `private-user-assistant-only`, evaluation score, anonymous ID,
  and UTC selection time. Set directory mode 0700 and files 0600. Delete
  rejected references and all generated evaluation outputs after the report's
  bounded metrics are recorded.

- [ ] **Step 7: Recreate and verify production TTS**

  Recreate TTS against the winner-only bundle. Health must prove Base revision,
  `shared-female-de-v1`, NVIDIA GB10, warm prompt, and restart count zero.
  Authenticated Traefik must return valid WAV and raw streaming PCM; unknown
  profile must return bounded HTTP 422.

### Task 6: Companion named-profile client

**Files in `/home/volsch/projekte/ai-companion`:**
- Create in an isolated worktree:
  `docs/superpowers/plans/2026-07-29-shared-private-voice-profile.md`
- Modify: `src/companion_core/config.py`
- Modify: `src/companion_core/clients/voice.py`
- Modify: `src/companion_core/app.py`
- Modify: `.env.example`
- Modify: `deploy/companion-stack.yml`
- Modify: `tests/test_config.py`
- Modify: `tests/test_voice.py`
- Modify: `tests/test_deployment_security.py`

**Interfaces:**
- Consumes live `shared-female-de-v1`.
- Produces `Settings.voice_profile`, env
  `COMPANION_VOICE_PROFILE`, and a fixed trusted adapter selector.

- [ ] **Step 1: Create the isolated Companion worktree**

  Use `superpowers:using-git-worktrees` from current `ai-companion/main`. Leave
  its main checkout and existing data untouched.

- [ ] **Step 2: Write and commit the repository-local plan**

  Use `superpowers:writing-plans`. Its tests must prove blank profile rejection,
  settings propagation through `create_app`, fixed German language, fixed
  profile JSON, no browser-controlled profile/model/instruction/backend, and
  Compose env `COMPANION_VOICE_PROFILE=shared-female-de-v1`.

- [ ] **Step 3: Execute with TDD**

  Replace `VOICE_INSTRUCTION` and `voice_instruction` with a profile setting.
  `VoiceClient` sends exactly:

  ```json
  {
    "input": "Hallo",
    "voice": "shared-female-de-v1",
    "language": "german"
  }
  ```

- [ ] **Step 4: Verify and commit Companion changes**

  Run its repository-local complete pytest, Ruff, Compose, and `git diff
  --check` commands. Commit locally; do not push.

### Task 7: Full review and live acceptance

**Files:**
- Update the existing progress ledger in each repository if one exists.
- Do not persist private audio in either repository.

**Interfaces:**
- Consumes Tasks 1–6.
- Produces fresh acceptance evidence or a completed rollback report.

- [ ] **Step 1: Run full VoIP verification**

  ```bash
  venv/bin/pytest -v --tb=short
  venv/bin/ruff check agent/ tests/ dgx/tts/
  venv/bin/ruff format --check agent/ tests/ dgx/tts/
  docker compose config --quiet
  docker compose -f dgx/docker-compose.yml config --quiet
  git diff --check
  ```

  Run the full Companion commands declared in its plan.

- [ ] **Step 2: Request independent review**

  Use `superpowers:requesting-code-review`. Review spec coverage, duplicate-key
  handling, path/symlink escape, hash and private-scope validation, read-only
  mount proof, Base/offline pin, bounded errors, streaming cancellation,
  license-asset exclusion, secret handling, stack ownership, and rollback.
  Fix every confirmed finding and rerun affected tests.

- [ ] **Step 3: Verify priority and cancellation**

  Start Companion streaming TTS and cancel it; prove synthesis/playback stops.
  Acquire a VoIP lease while Companion voice is active; prove Companion is
  cancelled or rejected, VoIP synthesis succeeds, renewal works, release
  restores Companion, and no detached GPU work continues.

- [ ] **Step 4: Verify GPU and latency**

  Correlate NVIDIA GB10 activity with real non-streaming and streaming TTS.
  Prove no CPU fallback, warm health, first-audio latency no greater than 3.0
  seconds, valid five-utterance winner metrics, and restart count zero.

- [ ] **Step 5: Run one real inbound call**

  Require greeting, multiple sentence segments, later turns, barge-in, and
  clean hangup. Blinded listening and speaker embeddings must show the same
  adult female identity throughout. Correlate ASR, LLM, TTS, GPU, and lease
  timestamps without recording caller audio or exposing caller IDs.

- [ ] **Step 6: Accept or roll back**

  If model loading, private mount, quality, reference leakage, latency,
  cancellation, priority, or call gates fail, restore the saved image and
  Compose and report failure. Otherwise record exact local commits, image ID,
  Portainer stack ID, test counts, bounded evaluation metrics, and clean/dirty
  worktree status.
