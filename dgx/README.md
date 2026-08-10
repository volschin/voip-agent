# DGX Spark Services

Run on the DGX Spark (GPU host).

## Private TTS profile runbook

The private profile bundle stays outside Git and outside the image. On the GX10,
`/home/volsch/voice-private/profiles` is owned by `volsch:volsch` with mode
`0700`. Its `profiles.json` manifest and reference WAV are owned by the same
account with mode `0600`; neither may be a symlink. The Compose mount exposes
that directory read-only at `/run/voice-profiles`.

The manifest uses schema version 2 and private-use scope
`private-user-assistant-only`. Its production entry has the canonical ID
`shared-female-de-v1`, names one directly contained reference WAV, and records
the WAV and source hashes plus the bounded provenance fields defined in
`tts/profiles.example.json`. The manifest's private transcript, hashes,
provenance values, and the reference bytes must never be copied into Git,
images, logs, tickets, or rollback notes. The WAV must be non-empty,
uncompressed mono 24-kHz 16-bit PCM.

Create or repair only the metadata boundary, without printing file contents:

```bash
sudo install -d -o volsch -g volsch -m 0700 \
  /home/volsch/voice-private/profiles
sudo chown volsch:volsch \
  /home/volsch/voice-private/profiles/profiles.json \
  /home/volsch/voice-private/profiles/shared-female-de-v1.wav
sudo chmod 0600 \
  /home/volsch/voice-private/profiles/profiles.json \
  /home/volsch/voice-private/profiles/shared-female-de-v1.wav
```

TTS loads only the pinned offline Base snapshot:

```text
/root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3
```

That cache and the profile bundle are read-only in the container. Startup stays
unhealthy unless CUDA reports `NVIDIA GB10`, the exact model path is present
offline, the profile mount is read-only, directory/file modes and manifest
schema pass, the profile ID and hashes match, the WAV contract passes, and a
warm clone synthesis succeeds. Only then may `/health` report the pinned
revision and `shared-female-de-v1`.

Before a profile or image change, record the Portainer `voice` stack ID,
secret-free Compose, current image digest, health, restart count, and the
private bundle's filenames and hashes in a private operator ledger. Keep the
last-known-good bundle in a separate `0700` private directory with `0600`
files. On validation or acceptance failure, leave the failed bundle intact for
private diagnosis, restore the recorded image digest and last-known-good bundle
through the existing stack, and re-run health/CUDA/audio acceptance. Never
paste a manifest, transcript, reference audio, credential, or private hash into
the rollback report.

## Start services

> **ASR rollout hold:** The repository vLLM 0.23 midpoint image passed the exact
> 1.7B non-speech discriminator and the complete paired 70/12
> quality/load/CUDA gate against production. It is eligible for rollout, but no
> rollout has been authorized or performed. Do not use the Compose command
> below to replace or recreate the currently running production `qwen3-asr`
> without separate explicit rollout authorization. The command remains the
> canonical bootstrap procedure only after that hold is resolved, or for
> services explicitly selected without `qwen3-asr`.

After provisioning and validating the private TTS profile:

```bash
cd dgx
cp .env.example .env
docker network inspect shared_ai_voice >/dev/null 2>&1 \
  || docker network create --internal shared_ai_voice
docker compose up -d
```

This Compose file remains the canonical owner of Qwen3-ASR and Qwen3-TTS. It
keeps the stack-private default network and additionally attaches those two
services to the pre-created neutral `shared_ai_voice` network consumed by
Traefik. The stack does not publish ports 8001 or 8002 after cutover.

ASR loads the already cached `UrocyonF/Qwen3-ASR-1.7B-NVFP4` snapshot
`61ad4d533c64e033a750b66c44aad6f18634997e` with Hugging Face and Transformers
offline. TTS refuses to start unless PyTorch reports CUDA on `NVIDIA GB10`;
there is no CPU fallback.

## Build the midpoint ASR candidate

To build the reproducible local midpoint candidate, run:

```bash
cd dgx
./asr/build-midpoint-base.sh
```

To reuse the already verified local base and build only the ASR derivative, add
`--reuse-base`. This produces only the local test tags
`dgx-spark-vllm:midpoint-v023` and `dgx-qwen3-asr:spark-midpoint-v023-test`;
it does not deploy either image.

To build the isolated FlashInfer wheel comparison from the currently qualified
vLLM 0.23 image, run:

```bash
cd dgx
./asr/build-flashinfer-wheel-candidate.sh
```

The script requires the exact local `dgx-qwen3-asr:vllm023-615e858c` image,
downloads and hash-verifies the three precompiled Spark-vLLM FlashInfer 0.6.18
wheels, and produces only
`dgx-qwen3-asr:vllm023-flashinfer0618-test`. It verifies that every installed
distribution outside the FlashInfer package family remains identical. The
candidate is not deployed by this command and must pass its staged benchmark
before it can be considered eligible.

`qwen3-tts` builds locally from `./tts`. Production callers use stable
whole-WAV `/v1/audio/speech`; `/v1/audio/speech/stream` remains available only
for diagnostics. The image applies a version-bound patch to
`faster-qwen3-tts==0.2.6` so cancelled stable requests stop between codec steps
and promptly release the single model lock. The build fails if the pinned
source no longer matches that patch. First `up` compiles flash-attn — expect
~10–15 min on the Spark; later boots reuse the cached layer.

## Health checks

```bash
# ASR
docker compose exec -T qwen3-asr \
  python3 -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8001/v1/models')))"

# TTS
docker compose exec -T qwen3-tts \
  python3 -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8002/health')))"

# Embedding (text-embeddings-inference)
curl -s http://localhost:8003/health
```

External ASR/TTS requests use the exact authenticated Traefik paths on
the agent's `AI_ORIGIN` (default `https://mate.olcon.de`); direct host access
to 8001 and 8002 is intentionally
closed. The independently managed Gemma stack is reached through the same
Traefik origin.
