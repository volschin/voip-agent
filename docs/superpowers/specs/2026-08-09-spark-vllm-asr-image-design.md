# Spark-vLLM ASR Image Design

**Date:** 2026-08-09

## Goal

Replace the current Qwen3-ASR production image with a smaller, reproducible
DGX-Spark-native derivative of `eugr/spark-vllm`, without changing the served
model, model revision, OpenAI-compatible transcription contract, German ASR
quality, or production latency.

The candidate may replace production only after isolated build, compatibility,
quality, performance, CUDA, and recovery gates pass. Until that adoption gate,
the running `qwen3-asr` container is immutable.

## Current Baseline

Production currently runs:

- image ID
  `sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3`;
- tag `ghcr.io/aeon-7/qwen3-asr-server:latest`;
- size `33,294,535,748` bytes, ARM64/Linux;
- Ubuntu 24.04, Python 3.12.3;
- PyTorch `2.12.0.dev20260408+cu130`, CUDA 13.0;
- vLLM `0.20.1.dev0+g88d34c640.d20260428`;
- UrocyonF/Qwen3-ASR-1.7B-NVFP4 snapshot
  `61ad4d533c64e033a750b66c44aad6f18634997e`;
- `running|healthy`, zero restarts, and `unless-stopped` restart policy.

The candidate baseline already present on the GX10 is:

- `eugr/spark-vllm@sha256:1d861bef8a6c0851140cec2575ebd32342d55bc0fd28ad4c6ca178269e9d1cff`;
- local image ID
  `sha256:579e1d8145d86e42b36b4ec6a7a4330ee474c4339094f7cc54f052ff58f27d78`;
- size `23,291,856,805` bytes, ARM64/Linux;
- Ubuntu 24.04 metadata, CUDA 13.0.2, and build architecture `12.1a`.

The digest, not the mutable `latest` tag, is the source of truth.

## Architecture

Create a repository-owned derivative in `dgx/asr/Dockerfile`:

```dockerfile
FROM eugr/spark-vllm@sha256:1d861bef8a6c0851140cec2575ebd32342d55bc0fd28ad4c6ca178269e9d1cff
```

The inherited Torch, CUDA, Triton, FlashInfer/attention, and vLLM stack remains
unchanged. The derivative adds only packages proven missing from the actual
Qwen3-ASR audio path. New Python distributions use exact versions and SHA-256
hashes. New APT packages, if any, use exact versions. Inherited files are bound
by the base-image digest and are not redundantly re-resolved.

Do not install a vLLM meta-extra, upgrade Torch, rebuild Flash-Attention, apply
broad vLLM source patches, or add multi-node launch machinery. Add runtime packages
only when a failing acceptance gate proves that exact addition necessary. If
the digest-pinned base cannot natively register Qwen3-ASR and serve its
transcription endpoint, this approach fails closed instead of growing into a
custom vLLM fork.

The image entrypoint remains `vllm serve`. Compose continues to own the exact
command so the model path, served name, memory limit, sequence limits, and API
port remain visible and reviewable in one place. Repository Compose builds the
service from `dgx/asr/Dockerfile`; the live Portainer stack uses the accepted
immutable local release tag.

The first live candidate added a local patch that suppressed vLLM's `usage`
field. Live production and client-code inspection proved that patch unnecessary:
production already returns `text+usage`, and `agent/stt.py` reads only `text`.
Remove the patch script, Docker invocation, and patch-specific tests, then
rebuild and repeat the real model/API/CUDA gate before benchmarking.

## Preserved Runtime Contract

The following are invariant:

- model: UrocyonF/Qwen3-ASR-1.7B-NVFP4;
- snapshot: `61ad4d533c64e033a750b66c44aad6f18634997e`;
- served model name: `qwen3-asr`;
- endpoint: `POST /v1/audio/transcriptions`;
- request: multipart WAV plus fixed `language=de` from the production client;
- response: a JSON object containing a string `text` field; additive response
  fields are permitted because the production `SttClient` consumes
  `response.json()["text"]` and the running production image already emits a
  `usage` field;
- model and Transformers operate offline from the cached snapshot;
- health checks `/v1/models` and require the served model entry;
- GPU reservation, 4 GiB shared memory, restart policy, autoheal label, and
  `voice_default` plus `shared_ai_voice` networks remain unchanged;
- no direct host publication of port 8001.

## Dependency Discovery and Fail-Closed Build

Before adding dependencies, inspect the digest-pinned base in an isolated
container and record:

- Python, Torch, CUDA, vLLM, Triton, and attention backend versions;
- Qwen3-ASR model registration;
- availability and importability of the actual audio decoders;
- absence or presence of compiler/runtime requirements during model load.

The final Docker build must verify imports for vLLM, Torch, and every added
audio decoder. It must also verify that the Qwen3-ASR model module is
registered. A dependency may be added only to close an observed failure.

## Isolated Candidate

Build on the GX10 as:

- image `dgx-qwen3-asr:spark-vllm-test`;
- container `qwen3-asr-spark-test`;
- loopback-only host port `127.0.0.1:18001`;
- the production model cache mounted read-only;
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Production stays running during candidate work. Candidate cleanup removes only
the named test container; the test image remains available for inspection.

## Acceptance Gates

### Repository and image

- all repository tests, Ruff checks, formatting checks, Compose rendering, and
  `git diff --check` pass;
- the base is digest-pinned and extension dependencies are exact/hash-locked;
- the candidate is ARM64/Linux and smaller than `33,294,535,748` bytes;
- the runtime exposes the intended inherited versions and contains no accidental
  second Torch or vLLM installation.

### Real model and CUDA

- the pinned local snapshot loads with networking disabled;
- `/v1/models` reports `qwen3-asr`;
- a real 16 kHz German WAV sent through `/v1/audio/transcriptions` returns HTTP
  200 and a JSON object with a string `text` field;
- the candidate process is observed as a CUDA compute process during a real
  transcription and aggregate SM utilization exceeds zero;
- model-start or request failure is fatal; there is no CPU fallback.

### Quality and performance A/B

Use the existing common ASR gateway and the versioned `asr-companion-de-v1`
German/Companion/VoIP corpus. A benchmark-only, hash-recorded gateway overlay
must apply the actual consumer contract to both backends: accept a JSON object
when `text` is a string, ignore additive response fields, and return only the
normalized text to the benchmark runner. The same built gateway image is used
for production and candidate. Production and candidate receive identical
bytes, language, ordering, and concurrency. Repository-safe evidence contains
no raw transcripts or private audio.

The initial unchanged-gateway attempt is invalid comparison evidence: its
strict exact-key check rejected production's valid `text+usage` response and
therefore produced zero successful production samples. Preserve that failed
attempt in protected evidence, but do not mix it into latency or quality
aggregates. Rerun both sides completely after the normalization overlay is
validated fail-closed.

The candidate is eligible only when:

- the benchmark reports no quality or non-speech-safety regression against the
  same model/revision baseline;
- median end-to-end latency is at most `1.05x` production;
- p90 end-to-end latency is at most `1.10x` production;
- every measured request succeeds and uses CUDA;
- any startup, recovery, or queue gate in the existing benchmark passes.

Do not delete outliers. If an isolated shared-host anomaly affects both images,
one predeclared complete replication may be used; report both series and the
combined result.

### Execution outcome

The protected 2026-08-09 A/B and language-hint series used the Qwen3-ASR-0.6B
candidate snapshot while live production served
`UrocyonF/Qwen3-ASR-1.7B-NVFP4`. Although the series used identical corpus
bytes, `language=de`, request order, concurrency, and a normalized gateway,
it changed both image and model identity. Its quality, latency, and
audio-encoder patch results are therefore invalid for image eligibility and
must not support an image-ineligible conclusion or a compatibility-patch root
cause. No production mutation was made; production remained healthy with zero
restarts.

The corrected same-model qualification is complete. Production and the
unchanged first Spark-vLLM candidate both served the exact Urocyon
Qwen3-ASR-1.7B-NVFP4 snapshot
`61ad4d533c64e033a750b66c44aad6f18634997e` through one immutable normalized
gateway image. Each side completed `70/70` quality and `12/12` load requests in
the prescribed order, with exact model identity revalidated before every
series and container-correlated CUDA observed for both load phases.

The candidate passed request-success, corpus/order, WER/CER, entity recall,
number/time/date, median-latency, p90-latency, and CUDA gates. Candidate versus
production load ratios were `0.9965589999544625` at p50 and
`0.9742123045873843` at nearest-rank p90. It failed the mandatory non-speech
safety gate because hallucinated words increased from `5` to `14`. The
candidate is therefore **not eligible for rollout**. The single replication
rule did not apply because a quality gate failed; no outlier was removed and no
quality series was rerun. Production remained on its original image, healthy,
with zero restarts, and no rollout occurred.

## Rollout and Rollback

Before rollout, tag the exact current production image ID as
`dgx-qwen3-asr:rollback-38f255cd`. Tag the accepted candidate with an immutable
release tag containing the Git commit.

Update only the `qwen3-asr` image line in Portainer stack `voice` (ID 16), with
pull and prune disabled. Preserve all environment, mounts, networks, command,
healthcheck, and other services byte-for-byte.

Post-rollout acceptance requires:

- the new image ID and immutable tag;
- `running|healthy`, zero restarts, and `unless-stopped`;
- `/v1/models` and a real German transcription from the consumer network;
- candidate-specific CUDA activity during that transcription;
- the old rollback tag still resolving to the original image.

Any failed health, API, CUDA, quality, latency, or restart gate restores the old
image line through Portainer and repeats health plus real-request acceptance.

## Security and Evidence

- Never commit audio, transcripts, caller data, model-cache contents, tokens, or
  benchmark corpus material.
- Repository-safe ASR evidence must be transcript-free and validate nested
  shapes, finite metrics, hashes, corpus version, and exact aggregate maps.
- Read the Portainer token only at runtime and never print or persist it.
- Production and rollback image IDs, candidate identity, commands, exit status,
  timings, aggregate quality metrics, and CUDA proof are recorded without raw
  process IDs or audio content.

## Known Trade-offs

This option deliberately prefers operational compatibility over the smallest
possible image. The eugr base includes broad DGX-Spark build and multi-node
capabilities, so the candidate is expected to remain much larger than the slim
TTS runtime. A later purpose-built vLLM runtime would be a separate project with
its own compatibility and benchmark gates.

The upstream `latest` image changes frequently. This design pins one observed
digest; adopting a newer digest requires a separate rebuild and the same full
acceptance sequence.
