# Slim CUDA 13.3.1 TTS Image Design

## Goal

Replace the oversized vLLM-derived TTS image with a purpose-built ARM64 image
based on CUDA 13.3.1 and Ubuntu 26.04 while preserving the current production
TTS API, model, private WAV profile, cooperative cancellation, and fail-closed
health behavior.

The deliverable is an isolated candidate image. Building and validating the
candidate must not recreate, stop, or replace the production `qwen3-tts`
container.

## Current State

The current production image is 33.7 GB uncompressed and inherits from a
vLLM development image. The TTS application does not import vLLM, Ray, or
FlashInfer, but the image retains their packages, CUDA development tools, and
large build layers.

Production currently uses:

- Python 3.12;
- PyTorch `2.12.0.dev20260408+cu130`;
- CUDA-capable NVIDIA GB10 execution;
- `faster-qwen3-tts==0.2.6` with the repository-owned cooperative-cancellation
  patch;
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` revision
  `fd4b254389122332181a7c3db7f27e918eec64e3`;
- private read-only `shared-female-de-v1` WAV and schema-2 manifest;
- 24 kHz mono PCM/WAV API contracts.

## Architecture

Use a two-stage Docker build.

### Builder

The builder uses the ARM64 manifest of:

```text
nvidia/cuda:13.3.1-devel-ubuntu26.04@sha256:da3989b0ea8e8b4b241711edd5823bc1cc83d05a01882258bddad84d7394c37e
```

It installs Ubuntu's Python 3.14 toolchain, creates `/opt/tts-venv`, installs a
fully pinned and hash-locked ARM64 dependency set, and builds
`flash-attn==2.8.3` with `MAX_JOBS=4` and `FLASH_ATTN_CUDA_ARCHS=120`.

PyTorch uses the official ARM64 CPython 3.14 CUDA 13.2 wheel:

```text
torch==2.13.0+cu132
```

There is no matching official `torchaudio==2.13.0+cu132` ARM64 CPython 3.14
wheel. The exercised Qwen 12 Hz runtime does not import TorchAudio, so install
`qwen-tts` without declared dependencies and supply its exercised dependencies
explicitly. A build-time import check and the real-model validation must fail
if that assumption is wrong. Do not install an incompatible TorchAudio version.

CUDA 13.3.1 is the container build toolchain. The PyTorch binary contract is
CUDA 13.2, so `torch.version.cuda` must report `13.2`; this is intentional and
must be reported explicitly in validation evidence. Flash-Attention is compiled
by the CUDA 13.3.1 builder against that PyTorch installation.

The existing version-bound cooperative-cancellation patch is applied and its
verification script must pass in the builder.

### Runtime

The runtime uses the ARM64 manifest of:

```text
nvidia/cuda:13.3.1-base-ubuntu26.04@sha256:f65b4f0b65bbf2e0a2520cebaec3120bf4ed110aecc3e7dcab3b11cb508a0484
```

It contains only:

- Python 3.14 and the system libraries required by the exercised 12 Hz Qwen
  TTS path;
- `/opt/tts-venv` copied from the builder;
- the five repository-owned TTS server modules;
- NVIDIA container runtime metadata inherited from the CUDA base.

It must not contain the CUDA compiler, vLLM, Ray, or FlashInfer. Model files and
the private profile remain read-only runtime mounts and are never copied into
the image.

## Dependency Locks

The existing lock relies on packages inherited from the vLLM base and cannot be
reused unchanged. Replace it with a complete ARM64/Python-3.14 lock that covers
all installed Python distributions, including PyTorch's CUDA dependencies.

The first candidate keeps dependencies required by the exercised Qwen runtime.
Exclude TorchAudio and optional demo dependencies such as Gradio only when a
build-time import test proves the production TTS modules do not require them.
Do not add a general compatibility layer or unrelated dependency upgrades.

Create an Ubuntu-26.04 runtime package lock from packages actually installed in
the final stage. Package installation must use exact versions. Repository and
Docker build checks must fail on unpinned base images or dependencies.

## Preserved Contracts

This phase changes packaging only. Preserve without modification:

- `/health`, `/v1/audio/speech`, and `/v1/audio/speech/stream` contracts;
- default profile ID and unknown-profile HTTP 422 behavior;
- model path and pinned model revision;
- whole-WAV production synthesis and diagnostic raw PCM streaming;
- cancellation, admission, locking, and priority behavior;
- offline Hugging Face/Transformers operation;
- private read-only profile validation and permissions;
- existing Compose service configuration, except for selecting the candidate
  image during isolated validation.

The precomputed vLLM-Omni ICL profile is explicitly outside this phase.

## Failure Behavior

The image build fails if dependency resolution, Flash-Attention compilation,
the cancellation patch, or its verification fails. Runtime startup remains
fail-closed when CUDA, NVIDIA GB10, the model revision, or the private profile
is unavailable.

No CPU fallback is acceptable. A candidate that imports successfully but does
not execute a real CUDA TTS workload is not accepted.

## Validation

Static validation must prove:

- both Docker stages use the pinned CUDA 13.3.1 Ubuntu 26.04 ARM64 manifests;
- all application and dependency locks are exact;
- the runtime stage does not install build tools;
- existing unit tests, Ruff, formatting, and Compose rendering pass;
- the candidate image is smaller than the current 33.7 GB image.

Isolated GX10 validation must prove:

- `/etc/os-release` reports Ubuntu 26.04;
- Python reports 3.14;
- PyTorch reports 2.13.0 with `torch.version.cuda == "13.2"`;
- `torch.cuda.is_available()` is true and the device is NVIDIA GB10;
- a real CUDA tensor operation succeeds;
- Flash-Attention imports and its CUDA extension loads;
- vLLM, Ray, and FlashInfer are absent;
- the pinned offline model loads and health becomes green;
- one real profile-backed speech request returns valid 24 kHz mono WAV;
- GPU activity occurs during that request;
- cancellation and immediate recovery retain their existing behavior;
- the candidate is removed after validation and production remains healthy with
  its original image ID and restart policy.

## Rollout Boundary

This work produces and validates a local candidate image only. It does not
authorize a production image switch, repository delivery, or ICL-profile
migration. Those require separate approval after the evidence is reported.
