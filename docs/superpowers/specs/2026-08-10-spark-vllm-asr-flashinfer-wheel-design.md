# Spark-vLLM ASR FlashInfer Wheel Candidate Design

**Date:** 2026-08-10

## Goal

Build and qualify one ASR candidate that differs from the current qualified
vLLM 0.23 production image only by replacing its locally built FlashInfer
0.6.12 distributions with the precompiled FlashInfer 0.6.18 wheels published
by `eugr/spark-vllm-docker`.

Run the frozen ten-case non-speech gate first. Run the complete same-model
quality and load comparison only if that discriminator passes. This experiment
does not authorize a production rollout.

## Exact Baseline and Single Delta

The immutable baseline is:

```text
tag:          dgx-qwen3-asr:vllm023-615e858c
image:        sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930
platform:     ARM64/Linux
vLLM:         0.23.0
Torch:        2.11.0+cu130
CUDA runtime: 13.0
Transformers: 5.12.1
FlashInfer:   0.6.12, locally built from d768c14e7cf5dd5df45a8a1de78ae815879f108a
SoundFile:    0.13.1
PyAV:         17.0.1
```

The candidate installs exactly these three official Spark-vLLM release assets
with `--no-deps --force-reinstall`:

```text
release id: 367461871
release: Prebuilt FlashInfer Wheels (0.6.18-4fbac49f-d20260809) - DGX Spark Only

flashinfer_python-0.6.18-py3-none-any.whl
asset id: 507452716
size: 17122160
sha256: a722af5bbabd9156a6f75cec948822f0e966f8a83fffd913ead1fac851da7754

flashinfer_jit_cache-0.6.18-cp39-abi3-manylinux_2_28_aarch64.whl
asset id: 507452715
size: 252992614
sha256: 6e1eadb95eeaff33eb9393f73d6ad45d1c2bce001370d4388f76950043d7f0f1

flashinfer_cubin-0.6.18-py3-none-any.whl
asset id: 507452717
size: 1239178852
sha256: fe03b57b9fa233a23efc3d29fa7a1fd48ebb9b7e8eda529187d505f2b1493315
```

The moving release URL is acceptable only with all three fixed SHA-256 values
and byte sizes. A moved, missing, renamed, or changed asset fails closed.

No resolver may change Torch, vLLM, Triton, Transformers, audio packages, or
any other distribution. The candidate inherits the baseline entrypoint,
command, environment, healthcheck, labels, and root filesystem, adds one wheel
replacement layer, and receives no source patch or runtime tuning.

## Repository Recipe

Add a candidate-only Dockerfile and build script. The production Dockerfile and
Compose service remain unchanged.

The build script must:

- reject a local baseline tag whose image ID is not exactly `0fadf01c...`;
- download only the three named release assets into an owner-only temporary
  directory;
- verify every size and SHA-256 before Docker sees the wheel;
- build with `--pull=false` from the exact local baseline tag;
- verify the candidate rootfs begins with every baseline layer;
- verify image configuration is inherited unchanged;
- verify one and only one installed distribution for every package;
- prove that the installed distribution inventory is byte-for-byte equal by
  name and version after excluding only the three FlashInfer distributions;
- require all three FlashInfer distributions at exactly `0.6.18` and retain
  the existing vLLM adapter hash.

The candidate tag is:

```text
dgx-qwen3-asr:vllm023-flashinfer0618-test
```

## Runtime Contract

Use the exact production model and command:

```text
model: UrocyonF/Qwen3-ASR-1.7B-NVFP4
revision: 61ad4d533c64e033a750b66c44aad6f18634997e
served name: qwen3-asr
language: de
gpu-memory-utilization: 0.08
max-model-len: 8192
max-num-seqs: 4
```

Start the candidate with restart policy `no`, loopback-only publication,
offline model access, the production model cache read-only, and the exact
production runtime environment. Use the already qualified immutable normalized
gateway. Require model readiness, exact image/model identity, valid normalized
response shape, zero restarts, candidate-correlated CUDA, and nonzero SM
activity. There is no CPU fallback or retry with changed settings.

## Staged Benchmark

First send the frozen ten non-speech probes once in their fixed order through
the normalized gateway. Preserve the existing audio, case, request-contract,
runner, scorer, and ordering hashes. Do not inspect transcripts before scoring,
retry one case, remove an outlier, tune decoding, or add filtering.

The discriminator passes only when:

```text
successful requests: 10/10
hallucinated words: <= 5
malformed responses: 0
non-transcript additions: 0
language changes: 0
command-risk cases: 0
normalized nondeterministic cases: 0
candidate CUDA: true
```

On failure, skip the full comparison and finish with cleanup and a rejection
report. On success, run the complete established same-model series in order:

```text
production quality: 70
candidate quality: 70
production load: 12
candidate load: 12
```

Every request must succeed. Quality and non-speech results must not regress;
candidate load p50 must be at most `1.05x` production and p90 at most `1.10x`.
Both sides require container-correlated CUDA evidence. The test compares the
current 0.6.12 production image with the otherwise identical 0.6.18 candidate;
historical runs are context, not substitutes for this full A/B.

## Evidence, Cleanup, and Outcome

Raw transcripts, audio, protected paths, caller data, and raw GPU rows remain
owner-only outside Git. Repository-safe evidence may contain immutable image
and wheel identities, hashes, versions, counts, aggregate and per-case word
counts, latency aggregates and ratios, structural response facts, CUDA
booleans/aggregate maxima, and gate decisions.

Remove only the named candidate and gateway containers after the run. Retain
the candidate image for inspection. Production must finish byte-for-byte on
the preflight image and command as `running|healthy|0|unless-stopped`.

Record the execution outcome in this design and the implementation plan. An
eligible result is a benchmark conclusion only and requires separate rollout
authorization.
