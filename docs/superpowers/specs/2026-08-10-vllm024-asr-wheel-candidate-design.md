# vLLM 0.24 ASR Wheel Candidate Design

**Date:** 2026-08-10

## Goal

Build and qualify one Qwen3-ASR candidate that differs from the currently
running vLLM 0.23 production image only where the official vLLM 0.24 package
contract requires a change. Reuse every production image layer, avoid source
builds, and install only official precompiled wheels.

Run the frozen ten-case non-speech discriminator first. Run the complete
same-model quality and load comparison only after a discriminator PASS. This
experiment does not authorize a production rollout.

## Immutable Baseline

The live baseline verified before design is:

```text
tag:                    dgx-qwen3-asr:vllm023-615e858c
image:                  sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930
platform:               ARM64/Linux
rootfs layers:          21
state:                  running|healthy|0|unless-stopped
vLLM:                   0.23.0
vLLM source commit:     0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
humming-kernels:        0.1.4
Torch:                  2.11.0+cu130
Torch CUDA:             13.0
Triton:                 3.6.0
Transformers:           5.12.1
FlashInfer distributions: 0.6.12
SoundFile:              0.13.1
PyAV:                   17.0.1
Qwen3-ASR adapter SHA:  e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d
```

The model and runtime contract remain exact:

```text
model: UrocyonF/Qwen3-ASR-1.7B-NVFP4
revision: 61ad4d533c64e033a750b66c44aad6f18634997e
served name: qwen3-asr
language: de
gpu-memory-utilization: 0.08
max-model-len: 8192
max-num-seqs: 4
```

## Exact Official Delta

The candidate replaces exactly two installed distributions:

```text
vllm:            0.23.0 -> 0.24.0
humming-kernels: 0.1.4  -> 0.1.6
```

Use only these two non-yanked PyPI wheel artifacts:

```text
vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl
size:   271361241
sha256: 700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5
url:    https://files.pythonhosted.org/packages/9e/80/51a071305b4eed0f6f512dc1c1c6957cbb14ccce38db1be90ffcff2a2844/vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl

humming_kernels-0.1.6-py3-none-any.whl
size:   178759
sha256: e64c0883fca930074bf920f4ba47cbf3acd244d7352f6c74c8d2182439770d8f
url:    https://files.pythonhosted.org/packages/25/85/490681b9ba24531da91d0bae801d2b26850e5a80bbd02c2efc500756e36b/humming_kernels-0.1.6-py3-none-any.whl
```

vLLM 0.24 changes other dependency constraints, but the live production
versions already satisfy all mandatory constraints. Optional extras such as
`soxr`, `helion`, and `vllm-gguf-plugin` are not activated. `gguf` remains
installed even though vLLM 0.24 no longer requires it directly; removing it
would create an unrelated delta.

The expected vLLM 0.24 Qwen3-ASR adapter SHA-256 is:

```text
639d3691fae9195ed38e17306a29b04bc60025e1119d0090443ec7d935eceffd
```

The adapter change is part of the official vLLM delta. It sanitizes
user-controlled transcription fields and changes prompt construction so the
normal transcription `language` field is reflected in the assistant prefix.
No repository patch may alter it.

## Candidate Image Construction

Add a candidate-only Dockerfile, build script, and verifier. Do not change the
production Dockerfile or Compose service.

The Dockerfile inherits the exact production image through a temporary local
tag that is created from and rechecked against the immutable image ID. A
read-only BuildKit bind mount exposes the two verified wheel files to one
`RUN` instruction. That instruction invokes:

```text
python3 -m pip install --no-cache-dir --no-deps --force-reinstall <two wheels>
```

No package index, dependency resolver, sdist, compiler, source checkout, or
network access is available during the Docker build. Wheel payloads do not
remain in a separate final-image layer. The final image has the exact 21-layer
production rootfs prefix plus one replacement layer and inherits image
configuration unchanged.

The candidate tag is:

```text
dgx-qwen3-asr:vllm024-pypi-test
```

Before promotion to that tag, fail closed unless all of these are true:

- the base tag resolves exactly to the immutable production image ID;
- both PyPI release entries have the exact package, version, filename, URL,
  size, SHA-256, Python requirement, and `yanked=false` metadata;
- downloaded files match exact sizes and SHA-256 values;
- the candidate is ARM64/Linux and has exactly one appended rootfs layer;
- image configuration equals production configuration;
- the normalized installed-distribution multiset is identical after excluding
  only `vllm` and `humming-kernels`, including duplicate multiplicities;
- base and candidate contain each changed distribution exactly once at the
  required old and new versions;
- Torch, CUDA, Triton, Transformers, all three FlashInfer distributions,
  SoundFile, PyAV, model adapter location, and all other packages remain fixed;
- the candidate adapter hash is exactly the official vLLM 0.24 hash;
- `pip check` succeeds without installing anything;
- no candidate tag is created until every verification passes.

## Runtime Isolation

Use exact container names:

```text
qwen3-asr-vllm024-test
asr-vllm024-gateway
asr-vllm024-production-gateway
```

The candidate uses restart policy `no`, the production model cache read-only,
offline model resolution, loopback-only publication, the exact production
command and environment, and the immutable normalized gateway. Require exact
image/model identity, readiness, normalized response shape, zero restarts,
candidate-correlated CUDA, and nonzero SM activity. There is no CPU fallback,
source-build fallback, altered prompt, changed model, or retry with different
settings.

## Staged Benchmark

First run the frozen ten non-speech cases exactly once, in their fixed order,
with unchanged request bytes, `language=de`, gateway, runner, scorer, timeout,
and hashes. Do not inspect transcripts before scoring, retry individual cases,
remove outliers, tune decoding, or add filtering.

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

On failure, skip the complete comparison and finish with cleanup and a
rejection report. On success, run exactly one complete series in this order:

```text
production quality: 70
candidate quality: 70
production load: 12
candidate load: 12
```

Every request must succeed. Quality, entity/number/time/date behavior, and
non-speech safety must not regress. Candidate load p50 must be at most `1.05x`
production and p90 at most `1.10x`. Both sides require container-correlated
CUDA. A replication is allowed only under the existing narrowly defined
shared-host latency rule; quality or safety failure may never be replicated.

## Evidence, Cleanup, and Outcome

Raw transcripts, audio, protected paths, caller data, raw GPU rows, wheels,
and image archives remain owner-only outside Git. Repository-safe reports may
contain artifact/image hashes, versions, counts, aggregate quality and latency
metrics, per-case non-speech word counts, structural response facts, CUDA
booleans/maxima, and gate decisions.

Remove only the three exact benchmark container names after the run. Retain
the candidate image for inspection. Re-read production immediately before any
live phase and again after cleanup. Production must finish on the same image,
command, model, state, health, restart count, and policy recorded at preflight.

The final report records build provenance, the complete static delta,
non-speech result, whether the full comparison ran, complete metrics when it
did, cleanup, production restoration, and candidate eligibility. Eligibility
does not authorize rollout, merge, push, or PR creation.
