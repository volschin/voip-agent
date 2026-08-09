# Spark-vLLM ASR v0.23.0 Midpoint Reactivation Design

**Date:** 2026-08-09

## Goal

Finish the previously interrupted vLLM `0.23.0` midpoint candidate and test it
against the frozen ten-case non-speech gate with the exact production model.
The work reuses the already-built midpoint base instead of repeating its
approximately 2.5-hour source build.

Production remains immutable. This design authorizes an isolated candidate
build and non-speech qualification only, not a full A/B or rollout.

## Recovered Base

The GX10 still contains:

```text
tag:      dgx-spark-vllm:midpoint-v023
image:    sha256:bfa7bcd7c70829e44cd919f22fc68a028816681abe8d4f3b4a2b1ba81e47c134
size:     approximately 19 GB
platform: ARM64/Linux
```

This base was produced by the previously reviewed midpoint build recipe after
the completed FlashInfer and vLLM native builds. Before reuse, validate its
exact runtime inventory and build metadata against the historical contract:

- eugr snapshot `b51af15a280d28c2ad9096b3ef581524eddbd0e7`;
- CUDA `13.0.2` base and runtime CUDA `13.0`;
- NCCL commit `6da422082f910a8dd230f7e42e26ece4dc37bccc`;
- Torch `2.11.0+cu130`;
- vLLM commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` with runtime version
  overridden to exactly `0.23.0`;
- FlashInfer `0.6.12`, commit
  `d768c14e7cf5dd5df45a8a1de78ae815879f108a`;
- Transformers `5.12.1`;
- Qwen3-ASR adapter SHA-256
  `e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d`.

Any mismatch rejects reuse and stops this design. Do not silently rebuild or
substitute a nearby image.

## Repository Recipe and Final Image

Restore the final reviewed midpoint recipe state that existed immediately
before commit `cf6f716` removed it as obsolete. Restore only:

- `dgx/asr/build-midpoint-base.sh`;
- `dgx/asr/eugr-midpoint.patch`;
- the midpoint form of `dgx/asr/Dockerfile`;
- its deployment-contract tests; and
- the focused DGX README instructions.

Preserve subsequent unrelated repository changes and the current UrocyonF
1.7B Compose contract. The build script gains a fail-closed reuse path: when
the exact local base image ID and complete runtime inventory match, it skips
the expensive base build and builds only:

```text
dgx-qwen3-asr:spark-midpoint-v023-test
```

The derivative adds only the existing hash-locked SoundFile `0.13.1` and PyAV
`17.0.1` audio dependencies. It adds no vLLM, encoder, prompt, language,
response, VAD, or transcript-filter patch. The final build asserts the base
image ID, versions, adapter hash, audio packages, ARM64/Linux platform, and
absence of an accidental second Torch or vLLM installation.

## Isolated Runtime Contract

Start the final image under a distinct test name with restart policy `no`, a
loopback-only port, offline model operation, and the production Hugging Face
cache mounted read-only. Serve exactly:

```text
UrocyonF/Qwen3-ASR-1.7B-NVFP4
revision 61ad4d533c64e033a750b66c44aad6f18634997e
served name qwen3-asr
language de
gpu-memory-utilization 0.08
max-model-len 8192
max-num-seqs 4
```

Use the same immutable normalized gateway as the corrected same-model and
ModelRunnerV1 runs. Require exact model readiness, valid normalized API shape,
candidate-correlated nonzero CUDA activity, and zero restarts before quality
testing. A startup, API, model-identity, or CUDA failure rejects the candidate
without an additional change.

## Non-Speech Gate

Send the frozen ten non-speech cases once, in their established order, through
the normalized gateway with `language=de`. Preserve the exact corpus, audio,
runner, scorer, request-contract, and case-order hashes. Do not retry a case,
inspect transcripts before scoring, tune decoding, remove an outlier, or add a
filter.

The candidate passes only when:

- all `10/10` requests succeed;
- aggregate hallucinated words are at most the production baseline of `5`;
- malformed responses, non-transcript additions, language changes,
  command-risk cases, and normalized nondeterministic cases are all absent; and
- candidate-correlated CUDA is true.

If the result exceeds `5` or any safety gate fails, stop and record the
candidate as rejected. If it passes, retain it as the next candidate and report
the result; do not automatically run the complete 70/12 A/B.

## Evidence and Cleanup

Raw outputs and protected paths remain owner-only outside Git. Repository-safe
evidence contains only immutable identities, hashes, versions, aggregate and
per-case word counts, structural response facts, CUDA booleans/aggregate
maximums, and gate decisions.

Remove only the named candidate and gateway containers after the run. Retain
the midpoint base and final candidate images. Production must finish on its
current image and UrocyonF 1.7B command as `running|healthy|0|unless-stopped`
with zero restarts.
