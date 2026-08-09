# Spark-vLLM ASR Midpoint Candidate Design

**Date:** 2026-08-09

## Goal

Build a production-capable Qwen3-ASR candidate at a coherent historical point
between the current production image and the rejected August Spark-vLLM
nightly. The experiment changes the runtime stack as one dated unit instead of
patching individual quality symptoms.

Production remains immutable until the midpoint candidate passes the existing
German quality, latency, CUDA, API, and operational gates.

## Endpoints

The production endpoint is image
`sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3`,
built on 2026-05-02. Its relevant runtime is vLLM
`0.20.1.dev0+g88d34c640.d20260428`, Transformers `5.7.0`, FlashInfer `0.6.9`,
and Qwen3-ASR adapter SHA-256
`e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d`.

The rejected endpoint is image
`sha256:ccbee8c22f1619e35ff5f56244d371c663d22628ee919e016a3fec9b535b0fb0`,
built from the 2026-08-07 Spark-vLLM nightly. Its relevant runtime is vLLM
`0.26.1rc1.dev468+g6b5bec7be.d20260807`, Transformers `5.14.1`, FlashInfer
`0.6.18`, and Qwen3-ASR adapter SHA-256
`04991f24cfa4d2b7c74fd838c08d36bc8c0a94a56b3641cbd7f83c1350bed340`.

Both endpoints use the same Qwen3-ASR-0.6B model snapshot
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. The rejected endpoint passed the
latency and aggregate WER/CER gates but regressed entity recall and number/time
accuracy.

## Selected Midpoint

The midpoint is the Spark-vLLM repository snapshot
`b51af15a280d28c2ad9096b3ef581524eddbd0e7` from 2026-06-18, built from
source with these immutable core references:

- base: `nvidia/cuda:13.0.2-devel-ubuntu24.04`, resolved to an immutable digest
  before the build;
- NCCL: upstream tag `v2.30u1`, resolved to its exact commit before the build;
- PyTorch: `2.11.0+cu130`;
- torchvision: `0.26.0+cu130`;
- torchaudio: `2.11.0+cu130`;
- vLLM: tag `v0.23.0`, commit
  `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`;
- FlashInfer: `0.6.12`, resolved to its exact tag commit before the build;
- Transformers: `5.12.1`;
- Qwen3-ASR adapter SHA-256:
  `e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d`.

Package resolution is capped at the end of 2026-06-18 UTC and the resolved
runtime inventory is recorded. The build must not silently consume current
August packages. Exact direct versions and source commits override the date
cutoff when both are specified.

This point is selected because it is near the time midpoint and is the newest
vLLM release before the Qwen3-ASR adapter diverges from production. vLLM
`v0.24.0` already contains the newer prompt and language implementation, while
`v0.23.0` retains the production adapter byte-for-byte. The candidate therefore
tests the newer Spark runtime while holding the ASR model adapter at the proven
production implementation.

## Image Structure

The repository owns a reproducible multi-stage ASR image recipe. It reproduces
the selected eugr snapshot locally instead of deriving from the rejected August
nightly. Builder tooling may exist only in build stages; runtime contents follow
the historical eugr runner unless a real load failure proves one additional
runtime requirement.

The runtime adds the existing hash-locked SoundFile `0.13.1` and PyAV `17.0.1`
audio packages. It does not add Flash-Attention, an encoder patch, a prompt
patch, a transcription-response patch, or any other vLLM source modification.
The API may return additive fields such as `usage`; `agent/stt.py` consumes only
the string `text` field.

The image asserts the exact vLLM commit, Qwen3-ASR adapter hash, Torch/CUDA
versions, FlashInfer version, Transformers version, and audio package versions
during the build. Build metadata records all source commits, image digests, and
the dependency cutoff.

## Preserved Runtime Contract

- model snapshot:
  `/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0`;
- served model name: `qwen3-asr`;
- endpoint: `POST /v1/audio/transcriptions`;
- request language: `de`, even though the selected production-era adapter
  ignores that parameter;
- offline model and Transformers operation;
- identical GPU reservation, memory limits, sequence limits, health checks,
  cache mount, networks, restart policy, and API port;
- no production container or Portainer mutation during build and qualification.

## Qualification Sequence

1. Build the midpoint image on the GX10 under a new immutable candidate tag.
2. Verify the complete runtime inventory, exact model snapshot, exact adapter
   hash, loaded model identity, API shape, and real CUDA activity.
3. Run the predeclared discriminator set covering all protected entity,
   number, time, and date expectations. Results remain transcript-free.
4. Stop early if any mandatory quality category remains below production.
5. If the discriminator passes, run the existing complete normalized-gateway
   comparison: 70 quality requests and 12 load requests per endpoint.

The full candidate is eligible only when all prior acceptance thresholds hold:
no entity or number/time/date degradation, no non-speech-safety regression,
median latency at most `1.05x` production, p90 latency at most `1.10x`
production, every request successful, and real CUDA activity on both sides.

## Outcomes and Next Boundary

If the midpoint passes, it becomes the rollout candidate and the existing
rollback-safe production procedure applies. If it fails with the same quality
map as the August candidate, the remaining causal interval is between the
production image and vLLM `v0.23.0` or in the Spark base itself. If it matches
production quality, the causal interval is after vLLM `v0.23.0`; the next
candidate can bisect the `v0.23.0` to August interval without retaining the EOL
production image.

All failure and success evidence records image IDs, versions, hashes, aggregate
metrics, CUDA proof, and production invariants without private audio or raw
transcripts.
