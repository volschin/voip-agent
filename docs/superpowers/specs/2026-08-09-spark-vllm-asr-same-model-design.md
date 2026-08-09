# Spark-vLLM ASR Same-Model Correction Design

**Date:** 2026-08-09

## Root Cause

The previous production/candidate comparison was not image-only. Live Docker
evidence shows that production image
`sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3`
serves `UrocyonF/Qwen3-ASR-1.7B-NVFP4`. The production container was created
with that command on 2026-08-02 and currently remains
`running|healthy|0|unless-stopped`.

The rejected Spark-vLLM candidate was tested with Qwen3-ASR-0.6B revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. Its quality result therefore
combined an image/runtime change with a model change. The entity and number/time
regressions cannot be attributed to Spark-vLLM.

## Corrected Candidate

Return to the unchanged first candidate image:

`sha256:ccbee8c22f1619e35ff5f56244d371c663d22628ee919e016a3fec9b535b0fb0`

Serve the exact cached production model snapshot:

`/root/.cache/huggingface/hub/models--UrocyonF--Qwen3-ASR-1.7B-NVFP4/snapshots/61ad4d533c64e033a750b66c44aad6f18634997e`

Keep the served model name, endpoint, language, environment, memory limits,
sequence limits, cache mount, shared memory, networks, and GPU settings
unchanged. Do not patch the candidate image.

## Repository Correction

Remove the superseded midpoint source-build implementation and documentation.
Restore `dgx/asr/Dockerfile` to the digest-pinned first Spark candidate. Change
the repository Compose model path and deployment regression tests from the
incorrect 0.6B snapshot to the exact Urocyon 1.7B snapshot. Mark the previous
0.6B-versus-production quality decision invalid rather than ineligible.

## Qualification

Production is not recreated or changed. Start the candidate under a distinct
name on `127.0.0.1:18001`, require the exact 1.7B snapshot in its command,
`/v1/models`, a real transcription, and observed CUDA activity.

Run the existing 21-case discriminator first. The candidate must meet or exceed
the historical production counts: entities `14/22`, numbers `1/6`, times `1/6`,
dates `0/4`. If it passes, run the complete normalized-gateway comparison with
70 quality and 12 load requests per endpoint. Both sides use identical audio,
`language=de`, ordering, concurrency, gateway image, and model weights.

Eligibility requires no entity/number/time/date or non-speech regression,
median latency at most `1.05x` production, p90 at most `1.10x`, all requests
successful, and real CUDA execution. Evidence remains transcript-free.

This work builds and qualifies a candidate only. Production rollout requires a
separate instruction.
