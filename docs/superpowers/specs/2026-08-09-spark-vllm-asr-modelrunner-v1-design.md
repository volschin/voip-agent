# Spark-vLLM ASR ModelRunnerV1 Diagnostic Design

**Date:** 2026-08-09

## Goal

Determine whether vLLM ModelRunnerV2 causes the remaining non-speech regression
in the smaller Spark-vLLM ASR image. The experiment changes exactly one runtime
variable and does not rebuild or patch the image.

The current production image remains immutable. This diagnostic does not
authorize a rollout.

## Baseline and Candidate

Both sides serve the exact cached
`UrocyonF/Qwen3-ASR-1.7B-NVFP4` snapshot
`61ad4d533c64e033a750b66c44aad6f18634997e` with served name `qwen3-asr`,
`language=de`, identical model limits, identical request bytes, and the same
normalized benchmark gateway.

The diagnostic candidate uses the already-built image
`sha256:ccbee8c22f1619e35ff5f56244d371c663d22628ee919e016a3fec9b535b0fb0`
and adds only:

```text
VLLM_USE_V2_MODEL_RUNNER=0
```

The image, model, dependencies, command, sampling parameters, audio handling,
gateway, and benchmark data remain unchanged. In particular, do not combine
this experiment with VAD, transcript filtering, source patches, or decoding
tuning.

## Execution

Before starting the candidate, re-read the current production image, command,
health, restart count, and restart policy. Start one separately named candidate
on a loopback-only port with restart policy `no`, offline model access, the
production cache mounted read-only, and the exact production serving command.

Prove that the candidate received `VLLM_USE_V2_MODEL_RUNNER=0` and that the
running vLLM configuration selected ModelRunnerV1. A startup failure or a
configuration that still selects ModelRunnerV2 fails this candidate without a
fallback or additional change.

## Staged Acceptance

The first stage sends only the frozen ten non-speech probes through the existing
normalized gateway. It is sufficient to reject the hypothesis quickly and must
not alter, remove, or selectively repeat a probe.

The candidate passes the discriminator only when:

- all ten requests succeed;
- the aggregate hallucinated-word count is at most the production baseline of
  `5`;
- there are no malformed responses, command-risk additions, language changes,
  or nondeterministic normalized results; and
- candidate-correlated CUDA activity is observed during a real request.

If the discriminator fails, stop. Record the result, remove only the candidate
and gateway containers, retain the image, and leave production unchanged. Do
not run the full benchmark.

If the discriminator passes, run the complete existing same-model gate in its
fixed order: production quality `70`, candidate quality `70`, production load
`12`, candidate load `12`. Eligibility still requires all existing quality,
non-speech, latency, request-success, provenance, and CUDA gates. Passing the
ten-case discriminator alone never authorizes rollout.

## Safety and Evidence

Protected audio and raw transcripts remain owner-only outside Git. Repository-
safe reports contain only hashes, aggregate metrics, fixed case identifiers,
version and image provenance, and transcript-free validation results.

The candidate and benchmark gateway are removed after the run. The retained
candidate image is not tagged for production. Production must finish on the
same current image and command, `running|healthy`, with zero restarts and its
existing restart policy.
