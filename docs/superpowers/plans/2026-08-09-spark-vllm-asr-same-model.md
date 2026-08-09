# Spark-vLLM ASR Same-Model Correction Implementation Plan

> **COMPLETED HISTORICAL RECORD — DO NOT EXECUTE.** This plan finished on
> 2026-08-09. Production and candidate each completed exact-model `70/70`
> quality plus `12/12` load series with container-correlated CUDA. The
> candidate failed the mandatory non-speech safety gate because hallucinated
> words increased from production `5` to candidate `14`; it is not eligible for
> rollout, and production was not changed. Every unchecked box and live command
> below records the historical procedure rather than pending work. Do not rerun
> or continue any step from this document. A future experiment requires a new
> plan, fresh authorization, and new exclusive evidence paths.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Requalify the unchanged first Spark-vLLM image against production with the exact same Urocyon Qwen3-ASR-1.7B-NVFP4 model snapshot.

**Architecture:** Remove the superseded midpoint recipe, correct the repository model contract under TDD, then start the retained first candidate image in isolation with the exact production model. Use the existing protected discriminator and normalized full benchmark to make the image-only decision.

**Tech Stack:** Docker/Compose, ARM64 NVIDIA GB10, Spark-vLLM, Qwen3-ASR-1.7B-NVFP4, pytest, Ruff, protected ASR benchmark gateway.

## Global Constraints

- Production image: `sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3`.
- Candidate image: `sha256:ccbee8c22f1619e35ff5f56244d371c663d22628ee919e016a3fec9b535b0fb0`.
- Model snapshot: `/root/.cache/huggingface/hub/models--UrocyonF--Qwen3-ASR-1.7B-NVFP4/snapshots/61ad4d533c64e033a750b66c44aad6f18634997e`.
- Production must remain `running|healthy|0|unless-stopped`; never roll it back to the repository's stale 0.6B command.
- Keep served name `qwen3-asr`, `language=de`, port 8001, GPU memory utilization `0.08`, max model length `8192`, max sequences `4`, offline mode, and all existing Spark environment flags.
- Candidate container: `qwen3-asr-spark-1p7b-test`; loopback port `127.0.0.1:18001`; restart policy `no`.
- No vLLM, encoder, prompt, language, or response patch.
- Previous 0.6B-versus-production quality evidence is invalid for image eligibility.
- Never expose protected audio or transcripts in Git or reports.
- No production rollout, push, PR, or merge in this plan.

---

### Task 1: Restore the first candidate and correct the model contract

**Files:**
- Remove through a non-destructive revert: midpoint-only build script, patch, design, plan, README text, and tests.
- Modify: `dgx/docker-compose.yml`
- Modify: `tests/test_dgx_deployment.py`
- Modify: `dgx/README.md`
- Modify: `docs/superpowers/specs/2026-08-09-spark-vllm-asr-image-design.md`
- Modify: `docs/superpowers/plans/2026-08-09-spark-vllm-asr-image.md`
- Modify: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/pr-body.md`

**Interfaces:**
- Consumes: commits `9c0320c..a640128` and the live 1.7B model evidence.
- Produces: the original digest-pinned Spark candidate recipe plus exact 1.7B Compose/test contract.

- [ ] Revert `9c0320c^..a640128` as one non-destructive revert commit, preserving all earlier first-candidate work.
- [ ] Add a deployment test requiring the exact Urocyon repository, `1.7B-NVFP4`, and snapshot `61ad4d...`; run it and record RED against the stale 0.6B Compose path.
- [ ] Change only the ASR model path in Compose and corresponding constants/documentation. Restore the Dockerfile's exact first-candidate base digest and runtime assertions through the revert; do not rebuild it.
- [ ] Mark the earlier benchmark result invalid because model identity differed; do not retain the false image-ineligible conclusion.
- [ ] Run focused tests, full pytest, Ruff, format, Compose config, Bash syntax where applicable, and diff check.
- [ ] Commit the model-contract correction and write a complete TDD report.

---

### Task 2: Run isolated 1.7B candidate acceptance and discriminator

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-same-model/task-2-report.md`
- Create only in the protected GX10 directory: raw 21-case output.

**Interfaces:**
- Consumes: retained candidate image and cached exact 1.7B snapshot.
- Produces: model/API/CUDA proof plus transcript-free 21-case counts.

- [ ] Re-read production image, health, restart count, policy, command, model cache revision, candidate image ID, and empty candidate-container namespace.
- [ ] Start only `qwen3-asr-spark-1p7b-test` with the exact snapshot and unchanged production parameters on `127.0.0.1:18001`.
- [ ] Require `/v1/models`, exact container command, HTTP 200 with string `text`, and candidate-correlated nonzero CUDA SM activity.
- [ ] Run the fixed 21-case selector with `language=de`; require selection hash `cb4b76a15c7182376ecb426abfd29af26e9c849d2fb3ddf01e5f09f3490fd957` and `21/21` successes.
- [ ] Require at least entities `14/22`, numbers `1/6`, times `1/6`, dates `0/4`. On failure, stop before the full A/B without tuning or selective reruns.
- [ ] Record only safe aggregates and production pre/post invariants; remove only the named candidate container after the decision and retain the image.

---

### Task 3: Run the complete model-identical A/B and finalize evidence

**Files:**
- Replace outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/performance-ab-report.md`
- Modify: `docs/superpowers/specs/2026-08-09-spark-vllm-asr-image-design.md`
- Modify: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/pr-body.md`

**Interfaces:**
- Consumes: successful Task 2 candidate and existing normalized common gateway.
- Produces: corrected image-only eligibility decision.

- [ ] Start the exact candidate again and build one immutable normalized gateway used for both sides.
- [ ] Run production quality `70/70`, candidate quality `70/70`, production load `12/12`, and candidate load `12/12` with identical bytes, `language=de`, order, concurrency, timeouts, model, and gateway.
- [ ] Require CUDA activity for both load phases and validate the safe comparator output.
- [ ] Apply all existing quality, non-speech, success, median `1.05x`, and p90 `1.10x` gates. Do not delete outliers or replicate a quality failure.
- [ ] Remove only exact benchmark/candidate containers, retain the candidate image, and re-prove production unchanged.
- [ ] Run full repository verification, commit the corrected outcome, and request task plus whole-branch review.
