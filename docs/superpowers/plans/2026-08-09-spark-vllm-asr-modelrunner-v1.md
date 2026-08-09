# Spark-vLLM ASR ModelRunnerV1 Diagnostic Implementation Plan

> **Execution status: COMPLETE — discriminator failed.** The exact
> ModelRunnerV1 candidate completed all ten frozen non-speech requests but
> produced `14` hallucinated words, above the production maximum of `5` and
> identical to the ModelRunnerV2 candidate. The conditional full A/B was
> skipped, test containers were removed, images retained, and production was
> unchanged. This candidate is not eligible for rollout.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether disabling ModelRunnerV2 makes the existing smaller Spark-vLLM image meet the production non-speech baseline, and run the full same-model qualification only if that discriminator passes.

**Architecture:** Reuse the immutable Spark candidate and normalized gateway images. Start one loopback-only candidate with the production model and command plus exactly `VLLM_USE_V2_MODEL_RUNNER=0`; first run the frozen ten-case non-speech discriminator, then conditionally reuse the established complete 70/70 quality and 12/12 load workflow.

**Tech Stack:** Docker, ARM64 NVIDIA GB10, vLLM `0.26.1rc1.dev468+g6b5bec7be.d20260807`, UrocyonF/Qwen3-ASR-1.7B-NVFP4, existing owner-only ASR benchmark runner, pytest, Ruff.

## Global Constraints

- Production `qwen3-asr` is read-only throughout this diagnostic; never stop, recreate, update, or restart it.
- Candidate image is exactly `sha256:ccbee8c22f1619e35ff5f56244d371c663d22628ee919e016a3fec9b535b0fb0`.
- Gateway image is exactly `sha256:f434a9c6533dd050968e56f547206ca4660ab4aa53d2f9d85ee6b5d8f12f473c`.
- Model snapshot is exactly `61ad4d533c64e033a750b66c44aad6f18634997e` from `UrocyonF/Qwen3-ASR-1.7B-NVFP4`.
- The only candidate-side experimental variable is `VLLM_USE_V2_MODEL_RUNNER=0`.
- Keep `language=de`, audio bytes, ordering, gateway, model command, model limits, sampling inputs, and all other environment variables unchanged.
- Never put protected paths, raw transcripts, audio, prompts, references, or per-case text into Git or a repository-safe report.
- If the ten-case discriminator exceeds `5` hallucinated words or any safety/request gate fails, stop without a full benchmark.
- Passing the discriminator authorizes only the established full benchmark, not production rollout.

---

### Task 1: Lock the live preflight and runner-selection contract

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-modelrunner-v1/task-1-report.md`
- Modify: none

**Interfaces:**
- Consumes: current feature branch, live container `qwen3-asr`, immutable candidate and gateway image IDs, owner-only benchmark artifact label `task2-same-model-1p7b-21case`.
- Produces: exact production invariant, zero conflicting test-container counts, and source-backed proof that environment value `0` selects ModelRunnerV1.

- [x] **Step 1: Verify the repository and live production baseline**

Run locally:

```bash
git status --short --branch
git rev-parse HEAD
```

Run on the GX10:

```bash
docker inspect --format '{{.Image}}|{{.State.Status}}|{{.State.Health.Status}}|{{.RestartCount}}|{{.HostConfig.RestartPolicy.Name}}|{{json .Config.Cmd}}' qwen3-asr
docker image inspect --format '{{.Id}}|{{.Size}}|{{.Architecture}}|{{.Os}}' dgx-qwen3-asr:spark-vllm-test
docker image inspect --format '{{.Id}}|{{.Architecture}}|{{.Os}}' sha256:f434a9c6533dd050968e56f547206ca4660ab4aa53d2f9d85ee6b5d8f12f473c
docker ps -aq --filter name='^/qwen3-asr-spark-mrv1-test$'
docker ps -aq --filter name='^/asr-modelrunner-v1-gateway$'
```

Expected: clean worktree; production is `running|healthy|0|unless-stopped` and serves the exact UrocyonF 1.7B model; candidate and gateway image IDs match the global constraints; both named-container queries are empty.

- [x] **Step 2: Prove the pinned image maps the environment variable to the V1 worker branch**

Run on the GX10:

```bash
docker run --rm --pull never \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  --entrypoint python3 \
  dgx-qwen3-asr:spark-vllm-test \
  -c 'import inspect; import vllm.envs as e; from vllm.config.vllm import VllmConfig; from vllm.v1.worker.gpu_worker import Worker; assert e.VLLM_USE_V2_MODEL_RUNNER is False; config_source = inspect.getsource(VllmConfig.use_v2_model_runner.fget); worker_source = inspect.getsource(Worker.init_device); assert "return use_v2_model_runner" in config_source; assert "GPUModelRunnerV1" in worker_source; assert "if self.use_v2_model_runner" in worker_source'
```

Expected: exit `0`. This binds the false environment value to the exact image source whose false branch instantiates `GPUModelRunnerV1`.

- [x] **Step 3: Validate only the owner-only benchmark artifact contract**

Locate the existing owner-only artifact by its exact label `task2-same-model-1p7b-21case`. Validate mode `0600`, the recorded corpus and ten-case subset hashes, runner hashes, and the fixed `language=de` request contract. Do not print or copy its absolute path or protected content.

Expected: exactly one artifact matches and every recorded hash equals the corrected same-model run. A missing, duplicate, or hash-mismatched artifact stops the task.

- [x] **Step 4: Write the transcript-free preflight report**

Record only image IDs, software versions, command hash, owner-only artifact label and hashes, production invariant, named-container counts, and the ModelRunnerV1 source proof. Re-scan the report for absolute protected paths and transcript-like fields.

Expected: the report contains no raw text/audio and is not tracked by Git.

### Task 2: Start and prove the isolated ModelRunnerV1 candidate

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-modelrunner-v1/task-2-report.md`
- Modify: none

**Interfaces:**
- Consumes: Task 1 preflight, exact candidate image, exact model snapshot, network `voice_default`.
- Produces: ready loopback candidate and gateway, exact runtime environment proof, real API response, and candidate-correlated CUDA evidence.

- [x] **Step 1: Start exactly one candidate with the single experimental variable**

Run on the GX10:

```bash
docker run -d \
  --name qwen3-asr-spark-mrv1-test \
  --pull never \
  --gpus all \
  --shm-size 4g \
  --restart no \
  --network voice_default \
  -p 127.0.0.1:18001:8001 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e ENABLE_NVFP4_SM100=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /home/volsch/.cache/huggingface:/root/.cache/huggingface:ro \
  dgx-qwen3-asr:spark-vllm-test \
  vllm serve /root/.cache/huggingface/hub/models--UrocyonF--Qwen3-ASR-1.7B-NVFP4/snapshots/61ad4d533c64e033a750b66c44aad6f18634997e \
  --served-model-name qwen3-asr \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.08 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --trust-remote-code
```

Expected: one running container, restart count `0`, restart policy `no`, exact image ID and command.

- [x] **Step 2: Prove runtime selection and bounded readiness**

Poll `http://127.0.0.1:18001/v1/models` for at most 300 seconds. Require HTTP `200` with exactly one model ID `qwen3-asr`. Then inspect the container environment and require exactly one `VLLM_USE_V2_MODEL_RUNNER=0` entry. Reject logs containing `Using V2 Model Runner`.

Expected: model readiness and the V1 source/env proof agree. Container exit, timeout, restart, a V2 log marker, or an unsupported-runner error rejects the candidate without changing any other variable.

- [x] **Step 3: Start the immutable normalized gateway**

Run on the GX10:

```bash
docker run -d \
  --name asr-modelrunner-v1-gateway \
  --pull never \
  --restart no \
  --network voice_default \
  -p 127.0.0.1:18002:8000 \
  -e ASR_BACKEND_URL=http://qwen3-asr-spark-mrv1-test:8001 \
  sha256:f434a9c6533dd050968e56f547206ca4660ab4aa53d2f9d85ee6b5d8f12f473c
```

Expected: gateway is ready, has the exact image ID and backend environment, and exposes no non-loopback host port.

- [x] **Step 4: Run one authorized smoke request with CUDA correlation**

Use one owner-only German WAV from the frozen benchmark through the gateway with `language=de`. Correlate the candidate container PID with `nvidia-smi` compute PIDs and require nonzero SM during the request. Validate HTTP `200` and a JSON object containing only the normalized string `text` field at the gateway boundary; do not print the value.

Expected: real request, gateway normalization, candidate PID correlation, and nonzero CUDA activity all pass.

- [x] **Step 5: Write the transcript-free live report and recheck production**

Record readiness duration, response shape only, runtime env/source proof, CUDA boolean/maximum aggregate SM, and current production invariant. Do not record PIDs, raw GPU rows, paths, audio, or text.

Expected: production still matches Task 1 exactly.

### Task 3: Run the staged non-speech decision and conditional full gate

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-modelrunner-v1/performance-report.md`
- Modify: none

**Interfaces:**
- Consumes: ready Task 2 candidate/gateway, exact owner-only runner and ten-case subset, corrected same-model production baseline of `5` hallucinated words.
- Produces: transcript-free discriminator result and, only on pass, a complete same-model eligibility result.

- [x] **Step 1: Run the exact ten non-speech probes once**

Use the owner-only runner and the same fixed ten case IDs, bytes, ordering, `language=de`, gateway contract, timeout, and scorer hashes from `task2-same-model-1p7b-21case`. Send all ten requests once to `http://127.0.0.1:18002/v1/audio/transcriptions`.

Expected: `10/10` successful responses and a transcript-free safe-result artifact. Do not retry individual cases or inspect text before the aggregate decision.

- [x] **Step 2: Apply the fixed discriminator**

Validate the safe-result schema, hashes, finite numeric values, exact ten-case map shape, and forbidden transcript-bearing shapes. Require:

```text
non_speech_hallucinated_words <= 5
malformed_responses == 0
non_transcript_addition_cases == []
language_change_cases == []
command_risk_cases == []
normalized_nondeterministic_cases == []
candidate_cuda == true
```

Expected: one unambiguous PASS or FAIL. On FAIL, skip Steps 3 and 4 and continue directly to Task 4 cleanup.

- [x] **Step 3: Skip the unchanged full A/B preparation on discriminator FAIL**

Revalidate production image/model/command and candidate image/model/command. Start an identical second gateway for production using the same immutable gateway image and only a different `ASR_BACKEND_URL`. Assert both gateway configurations are byte-equivalent except backend URL.

Expected: production and candidate identities match the corrected same-model design, and both gateways use the same image and request normalization.

- [x] **Step 4: Skip the full fixed series on discriminator FAIL**

Run exactly in this order:

```text
production quality: 70
candidate quality: 70
production load: 12
candidate load: 12
```

Use the existing protected runner, frozen corpus/subset hashes, request ordering, concurrency, `language=de`, timeout, scoring, CUDA correlation, and safe-result validation unchanged. Apply the existing gates: no quality or non-speech regression; p50 ratio at most `1.05`; nearest-rank p90 ratio at most `1.10`; all requests successful; CUDA true for both.

Expected: an eligibility result based on all existing gates. No tuning, selective rerun, outlier deletion, or gate change is allowed.

- [x] **Step 5: Write the transcript-free performance report**

Record exact identities, request counts, aggregate quality map, per-case non-speech word counts, latency aggregates/ratios, CUDA booleans and aggregate maxima, gate map, conditional-step decision, and protected artifact hashes. Omit paths and raw content.

### Task 4: Clean up, record the outcome, and verify the branch

**Files:**
- Modify: `docs/superpowers/specs/2026-08-09-spark-vllm-asr-modelrunner-v1-design.md`
- Modify: `docs/superpowers/plans/2026-08-09-spark-vllm-asr-modelrunner-v1.md`
- Test: `tests/test_dgx_deployment.py`

**Interfaces:**
- Consumes: Task 3 discriminator/full-gate decision.
- Produces: zero test containers, unchanged production, tracked historical outcome, green repository, one focused documentation commit.

- [x] **Step 1: Remove only the named benchmark containers**

Remove `asr-modelrunner-v1-gateway`, `qwen3-asr-spark-mrv1-test`, and the production gateway only if Task 3 Step 3 created it. Require zero containers with those exact names afterward. Retain both immutable images.

Expected: no test container remains; no image is pruned.

- [x] **Step 2: Prove the final production invariant**

Run the same `docker inspect` command from Task 1 and compare the complete output byte-for-byte with the preflight value.

Expected: same image and UrocyonF 1.7B command, `running|healthy|0|unless-stopped`.

- [x] **Step 3: Append the exact execution outcome**

Update the design and this plan with the immutable candidate/gateway IDs, discriminator counts, whether the full gate ran, final eligibility, cleanup counts, and production invariant. Mark rollout prohibited unless every complete gate passed; even a complete pass still requires separate user rollout authorization.

- [x] **Step 4: Run fresh repository verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check agent tests dgx
.venv/bin/ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
git status --short
```

Expected: all tests pass, Ruff and formatting are clean, Compose renders, diff check passes, and only the two intended documentation files are modified.

- [x] **Step 5: Commit the evidenced outcome**

```bash
git add \
  docs/superpowers/specs/2026-08-09-spark-vllm-asr-modelrunner-v1-design.md \
  docs/superpowers/plans/2026-08-09-spark-vllm-asr-modelrunner-v1.md
git commit -m "docs(asr): record ModelRunnerV1 diagnostic"
```

Expected: one commit containing only the two documentation files; owner-only reports remain untracked.
