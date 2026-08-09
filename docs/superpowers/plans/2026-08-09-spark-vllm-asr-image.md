# Spark-vLLM ASR Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current 33.29 GB Qwen3-ASR production image with a smaller digest-pinned `eugr/spark-vllm` derivative while preserving the exact model revision, OpenAI transcription contract, German quality, CUDA execution, and production latency.

**Architecture:** `dgx/asr/Dockerfile` inherits the already validated ARM64 Spark-vLLM image and adds only the two missing audio decoder distributions from a direct-only hash lock. Repository Compose builds that derivative but keeps the existing model path, command, healthcheck, networks, and GPU settings. A separately named loopback-only candidate is built and tested on the GX10 while production stays healthy; the existing common ASR gateway and protected `asr-companion-de-v1` corpus decide quality and latency eligibility before a narrowly scoped Portainer update.

**Tech Stack:** Docker/Compose, ARM64 NVIDIA GB10, CUDA 13.0, PyTorch 2.11, Spark-vLLM, Qwen3-ASR-0.6B, uv, pytest, Ruff, Portainer, the existing ASR benchmark gateway and protected German corpus.

## Global Constraints

- Base image: `eugr/spark-vllm@sha256:1d861bef8a6c0851140cec2575ebd32342d55bc0fd28ad4c6ca178269e9d1cff`.
- Expected inherited runtime: Python `3.12.3`, Torch `2.11.0+cu130`, `torch.version.cuda == "13.0"`, vLLM `0.26.1rc1.dev468+g6b5bec7be.d20260807`, Triton `3.6.0`, and FlashInfer `0.6.18`.
- Added distributions: only `soundfile==0.13.1` and `av==17.0.1`, installed with `--require-hashes --no-deps`. Inherited `cffi==2.1.1` is reused.
- Do not install or rebuild Torch, vLLM, Triton, FlashInfer, Flash-Attention, CUDA, CFFI, or a compiler, and do not patch inherited vLLM source.
- Model snapshot: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0`.
- API: multipart WAV plus `language=de` to `POST /v1/audio/transcriptions`; the response must be a JSON object with a string `text` field and may contain additive fields such as `usage`.
- Candidate: image `dgx-qwen3-asr:spark-vllm-test`, container `qwen3-asr-spark-test`, host bind `127.0.0.1:18001:8001`.
- Current production invariant: image `sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3`, size `33,294,535,748`, `running|healthy|0|unless-stopped`.
- Portainer stack is `voice`, stack ID `16`, endpoint ID `1`, API origin `http://192.168.68.41:9000/api`.
- Never print, commit, or copy out raw audio, references, transcripts, caller data, Portainer environments, tokens, or model-cache content.
- Until the corrected Task 7 A/B passes, do not stop, recreate, update, or replace production `qwen3-asr`.

---

### Task 1: Re-prove the repository, base-image, corpus, and live baselines

**Files:**
- No repository files changed.
- Create outside Git during execution: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/task-1-report.md`

**Interfaces:**
- Consumes: design commit `8ae3a04`, the current feature branch, digest-pinned base image, protected corpus freeze, and live production.
- Produces: a non-private preflight report and an explicit stop/go decision before implementation.

- [ ] **Step 1: Confirm the approved branch and clean starting point**

```bash
cd /home/volsch/projekte/voip-agent
test "$(git branch --show-current)" = "perf/spark-vllm-asr"
git rev-parse --verify 8ae3a04^{commit}
test -z "$(git status --porcelain=v1)"
```

Expected: all commands exit 0. If another user change is present, preserve it and stop before editing overlapping files.

- [ ] **Step 2: Run the current repository gates**

```bash
pytest -q
ruff check agent tests dgx
ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
```

Expected: all commands pass before the ASR image is changed.

- [ ] **Step 3: Re-read the exact live production invariant**

```bash
ssh volsch@192.168.68.41 'docker inspect qwen3-asr --format "{{.Image}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|{{.RestartCount}}|{{.HostConfig.RestartPolicy.Name}}"'
ssh volsch@192.168.68.41 'docker image inspect sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3 --format "{{.Id}}|{{.Size}}|{{.Architecture}}|{{.Os}}"'
```

Expected: exact values are `sha256:38f255...f2cbe3|running|healthy|0|unless-stopped` and `...|33294535748|arm64|linux`. Any drift requires treating the newly observed healthy image as the baseline; never roll production back to the older recorded image without explicit authorization.

- [ ] **Step 4: Re-prove the digest-pinned base without model or GPU mutation**

```bash
ssh volsch@192.168.68.41 'docker run --rm --pull never --entrypoint python3 \
  eugr/spark-vllm@sha256:1d861bef8a6c0851140cec2575ebd32342d55bc0fd28ad4c6ca178269e9d1cff \
  -c "import importlib.metadata as m, importlib.util as u, torch, vllm; assert torch.__version__ == '\''2.11.0+cu130'\''; assert torch.version.cuda == '\''13.0'\''; assert m.version('\''vllm'\'') == '\''0.26.1rc1.dev468+g6b5bec7be.d20260807'\''; assert u.find_spec('\''vllm.model_executor.models.qwen3_asr'\''); assert u.find_spec('\''soundfile'\'') is None; assert u.find_spec('\''av'\'') is None"'
```

Expected: exit 0. Do not add Flash-Attention merely because `flash_attn` is absent; Spark-vLLM's inherited attention stack is the baseline.

- [ ] **Step 5: Validate only metadata and permissions of the protected corpus**

Use the exact protected root:

```bash
ssh volsch@192.168.68.41 'freeze=/home/volsch/ai-companion/runtime/asr-synthetic/freeze-a4dfee49011727ad9a49fefbe76e7352a423c604; test "$(stat -c %a "$freeze")" = 700; test "$(stat -c %a "$freeze/benchmarks/asr/corpus-asr-companion-de-v1.json")" = 600; test "$(stat -c %a "$freeze/benchmarks/asr/latency-subset-v1.json")" = 600; test -x "$freeze/.venv/bin/python"; test -x "$freeze/.venv/bin/uvicorn"'
```

Expected: exit 0 without printing the manifest. Record only corpus version/hash, file counts, modes, and pass/fail in the task report.

---

### Task 2: Lock the minimal audio extension with regression tests

**Files:**
- Create: `dgx/asr/requirements-audio-arm64.in`
- Create: `dgx/asr/requirements-audio-arm64.lock`
- Modify: `tests/test_dgx_deployment.py`

**Interfaces:**
- Consumes: Python 3.12 ARM64 wheel availability and the fixed base-image contract.
- Produces: a direct-only, hash-locked two-package extension and tests that reject accidental core-stack replacement.

- [ ] **Step 1: Write failing deployment tests first**

Add tests that require all of the following:

- the ASR input contains exactly `soundfile==0.13.1` and `av==17.0.1`;
- the generated lock has exactly those two normalized headers and at least one valid SHA-256 hash per entry;
- the Dockerfile installs the lock with `--require-hashes` and `--no-deps`;
- no logical Docker instruction runs `pip install` for `torch`, `vllm`, `triton`, `flashinfer`, `flash-attn`, `flash_attn`, or `cffi`;
- the Dockerfile's `FROM` uses the exact eugr digest;
- Compose builds `qwen3-asr` from context `.` and `asr/Dockerfile`, with no mutable external ASR image tag;
- all existing ASR model, command, health, GPU, network, shared-memory, restart, label, and offline assertions remain true.

Run:

```bash
pytest -q tests/test_dgx_deployment.py -k 'asr and (spark or audio or image)'
```

Expected: the new tests fail because the lock and digest-pinned derivative do not exist yet.

- [ ] **Step 2: Create the direct-only input**

Create `dgx/asr/requirements-audio-arm64.in` with exactly:

```text
soundfile==0.13.1
av==17.0.1
```

- [ ] **Step 3: Resolve the Python 3.12 ARM64 lock without dependencies**

```bash
uvx --from uv uv pip compile \
  --python-version 3.12 \
  --python-platform aarch64-manylinux_2_39 \
  --no-deps \
  --generate-hashes \
  --output-file dgx/asr/requirements-audio-arm64.lock \
  dgx/asr/requirements-audio-arm64.in
```

Expected: the lock contains exactly two distribution headers, with hash-pinned ARM64/Python 3.12-compatible artifacts. If either exact version has no compatible artifact, stop; do not loosen the version or add build tooling without a reviewed plan change.

- [ ] **Step 4: Make the lock tests green**

```bash
pytest -q tests/test_dgx_deployment.py -k 'asr and (spark or audio or image)'
git diff --check
```

Expected: lock-focused assertions pass; Docker/Compose-focused assertions may remain red until Task 3.

- [ ] **Step 5: Commit the lock and its contract**

```bash
git add dgx/asr/requirements-audio-arm64.in dgx/asr/requirements-audio-arm64.lock tests/test_dgx_deployment.py
git commit -m "build(asr): lock Spark audio dependencies"
```

---

### Task 3: Implement the digest-pinned Spark-vLLM derivative and Compose wiring

**Files:**
- Modify: `dgx/asr/Dockerfile`
- Modify: `dgx/docker-compose.yml`
- Modify if a newly observed contract needs a regression: `tests/test_dgx_deployment.py`

**Interfaces:**
- Consumes: Task 2's two-package lock and existing Compose service contract.
- Produces: a reproducible ASR image recipe whose only new runtime distributions are SoundFile and PyAV.

- [ ] **Step 1: Confirm the Docker/Compose tests are red**

```bash
pytest -q tests/test_dgx_deployment.py -k 'asr and (spark or audio or image)'
```

Expected: failures identify the old AEON `FROM`, mutable external Compose image, and unbounded installs.

- [ ] **Step 2: Replace `dgx/asr/Dockerfile` with the minimal derivative**

Implement this structure:

```dockerfile
FROM eugr/spark-vllm@sha256:1d861bef8a6c0851140cec2575ebd32342d55bc0fd28ad4c6ca178269e9d1cff

COPY asr/requirements-audio-arm64.lock /tmp/requirements-audio-arm64.lock
RUN python3 -m pip install --no-cache-dir --require-hashes --no-deps \
      -r /tmp/requirements-audio-arm64.lock \
 && python3 -c "import importlib.metadata as m, importlib.util as u, torch; assert torch.__version__ == '2.11.0+cu130'; assert torch.version.cuda == '13.0'; assert m.version('vllm') == '0.26.1rc1.dev468+g6b5bec7be.d20260807'; assert m.version('soundfile') == '0.13.1'; assert m.version('av') == '17.0.1'; assert m.version('cffi') == '2.1.1'; assert u.find_spec('vllm.model_executor.models.qwen3_asr')"

EXPOSE 8001
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python3 -c "import urllib.request,sys,json; r=urllib.request.urlopen('http://127.0.0.1:8001/v1/models',timeout=4); d=json.loads(r.read()); sys.exit(0 if any(model.get('id') == 'qwen3-asr' for model in d.get('data',[])) else 1)" || exit 1
```

Do not add an `ENTRYPOINT`, a model-specific `CMD`, Flash-Attention, compiler packages, or a vLLM patch. The inherited executable plus Compose command remains authoritative.

- [ ] **Step 3: Change only ASR image ownership in Compose**

Replace the ASR service's external `image:` line with:

```yaml
    build:
      context: .
      dockerfile: asr/Dockerfile
```

Keep every other `qwen3-asr` value byte-for-byte unchanged: container name, devices, environment, model revision, command arguments, cache mount, networks, shared memory, restart, label, and healthcheck.

- [ ] **Step 4: Run focused Red/Green verification**

```bash
pytest -q tests/test_dgx_deployment.py
ruff check tests/test_dgx_deployment.py
ruff format --check tests/test_dgx_deployment.py
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
```

Expected: all pass. Inspect rendered Compose to confirm the model snapshot revision and every command flag are unchanged.

- [ ] **Step 5: Self-review the implementation diff**

```bash
git diff -- dgx/asr dgx/docker-compose.yml tests/test_dgx_deployment.py
```

Reject any unrelated dependency, fallback, port publication, model change, or production setting drift.

- [ ] **Step 6: Commit the derivative**

```bash
git add dgx/asr/Dockerfile dgx/docker-compose.yml tests/test_dgx_deployment.py
git commit -m "build(asr): use digest-pinned Spark vLLM"
```

---

### Task 4: Build and statically validate the candidate on GX10

**Files:**
- Modify only if an evidenced failure requires a TDD fix: files from Tasks 2-3.
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/task-4-report.md`

**Interfaces:**
- Consumes: committed Docker recipe and unchanged healthy production.
- Produces: candidate image `dgx-qwen3-asr:spark-vllm-test` with exact identity, size, platform, and static import proof.

- [ ] **Step 1: Recheck production immediately before the build**

Require the Task 1 invariant, allowing only a newly observed healthy image if it is explicitly adopted as the updated baseline. Record the image ID and restart count before and after the build.

- [ ] **Step 2: Build the exact candidate through the remote Docker host**

```bash
docker --host ssh://volsch@192.168.68.41 build \
  --progress=plain \
  --file dgx/asr/Dockerfile \
  --tag dgx-qwen3-asr:spark-vllm-test \
  dgx
```

Expected: build succeeds without pulling another base digest and without source compilation.

- [ ] **Step 3: Inspect size, platform, layers, and installed distributions**

```bash
ssh volsch@192.168.68.41 'docker image inspect dgx-qwen3-asr:spark-vllm-test --format "{{.Id}}|{{.Size}}|{{.Architecture}}|{{.Os}}"'
ssh volsch@192.168.68.41 'docker run --rm --pull never --entrypoint python3 dgx-qwen3-asr:spark-vllm-test -c "import importlib.metadata as m, importlib.util as u, torch; assert torch.__version__ == '\''2.11.0+cu130'\''; assert torch.version.cuda == '\''13.0'\''; assert m.version('\''vllm'\'') == '\''0.26.1rc1.dev468+g6b5bec7be.d20260807'\''; assert m.version('\''triton'\'') == '\''3.6.0'\''; assert m.version('\''flashinfer-python'\'') == '\''0.6.18'\''; assert m.version('\''soundfile'\'') == '\''0.13.1'\''; assert m.version('\''av'\'') == '\''17.0.1'\''; assert m.version('\''cffi'\'') == '\''2.1.1'\''; assert u.find_spec('\''vllm.model_executor.models.qwen3_asr'\'')"'
```

Expected: ARM64/Linux, size below `33,294,535,748`, and every assertion passes. Inspect `docker history --no-trunc` to confirm the derivative adds only the lock, one pip install layer, metadata, and healthcheck.

- [ ] **Step 4: Handle failures systematically and minimally**

For any build/import failure, capture the first causal error, reproduce it with the smallest command, add a failing regression test, and make only the smallest verified correction. Rebuild after every Dockerfile or lock change. Do not infer missing dependencies from package extras.

- [ ] **Step 5: Run fresh repository verification after any correction**

```bash
pytest -q tests/test_dgx_deployment.py
pytest -q
ruff check agent tests dgx
ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
```

Commit each evidenced correction separately using `fix(asr): ...`. Leave production unchanged and the candidate image retained.

---

### Task 5: Prove isolated model load, API compatibility, and real CUDA execution

**Files:**
- No planned repository changes.
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/task-5-report.md`

**Interfaces:**
- Consumes: Task 4 candidate, pinned local model cache, and a private 16 kHz German WAV.
- Produces: health, API-shape, latency, GPU-process, and nonzero-SM proof without transcript disclosure.

- [ ] **Step 1: Start exactly one isolated candidate**

```bash
ssh volsch@192.168.68.41 'docker rm -f qwen3-asr-spark-test >/dev/null 2>&1 || true; docker run -d --name qwen3-asr-spark-test --pull never --gpus all --shm-size 4g --restart no --label com.docker.compose.project=asr-benchmark --label com.docker.compose.service=asr-worker --network voice_default -p 127.0.0.1:18001:8001 -e NVIDIA_VISIBLE_DEVICES=all -e ENABLE_NVFP4_SM100=0 -e VLLM_TEST_FORCE_FP8_MARLIN=1 -e VLLM_USE_FLASHINFER_SAMPLER=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -v /home/volsch/.cache/huggingface:/root/.cache/huggingface:ro dgx-qwen3-asr:spark-vllm-test vllm serve /root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 --served-model-name qwen3-asr --host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.08 --max-model-len 8192 --max-num-seqs 4 --trust-remote-code'
```

Expected: only the named candidate is created; production remains untouched.

- [ ] **Step 2: Wait for the loaded-model health contract**

Poll `http://127.0.0.1:18001/v1/models` on the GX10 for at most 300 seconds. Require HTTP 200 and exactly one entry whose `id` is `qwen3-asr`. On timeout or container exit, capture logs, remove only the candidate, and diagnose before changing code.

- [ ] **Step 3: Send one private real WAV and validate only response shape**

Use an existing authorized 16 kHz mono German test WAV without printing its path or content. Send it from the GX10 to the loopback candidate with `language=de`. Require:

- HTTP 200;
- `Content-Type: application/json`;
- response is a JSON object whose `text` value is a string; additive fields are allowed;
- response and timing stay in an owner-only temporary directory and are never copied to Git or the report.

- [ ] **Step 4: Correlate candidate PID and nonzero SM with the request**

During a second real request, sample `nvidia-smi --query-compute-apps=pid` and `nvidia-smi dmon -s u`. Require at least one PID belonging to `qwen3-asr-spark-test` and maximum aggregate SM utilization greater than zero. Record only booleans, maximum SM percentage, image ID, and request latency.

- [ ] **Step 5: Verify production and clean the isolated container**

```bash
ssh volsch@192.168.68.41 'docker inspect qwen3-asr --format "{{.Image}}|{{.State.Status}}|{{.State.Health.Status}}|{{.RestartCount}}|{{.HostConfig.RestartPolicy.Name}}"; docker rm -f qwen3-asr-spark-test >/dev/null; test "$(docker ps -aq --filter name=^/qwen3-asr-spark-test$ | wc -l)" -eq 0'
```

Expected: production invariant is unchanged; candidate image remains available.

---

### Task 6: Remove the unnecessary response patch and revalidate the candidate

**Files:**
- Modify: `dgx/asr/Dockerfile`
- Delete: `dgx/asr/patch_vllm_transcription_contract.py`
- Modify: `tests/test_dgx_deployment.py`
- Modify: `tests/test_stt.py`
- Append outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/task-6-contract-correction-report.md`

**Interfaces:**
- Consumes: design correction commit `80c86e6`, the real client behavior in `agent/stt.py`, and the Task 4 unpatched recipe.
- Produces: an unpatched candidate whose API contains a string `text` field and whose image, model, CUDA, and production-isolation gates pass.

- [ ] **Step 1: Add failing tests for the corrected contract**

Add this client regression to `tests/test_stt.py`:

```python
@respx.mock
async def test_transcribe_accepts_additive_response_fields(stt):
    respx.post("http://stt:8001/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={"text": "Hallo Welt", "usage": {"seconds": 0.5}},
        )
    )

    assert await stt.transcribe(_pcm_16k()) == "Hallo Welt"
```

Replace the patch-retention deployment tests with a regression that requires:

```python
assert "patch_vllm_transcription_contract.py" not in dockerfile
assert "TranscriptionResponse" not in dockerfile
assert not (ROOT / "dgx/asr/patch_vllm_transcription_contract.py").exists()
```

Run:

```bash
.venv/bin/pytest -q tests/test_stt.py::test_transcribe_accepts_additive_response_fields \
  tests/test_dgx_deployment.py -k 'additive or unpatched'
```

Expected: the client test passes against existing behavior; the deployment regression fails because the patch file and Docker invocation still exist. This is the Red proof that the image violates the corrected design.

- [ ] **Step 2: Remove only the response patch**

Delete `dgx/asr/patch_vllm_transcription_contract.py`, its Docker `COPY`/`RUN`, the installed-runtime `TranscriptionResponse` serialization assertion, and patch-specific helpers/fixtures. Restore the Task 3 post-install assertion that checks only inherited versions, locked audio packages, CFFI, and Qwen3-ASR module registration. Do not change the lock, base digest, Compose, model command, or healthcheck.

- [ ] **Step 3: Run focused and full Green gates**

```bash
.venv/bin/pytest -q tests/test_stt.py tests/test_dgx_deployment.py
.venv/bin/pytest -q
.venv/bin/ruff check agent tests dgx
.venv/bin/ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
```

Expected: all pass, and no test or image source requires exact response keys.

- [ ] **Step 4: Commit the corrected image contract**

```bash
git add dgx/asr/Dockerfile dgx/asr/patch_vllm_transcription_contract.py \
  tests/test_dgx_deployment.py tests/test_stt.py
git commit -m "fix(asr): follow production transcription contract"
```

- [ ] **Step 5: Rebuild and statically identify the unpatched candidate**

```bash
docker --host ssh://volsch@192.168.68.41 build \
  --progress=plain \
  --file dgx/asr/Dockerfile \
  --tag dgx-qwen3-asr:spark-vllm-test \
  dgx
```

Require ARM64/Linux, size below `33,294,535,748`, exact inherited and added versions, Qwen3-ASR module availability, and no patch file in the image. Record the new candidate ID; do not assume it equals the earlier Task 4 ID even if cache reuse makes it so.

- [ ] **Step 6: Repeat real model, API, and CUDA acceptance**

Start the exact Task 5 candidate command. Require `/v1/models` with `qwen3-asr`, HTTP 200 JSON whose `text` field is a string, candidate PID correlation, and nonzero SM during a real authorized German request. Additive response fields are allowed and only their key names/types may be recorded. Remove only `qwen3-asr-spark-test`, retain the image, and prove production remains `running|healthy|0|unless-stopped` on the original image.

---

### Task 7: Run the corrected production-versus-candidate German A/B

**Files:**
- No tracked repository files changed.
- Create only in the plan workspace: `gateway-overlay/`
- Create only in the plan workspace: `gateway-overlay/tests/test_asr_gateway_additive_response.py`
- Protected raw artifacts remain only below `/home/volsch/ai-companion/runtime/asr-image-ab-*` on GX10.
- Replace outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/performance-ab-report.md`

**Interfaces:**
- Consumes: Task 6 unpatched candidate, current production, common gateway source in `/home/volsch/projekte/ai-companion/.worktrees/asr-baseline-first`, and the frozen protected corpus.
- Produces: one fully successful paired series and a transcript-free comparison with explicit quality, CUDA, p50, and p90 gates.

- [ ] **Step 1: Preserve and exclude the invalid first attempt**

Keep the original owner-only evidence proving the unchanged gateway produced production `0/70` and `0/12`. Mark it `invalid_gateway_contract` and assert its raw files are never supplied to the corrected comparator. The corrected series starts with four new exclusive output paths.

- [ ] **Step 2: Create a private gateway overlay and failing test**

Create the private copy with:

```bash
overlay_root=/home/volsch/projekte/voip-agent/.worktrees/spark-vllm-asr/.superpowers/sdd/2026-08-09-spark-vllm-asr-image/gateway-overlay
install -d -m 0700 "$overlay_root"
rsync -a --delete \
  --exclude .git --exclude .venv --exclude .pytest_cache --exclude .ruff_cache \
  --exclude benchmarks/results --exclude runtime \
  /home/volsch/projekte/ai-companion/.worktrees/asr-baseline-first/ \
  "$overlay_root/"
```

Create `gateway-overlay/tests/test_asr_gateway_additive_response.py` with:

```python
import httpx
import pytest

from asr_benchmark_gateway.client import (
    BackendClient,
    InvalidBackendResponse,
)


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Hallo"},
        {"text": "Hallo", "usage": {"seconds": 0.5}},
    ],
)
async def test_backend_accepts_text_with_additive_fields(payload):
    async def handler(request):
        assert request.url.path == "/v1/audio/transcriptions"
        return httpx.Response(200, json=payload)

    client = BackendClient(
        "http://backend",
        1,
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await client.transcribe(b"wav", "de", 1.0) == "Hallo"
    finally:
        await client.aclose()


@pytest.mark.parametrize("payload", [{}, {"text": 7}])
async def test_backend_rejects_missing_or_non_string_text(payload):
    async def handler(request):
        assert request.url.path == "/v1/audio/transcriptions"
        return httpx.Response(200, json=payload)

    client = BackendClient(
        "http://backend",
        1,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(InvalidBackendResponse):
            await client.transcribe(b"wav", "de", 1.0)
    finally:
        await client.aclose()
```

Run:

```bash
cd "$overlay_root"
uv sync --frozen
uv run pytest -q tests/test_asr_gateway_additive_response.py
```

Expected Red: the additive-field case raises `InvalidBackendResponse` because the copied client still requires exact keys.

- [ ] **Step 3: Implement the minimal normalization overlay**

In only the private copied `src/asr_benchmark_gateway/client.py`, replace the exact-key condition with:

```python
if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
    raise InvalidBackendResponse
return payload["text"]
```

Run:

```bash
uv run pytest -q tests/test_asr_gateway.py tests/test_asr_gateway_additive_response.py
uv run ruff check src/asr_benchmark_gateway tests/test_asr_gateway_additive_response.py
uv run ruff format --check src/asr_benchmark_gateway tests/test_asr_gateway_additive_response.py
```

Expected: both accepted shapes pass; invalid/missing text remains fail-closed. Record SHA-256 hashes for the original client, overlaid client, overlay test, Dockerfile, `pyproject.toml`, and lock.

- [ ] **Step 4: Build one normalized common gateway image**

Build from the private overlay:

```bash
docker --host ssh://volsch@192.168.68.41 build \
  --file "$overlay_root/deploy/asr-benchmark-gateway.Dockerfile" \
  --tag asr-benchmark-gateway:spark-image-ab-normalized \
  "$overlay_root"
```

Record its immutable image ID and source hashes. Both production and candidate gateway containers must use that exact image ID; no per-side source or environment difference is allowed.

- [ ] **Step 5: Create protected run state and start both sides**

On GX10 create a unique mode-`0700` run directory with `umask 077`. Stage only `asr_benchmark.py`, `asr_corpus.py`, and `asr_score.py` as mode `0600`; never copy corpus files off GX10. Start the Task 6 candidate on `voice_default`, then identical gateways:

- `asr-image-ab-gateway-prod` -> `http://qwen3-asr:8001`, loopback `18110`;
- `asr-image-ab-gateway-candidate` -> `http://qwen3-asr-spark-test:8001`, loopback `18111`.

Both must report `{"status":"ok"}` and resolve to the same gateway image ID.

- [ ] **Step 6: Run one complete paired series**

Run four new exclusive mode-`0600` outputs in this order:

```text
production quality: 70 cases, concurrency 1, gateway 18110
candidate quality:  70 cases, concurrency 1, gateway 18111
production load:    frozen 12-case subset, concurrency 4, gateway 18110
candidate load:     frozen 12-case subset, concurrency 4, gateway 18111
```

Every request uses identical bytes, `language=de`, case order, and concurrency. Observe container-correlated CUDA and nonzero SM during each load phase. No raw JSON is printed.

- [ ] **Step 7: Produce and validate the transcript-free result**

The protected comparator must reject duplicate keys, unknown aggregate fields, failures, non-finite metrics, case-order drift, corpus/hash drift, forbidden content keys, and either invalid first-attempt path. It may read transcripts only inside the protected directory. Require:

- 70/70 quality and 12/12 load successes per side;
- candidate WER, CER, macro WER, real-path macro WER, empty-speech count,
  malformed count, non-speech hallucinated words, language-change cases,
  non-transcript additions, and command-risk cases no worse than production;
- candidate entity recall and every non-null number/time/date accuracy no lower
  than production;
- candidate p50 divided by production p50 at most `1.05`;
- candidate nearest-rank p90 divided by production p90 at most `1.10`;
- both CUDA observations true.

Include exact safe aggregate maps and hashes for candidate, production, model revision, corpus, subset, runner, comparator, gateway overlay sources, and built gateway image. Validate the mode-`0600` result again from disk.

- [ ] **Step 8: Use the single replication rule only for a shared-host latency anomaly**

If and only if quality, success, and CUDA gates pass but one isolated paired latency anomaly affects both sides, preserve series 1 and run one complete unchanged replication. Report both series and combined 24-sample load aggregates. Never delete outliers or selectively rerun cases.

- [ ] **Step 9: Clean exact benchmark containers and report**

Remove only the candidate and the two named gateways. Recheck production `running|healthy|0|unless-stopped`. The report contains safe aggregates, hashes, ratios, CUDA booleans, and eligibility only; no raw records or transcripts.

---

### Task 8: Roll out the accepted image with exact rollback protection

**Execution status: NOT RUN.** The corrected Task 7 A/B failed the mandatory
entity-recall and number/time non-degradation gates. A follow-up full-corpus
language-hint diagnostic confirmed that `language=de` changes some candidate
outputs and improves non-speech behavior, but does not change the failed
quality metrics. The all-green prerequisite below is therefore false, and no
Portainer mutation, release tag, rollback tag, or production replacement is
authorized by this plan execution.

**Files:**
- No planned repository changes.
- Protected backups and environments remain in a mode-`0700` local run directory.
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/task-8-report.md`

**Interfaces:**
- Consumes: all-green Task 6-7 evidence and current live stack identity.
- Produces: updated healthy ASR production or automatic restoration of the exact pre-update stack.

- [ ] **Step 1: Re-resolve all identities immediately before mutation**

Require stack `voice`, ID `16`, endpoint `1`; production healthy with zero restarts stable for 30 seconds; candidate image ID equal to the Task 7 image; and clean repository. If production changed since Task 7, stop rather than replacing newer production.

- [ ] **Step 2: Create immutable rollback and release tags**

```bash
release_tag="dgx-qwen3-asr:spark-vllm-$(git rev-parse --short=12 HEAD)"
ssh volsch@192.168.68.41 'docker image tag sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3 dgx-qwen3-asr:rollback-38f255cd'
docker --host ssh://volsch@192.168.68.41 image tag dgx-qwen3-asr:spark-vllm-test "$release_tag"
```

- [ ] **Step 3: Back up exact Portainer Compose and environment**

Use the existing `scripts/portainer.py` from `asr-baseline-first` with `GX10_PORTAINER_TOKEN_FILE=/home/volsch/.gx10_portainer_token`, base URL `http://192.168.68.41:9000/api`, endpoint `1`, and `backup --name voice`. Store Compose, environment, and redacted metadata as new mode-`0600` files; never print environment values.

- [ ] **Step 4: Prove the candidate stack has one semantic change**

Replace only `services.qwen3-asr.image` with the immutable release tag. Parse both YAML files and assert every other value is deeply equal; require the textual diff to contain one removed and one added image line.

- [ ] **Step 5: Apply without pull or prune**

Run `portainer.py upsert --name voice --compose "$run_dir/voice-candidate.yml" --environment "$run_dir/voice-environment.json" --expected-stack-id 16 --apply`. The accepted image is local and the helper sets `Prune=false`.

- [ ] **Step 6: Run production acceptance**

Within 300 seconds require the accepted image ID/release tag, `running|healthy|0|unless-stopped` stable for 30 seconds, `/v1/models` with `qwen3-asr`, one real German request returning a JSON object with string `text`, candidate-specific CUDA PID plus nonzero SM, and the rollback tag still resolving to the pre-update image.

- [ ] **Step 7: Roll back on any failed gate**

Upsert the exact backup Compose and environment, then require the old image ID, health/restart invariant, model list, real request, and CUDA proof. Do not retry rollout in the same task.

---

### Task 9: Run final verification and whole-branch review

**Files:**
- Modify only if evidence invalidated it: `docs/superpowers/specs/2026-08-09-spark-vllm-asr-image-design.md`
- Modify only for execution truth: `docs/superpowers/plans/2026-08-09-spark-vllm-asr-image.md`

**Interfaces:**
- Consumes: final implementation, A/B, rollout, and rollback evidence.
- Produces: a clean reviewable branch with no private artifacts.

- [ ] **Step 1: Run fresh complete gates**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check agent tests dgx
.venv/bin/ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
```

- [ ] **Step 2: Reverify final image and live state**

Repeat Task 6 static identity/import checks and Task 8 live invariant. If rollout failed, prove the exact original production image is restored and healthy.

- [ ] **Step 3: Inspect complete branch scope and privacy**

```bash
git diff --stat 8ae3a04..HEAD
git diff --check 8ae3a04..HEAD
git status --short --branch
```

Require no audio, transcript, caller data, token, Portainer environment, raw benchmark output, model cache, wheel, or image archive. Expected tracked scope is ASR locks/Dockerfile/Compose/tests, the authorized embedding format fix, STT contract test, design, and plan.

- [ ] **Step 4: Run whole-branch review**

Use `superpowers:requesting-code-review`, including deferred minors from the SDD ledger. Address verified findings through one reviewed fix wave, rerun affected and full gates, and do not add speculative compatibility code.

Do not push, open a PR, merge, or delete the branch unless the user separately requests that delivery workflow.
