# Slim CUDA 13.3.1 TTS Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a materially smaller ARM64 Qwen3-TTS candidate image on CUDA 13.3.1 and Ubuntu 26.04 without changing production.

**Architecture:** A CUDA 13.3.1 development stage creates a self-contained Python 3.14 virtual environment and compiles Flash-Attention for SM120. A CUDA 13.3.1 base stage receives only that environment, minimal runtime libraries, and the existing TTS application; model and private profile data stay read-only mounts.

**Tech Stack:** Docker multi-stage builds, NVIDIA CUDA 13.3.1, Ubuntu 26.04, ARM64, Python 3.14, PyTorch 2.13.0 cu132, Flash-Attention 2.8.3, FastAPI, Qwen3-TTS, pytest, Ruff.

## Global Constraints

- Builder image: `nvidia/cuda:13.3.1-devel-ubuntu26.04@sha256:da3989b0ea8e8b4b241711edd5823bc1cc83d05a01882258bddad84d7394c37e`.
- Runtime image: `nvidia/cuda:13.3.1-base-ubuntu26.04@sha256:f65b4f0b65bbf2e0a2520cebaec3120bf4ed110aecc3e7dcab3b11cb508a0484`.
- Runtime Python must be 3.14; PyTorch must be `2.13.0+cu132`; `torch.version.cuda` must be `13.2`; TorchAudio must be absent.
- Compile `flash-attn==2.8.3` with `MAX_JOBS=4` and `FLASH_ATTN_CUDA_ARCHS=120` in the builder only.
- Preserve model revision `fd4b254389122332181a7c3db7f27e918eec64e3`, API contracts, private WAV profile, health checks, offline operation, and cooperative cancellation.
- Do not add vLLM, Ray, FlashInfer, a CPU fallback, ICL support, or unrelated dependency upgrades.
- Do not stop, recreate, update, or replace production `qwen3-tts`; use candidate name `qwen3-tts-cuda1331-test`, loopback port `18002`, and image tag `dgx-qwen3-tts:cuda1331-cu132-test`.
- Remove the candidate container after validation. Keep the candidate image for inspection unless the user requests its removal.

---

### Task 1: Create the isolated worktree and prove the baseline

**Files:**
- No repository files changed.

**Interfaces:**
- Consumes: clean `main` containing design commit `eb072a7`.
- Produces: isolated worktree `.worktrees/slim-cuda1331-tts` on branch `perf/slim-cuda1331-tts` with passing baseline checks.

- [ ] **Step 1: Confirm isolation prerequisites**

```bash
cd /home/volsch/projekte/voip-agent
test -z "$(git status --porcelain=v1)"
git check-ignore -q .worktrees
git rev-parse --verify eb072a7^{commit}
```

- [ ] **Step 2: Create and enter the worktree**

```bash
git worktree add .worktrees/slim-cuda1331-tts -b perf/slim-cuda1331-tts
cd /home/volsch/projekte/voip-agent/.worktrees/slim-cuda1331-tts
```

- [ ] **Step 3: Run the baseline unit and static checks**

```bash
pytest -q
ruff check agent tests dgx/tts
ruff format --check agent tests dgx/tts
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
```

Expected: all commands exit 0. If an existing baseline command fails, stop and diagnose before changing files.

### Task 2: Define complete slim-image dependency locks

**Files:**
- Create: `dgx/tts/requirements-slim-arm64.in`
- Create: `dgx/tts/requirements-slim-arm64.lock`
- Create: `dgx/tts/tts-packages-arm64.lock`
- Create: `dgx/tts/flash-attn-arm64.lock`
- Create: `dgx/tts/apt-builder-packages-arm64.lock`
- Create: `dgx/tts/apt-runtime-packages-arm64.lock`
- Delete: `dgx/tts/requirements-arm64.lock`
- Delete: `dgx/tts/apt-packages-arm64.lock`

**Interfaces:**
- Consumes: package versions from the current production lock plus PyTorch 2.13.0 cu132.
- Produces: exact, hash-locked Python runtime requirements and exact Ubuntu 26.04 package lists consumed by Task 3.

- [ ] **Step 1: Demonstrate that the slim dependency contract is absent**

```bash
test ! -e dgx/tts/requirements-slim-arm64.in
test ! -e dgx/tts/requirements-slim-arm64.lock
test ! -e dgx/tts/tts-packages-arm64.lock
test ! -e dgx/tts/flash-attn-arm64.lock
```

Expected: all four checks exit 0, proving the new contract has not already been implemented.

- [ ] **Step 2: Create the direct dependency input**

Create `dgx/tts/requirements-slim-arm64.in` with these exact direct requirements:

```text
--extra-index-url https://download.pytorch.org/whl/cu132

accelerate==1.12.0
einops==0.8.2
fastapi==0.135.3
huggingface-hub==0.36.2
librosa==0.11.0
onnxruntime==1.28.0
pydantic==2.12.5
soundfile==0.14.0
sox==1.5.0
torch==2.13.0+cu132
transformers==4.57.3
uvicorn[standard]==0.44.0
```

Do not include Qwen-TTS, Faster-Qwen3-TTS, or Flash-Attention here. Qwen-TTS
declares unused Gradio and TorchAudio dependencies, while Flash-Attention needs
PyTorch installed before its build metadata can be prepared.

- [ ] **Step 3: Resolve the complete Python 3.14 ARM64 lock with hashes**

```bash
uvx --from uv uv pip compile \
  --python-version 3.14 \
  --python-platform aarch64-manylinux_2_39 \
  --index-strategy unsafe-best-match \
  --generate-hashes \
  --output-file dgx/tts/requirements-slim-arm64.lock \
  dgx/tts/requirements-slim-arm64.in
```

Expected: the lock contains `torch==2.13.0+cu132`, every transitive dependency of the explicit runtime set, and one or more SHA-256 hashes per distribution.

- [ ] **Step 4: Create the no-dependency Qwen runtime lock**

Create `dgx/tts/tts-packages-arm64.lock` with the hashes already verified by production:

```text
faster-qwen3-tts==0.2.6 --hash=sha256:3881a41dc189f0a6e93fa047f376deffeb2fa84e888e7d570f79b3e2267765cc
qwen-tts==0.1.1 --hash=sha256:11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d
```

The Docker build installs this lock with `--no-deps`; the explicit runtime lock supplies every exercised dependency.

- [ ] **Step 5: Create the isolated Flash-Attention lock**

Create `dgx/tts/flash-attn-arm64.lock` with the verified source-distribution hash already used by production:

```text
flash-attn==2.8.3 --hash=sha256:1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d
```

- [ ] **Step 6: Resolve exact Ubuntu 26.04 builder and runtime package versions**

Run both queries against the pinned images:

```bash
ssh volsch@192.168.68.41 'docker run --rm --entrypoint sh \
  nvidia/cuda:13.3.1-devel-ubuntu26.04@sha256:da3989b0ea8e8b4b241711edd5823bc1cc83d05a01882258bddad84d7394c37e \
  -lc "apt-get update >/dev/null && apt-cache policy python3 python3-venv python3-dev build-essential ninja-build patch"'

ssh volsch@192.168.68.41 'docker run --rm --entrypoint sh \
  nvidia/cuda:13.3.1-base-ubuntu26.04@sha256:f65b4f0b65bbf2e0a2520cebaec3120bf4ed110aecc3e7dcab3b11cb508a0484 \
  -lc "apt-get update >/dev/null && apt-cache policy python3-minimal libsndfile1 libgomp1 sox libsox-fmt-base"'
```

Create the two lock files as `package=CandidateVersion` lines using exactly the reported candidate versions. Builder packages are `python3`, `python3-venv`, `python3-dev`, `build-essential`, `ninja-build`, and `patch`. Runtime packages are `python3-minimal`, `libsndfile1`, `libgomp1`, `sox`, and `libsox-fmt-base`.

- [ ] **Step 7: Verify the generated lock contract**

```bash
rg -n '^torch==2\.13\.0\+cu132( |$)' dgx/tts/requirements-slim-arm64.lock
rg -n '^faster-qwen3-tts==0\.2\.6 ' dgx/tts/tts-packages-arm64.lock
rg -n '^qwen-tts==0\.1\.1 ' dgx/tts/tts-packages-arm64.lock
rg -n '^flash-attn==2\.8\.3 --hash=sha256:1e71dd64' dgx/tts/flash-attn-arm64.lock
test "$(rg -c '^[a-zA-Z0-9][^=]*=[^=]' dgx/tts/apt-builder-packages-arm64.lock)" -eq 6
test "$(rg -c '^[a-zA-Z0-9][^=]*=[^=]' dgx/tts/apt-runtime-packages-arm64.lock)" -eq 5
! rg -n '^(vllm|ray|flashinfer|gradio|hf-gradio|torchaudio)==' dgx/tts/requirements-slim-arm64.lock
```

Expected: every command exits 0 and excluded packages are absent. The build-time Qwen import and real-model acceptance are the authoritative checks for the explicit no-dependency installation.

- [ ] **Step 8: Commit the dependency contract**

```bash
git add dgx/tts/requirements-slim-arm64.in \
  dgx/tts/requirements-slim-arm64.lock \
  dgx/tts/tts-packages-arm64.lock \
  dgx/tts/flash-attn-arm64.lock \
  dgx/tts/apt-builder-packages-arm64.lock \
  dgx/tts/apt-runtime-packages-arm64.lock \
  dgx/tts/requirements-arm64.lock \
  dgx/tts/apt-packages-arm64.lock
git commit -m "build(tts): lock slim CUDA runtime dependencies"
```

### Task 3: Replace the vLLM-derived Dockerfile with the slim multi-stage image

**Files:**
- Modify: `dgx/tts/Dockerfile`

**Interfaces:**
- Consumes: the six lock/input files from Task 2 and existing cancellation patch plus verification script.
- Produces: final runtime image with `/opt/tts-venv/bin/python3` and unchanged `python3 -m dgx.tts.server` entrypoint behavior.

- [ ] **Step 1: Run the static contract check and observe failure**

```bash
rg -n '13\.3\.1-devel-ubuntu26\.04@sha256:da3989b0' dgx/tts/Dockerfile
```

Expected: exit 1 because the current Dockerfile still uses the vLLM-derived base.

- [ ] **Step 2: Implement the builder stage**

Replace the current single-stage header and installation with:

```dockerfile
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.3.1-devel-ubuntu26.04@sha256:da3989b0ea8e8b4b241711edd5823bc1cc83d05a01882258bddad84d7394c37e
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.3.1-base-ubuntu26.04@sha256:f65b4f0b65bbf2e0a2520cebaec3120bf4ed110aecc3e7dcab3b11cb508a0484

FROM ${CUDA_DEVEL_IMAGE} AS builder

COPY tts/apt-builder-packages-arm64.lock /tmp/apt-builder-packages-arm64.lock
RUN apt-get update \
 && xargs -r apt-get install -y --no-install-recommends \
      < /tmp/apt-builder-packages-arm64.lock \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/tts-venv
ENV PATH=/opt/tts-venv/bin:$PATH \
    MAX_JOBS=4 \
    FLASH_ATTN_CUDA_ARCHS=120

COPY tts/requirements-slim-arm64.lock /tmp/requirements-slim-arm64.lock
RUN pip install --no-cache-dir --require-hashes \
      -r /tmp/requirements-slim-arm64.lock

COPY tts/tts-packages-arm64.lock /tmp/tts-packages-arm64.lock
RUN pip install --no-cache-dir --require-hashes --no-deps \
      -r /tmp/tts-packages-arm64.lock

COPY tts/flash-attn-arm64.lock /tmp/flash-attn-arm64.lock
RUN pip install --no-cache-dir --require-hashes --no-deps --no-build-isolation \
      -r /tmp/flash-attn-arm64.lock

RUN python3 -c 'import faster_qwen3_tts, qwen_tts; import qwen_tts.core; import qwen_tts.inference.qwen3_tts_model'
```

Retain the existing cancellation patch application immediately after dependency installation, using `/opt/tts-venv/bin/python3` through `PATH`.

- [ ] **Step 3: Implement the runtime stage**

Append the runtime stage and existing application copies:

```dockerfile
FROM ${CUDA_RUNTIME_IMAGE} AS runtime

COPY tts/apt-runtime-packages-arm64.lock /tmp/apt-runtime-packages-arm64.lock
RUN apt-get update \
 && xargs -r apt-get install -y --no-install-recommends \
      < /tmp/apt-runtime-packages-arm64.lock \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/tts-venv /opt/tts-venv
ENV PATH=/opt/tts-venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY tts/server.py /app/dgx/tts/server.py
COPY tts/runtime.py /app/dgx/tts/runtime.py
COPY tts/profiles.py /app/dgx/tts/profiles.py
COPY tts/clone_runtime.py /app/dgx/tts/clone_runtime.py
COPY tts/api.py /app/dgx/tts/api.py

EXPOSE 8002
CMD ["python3", "-m", "dgx.tts.server"]
```

- [ ] **Step 4: Verify the Dockerfile contract turns green**

```bash
rg -n '13\.3\.1-devel-ubuntu26\.04@sha256:da3989b0' dgx/tts/Dockerfile
rg -n '13\.3\.1-base-ubuntu26\.04@sha256:f65b4f0b' dgx/tts/Dockerfile
test "$(rg -c '^FROM ' dgx/tts/Dockerfile)" -eq 2
! rg -n 'vllm-aeon|FROM .*vllm' dgx/tts/Dockerfile
rg -n 'FLASH_ATTN_CUDA_ARCHS=120' dgx/tts/Dockerfile
git diff --check
```

Expected: every positive check finds one contract line, both negative checks remain silent, and all commands exit 0.

- [ ] **Step 5: Commit the image definition**

```bash
git add dgx/tts/Dockerfile
git commit -m "build(tts): use slim CUDA 13.3.1 image"
```

### Task 4: Run repository validation before the expensive build

**Files:**
- No new files.

**Interfaces:**
- Consumes: committed dependency and Dockerfile changes.
- Produces: fresh local evidence that application behavior and Compose configuration remain unchanged.

- [ ] **Step 1: Run focused TTS tests**

```bash
pytest tests/test_tts_profiles.py tests/test_tts_clone_runtime.py tests/test_tts_api.py -q
```

- [ ] **Step 2: Run the complete unit suite**

```bash
pytest -q
```

- [ ] **Step 3: Run lint and formatting checks**

```bash
ruff check agent tests dgx/tts
ruff format --check agent tests dgx/tts
git diff --check
```

- [ ] **Step 4: Render Compose without changing live state**

```bash
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
```

Expected: all validation commands exit 0.

### Task 5: Build and inspect the ARM64 candidate on the GX10

**Files:**
- No repository files changed unless build evidence exposes a reproducibility defect; such a defect starts a new Red/Green cycle in Tasks 2 or 3.

**Interfaces:**
- Consumes: Docker build context `dgx/` from the isolated worktree.
- Produces: local GX10 image `dgx-qwen3-tts:cuda1331-cu132-test` and static runtime evidence.

- [ ] **Step 1: Record production invariants before the build**

```bash
ssh volsch@192.168.68.41 'docker inspect --format \
  "{{.Image}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}|{{.RestartCount}}|{{.HostConfig.RestartPolicy.Name}}" \
  qwen3-tts'
```

Save the exact line for final comparison; do not print or read private profile text.

- [ ] **Step 2: Build the candidate through the remote Docker daemon**

```bash
docker --host ssh://volsch@192.168.68.41 build \
  --progress=plain \
  --file dgx/tts/Dockerfile \
  --tag dgx-qwen3-tts:cuda1331-cu132-test \
  dgx
```

Expected: exit 0, including successful Flash-Attention build and cancellation-patch verification.

- [ ] **Step 3: Inspect image identity and size**

```bash
docker --host ssh://volsch@192.168.68.41 image inspect \
  --format '{{.Id}}|{{.Size}}|{{.Architecture}}|{{.Os}}' \
  dgx-qwen3-tts:cuda1331-cu132-test
```

Expected: `arm64|linux` and size below `33713982655` bytes.

- [ ] **Step 4: Prove static runtime composition**

```bash
docker --host ssh://volsch@192.168.68.41 run --rm \
  --entrypoint sh dgx-qwen3-tts:cuda1331-cu132-test -lc '
    . /etc/os-release
    test "$VERSION_ID" = 26.04
    test "$(python3 -c "import sys; print(f'\''{sys.version_info.major}.{sys.version_info.minor}'\'')")" = 3.14
    ! command -v nvcc
    python3 -c "import importlib.util; assert importlib.util.find_spec('\''vllm'\'') is None; assert importlib.util.find_spec('\''ray'\'') is None; assert importlib.util.find_spec('\''flashinfer'\'') is None; assert importlib.util.find_spec('\''torchaudio'\'') is None"
    python3 -c "import dgx.tts.api, dgx.tts.clone_runtime, dgx.tts.profiles, faster_qwen3_tts, flash_attn, qwen_tts, torch"
  '
```

Expected: exit 0; no compiler and no excluded inference frameworks.

- [ ] **Step 5: Prove real CUDA and extension loading**

```bash
docker --host ssh://volsch@192.168.68.41 run --rm --gpus all \
  --entrypoint python3 dgx-qwen3-tts:cuda1331-cu132-test -c '
import json, torch
import flash_attn_2_cuda
assert torch.__version__.startswith("2.13.0+")
assert torch.version.cuda == "13.2"
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GB10"
x = torch.arange(1024, device="cuda", dtype=torch.float32)
assert float((x * x).sum().cpu()) > 0
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}))
'
```

Expected: exit 0 with CUDA `13.2` and device `NVIDIA GB10`.

### Task 6: Run isolated real-model acceptance and restore the test surface

**Files:**
- No repository files changed.

**Interfaces:**
- Consumes: candidate image, pinned model cache, private production WAV profile, loopback port 18002.
- Produces: health, real WAV, GPU, cancellation/recovery evidence; candidate container removed; production invariant unchanged.

- [ ] **Step 1: Start the isolated candidate without touching production**

```bash
ssh volsch@192.168.68.41 'docker run -d \
  --name qwen3-tts-cuda1331-test \
  --gpus all \
  --shm-size 4g \
  -p 127.0.0.1:18002:8002 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e QWEN_TTS_MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3 \
  -e QWEN_TTS_DEFAULT_PROFILE=shared-female-de-v1 \
  -v /home/volsch/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /home/volsch/voice-private/profiles:/run/voice-profiles:ro \
  dgx-qwen3-tts:cuda1331-cu132-test'
```

- [ ] **Step 2: Wait for fail-closed health and validate metadata**

```bash
ssh volsch@192.168.68.41 'for i in $(seq 1 150); do
  status=$(docker inspect --format "{{.State.Status}}" qwen3-tts-cuda1331-test)
  if [ "$status" = exited ] || [ "$status" = dead ]; then docker logs --tail 120 qwen3-tts-cuda1331-test >&2; exit 1; fi
  if curl -fsS http://127.0.0.1:18002/health | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['\''status'\'']=='\''ok'\''; assert d['\''model_revision'\'']=='\''fd4b254389122332181a7c3db7f27e918eec64e3'\''; assert d['\''default_profile'\'']=='\''shared-female-de-v1'\''"; then exit 0; fi
  sleep 2
done
docker logs --tail 120 qwen3-tts-cuda1331-test >&2
exit 1'
```

- [ ] **Step 3: Execute a real WAV request and validate the wire format**

```bash
ssh volsch@192.168.68.41 'python3 - <<'\''PY'\''
import io, json, urllib.request, wave
payload = json.dumps({
    "model": "qwen3-tts",
    "input": "Guten Tag. Dies ist ein Test des schlanken Sprachcontainers.",
    "voice": "shared-female-de-v1",
    "response_format": "wav",
}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:18002/v1/audio/speech",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    audio = response.read()
assert response.status == 200
with wave.open(io.BytesIO(audio), "rb") as wav:
    assert wav.getframerate() == 24000
    assert wav.getnchannels() == 1
    assert wav.getsampwidth() == 2
    assert wav.getnframes() > 0
print(json.dumps({"bytes": len(audio), "sample_rate": 24000, "channels": 1}))
PY'
```

- [ ] **Step 4: Prove GPU activity during a real request**

Run a second long request while polling only the candidate PID and aggregate SM utilization:

```bash
ssh volsch@192.168.68.41 'set -euo pipefail
candidate_pid=$(docker inspect --format "{{.State.Pid}}" qwen3-tts-cuda1331-test)
curl -fsS --output /tmp/qwen3-tts-cuda1331-gpu.wav \
  -H "Content-Type: application/json" \
  -d '\''{"model":"qwen3-tts","input":"Guten Tag. Diese längere Messung bestätigt, dass der neue Sprachcontainer die Grafikkarte tatsächlich für die Spracherzeugung verwendet.","voice":"shared-female-de-v1","response_format":"wav"}'\'' \
  http://127.0.0.1:18002/v1/audio/speech &
request_pid=$!
seen_candidate=0
max_sm=0
while kill -0 "$request_pid" 2>/dev/null; do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -qx "$candidate_pid"; then
    seen_candidate=1
  fi
  sample=$(nvidia-smi dmon -s u -c 1 | awk '\''$1 !~ /^#/ && $2 ~ /^[0-9]+$/ {print $2; exit}'\'')
  if [ -n "$sample" ] && [ "$sample" -gt "$max_sm" ]; then max_sm=$sample; fi
  sleep 0.2
done
wait "$request_pid"
test "$seen_candidate" -eq 1
test "$max_sm" -gt 0
printf "candidate_compute_pid_seen=yes|max_sm_percent=%s\n" "$max_sm"
unlink /tmp/qwen3-tts-cuda1331-gpu.wav'
```

Expected: candidate PID seen and `max_sm_percent` greater than zero; no unrelated process IDs are printed.

- [ ] **Step 5: Verify cancellation and immediate recovery**

Start a long stream, require the client timeout, inspect only new candidate logs for cancellation, and issue an immediate recovery request:

```bash
ssh volsch@192.168.68.41 'set -euo pipefail
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
curl -fsS --max-time 1.4 --output /dev/null \
  -H "Content-Type: application/json" \
  -d '\''{"model":"qwen3-tts","input":"Dies ist ein absichtlich langer Satz für den Abbruchtest, der nicht vollständig erzeugt werden darf, weil der Client die Verbindung frühzeitig beendet.","voice":"shared-female-de-v1"}'\'' \
  http://127.0.0.1:18002/v1/audio/speech/stream
curl_rc=$?
set -e
test "$curl_rc" -eq 28
sleep 1
docker logs --since "$since" qwen3-tts-cuda1331-test 2>&1 | grep -Eiq "cancel|abort"
python3 - <<'\''PY'\''
import io, json, urllib.request, wave
payload = json.dumps({
    "model": "qwen3-tts",
    "input": "Die Sprachausgabe ist nach dem Abbruch wieder bereit.",
    "voice": "shared-female-de-v1",
    "response_format": "wav",
}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:18002/v1/audio/speech",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    audio = response.read()
assert response.status == 200
with wave.open(io.BytesIO(audio), "rb") as wav:
    assert wav.getframerate() == 24000
    assert wav.getnchannels() == 1
    assert wav.getsampwidth() == 2
    assert wav.getnframes() > 0
print(json.dumps({"recovery": "ok", "bytes": len(audio)}))
PY'
```

Expected: curl exit 28, a cancellation/abort marker, and a valid recovery WAV.

- [ ] **Step 6: Remove only the candidate container**

```bash
ssh volsch@192.168.68.41 'docker stop --timeout 30 qwen3-tts-cuda1331-test >/dev/null && docker rm qwen3-tts-cuda1331-test >/dev/null'
```

- [ ] **Step 7: Re-check production invariants**

```bash
ssh volsch@192.168.68.41 'docker inspect --format \
  "{{.Image}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}|{{.RestartCount}}|{{.HostConfig.RestartPolicy.Name}}" \
  qwen3-tts'
```

Expected: the exact pre-build image ID and restart policy, `running|healthy`, restart count unchanged, and no `qwen3-tts-cuda1331-test` container.

### Task 7: Final verification and evidence handoff

**Files:**
- Modify only if evidence changes documented behavior: `docs/superpowers/specs/2026-08-08-slim-cuda1331-tts-image-design.md`

**Interfaces:**
- Consumes: all repository and GX10 evidence from Tasks 4-6.
- Produces: clean candidate branch with verified build definition and concise adoption evidence; no production deployment.

- [ ] **Step 1: Run all repository gates freshly**

```bash
pytest -q
ruff check agent tests dgx/tts
ruff format --check agent tests dgx/tts
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
```

- [ ] **Step 2: Verify branch and artifact state**

```bash
git status --short
git log --oneline --decorate main..HEAD
docker --host ssh://volsch@192.168.68.41 image inspect \
  --format '{{.Id}}|{{.Size}}|{{.Architecture}}|{{.Os}}' \
  dgx-qwen3-tts:cuda1331-cu132-test
ssh volsch@192.168.68.41 '! docker inspect qwen3-tts-cuda1331-test >/dev/null 2>&1'
```

- [ ] **Step 3: Commit any evidence-driven documentation correction**

If the build proves a documented version or contract wrong, update only that statement and commit it:

```bash
git add docs/superpowers/specs/2026-08-08-slim-cuda1331-tts-image-design.md
git commit -m "docs(tts): align slim image design with build evidence"
```

Skip this commit when the specification remains accurate.

- [ ] **Step 4: Report without deploying**

Report the candidate image ID and size, current-versus-candidate reduction, exact Python/PyTorch/CUDA versions, Flash-Attention/CUDA proof, health and real-WAV result, cancellation/recovery outcome, production invariant, tests run, branch commits, and any unresolved adoption risk. Explicitly state that production was not switched.
