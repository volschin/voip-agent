# Spark-vLLM ASR Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current 33.29 GB Qwen3-ASR production image with a smaller digest-pinned `eugr/spark-vllm` derivative while preserving the exact model revision, OpenAI transcription contract, German quality, CUDA execution, and production latency.

**Architecture:** `dgx/asr/Dockerfile` inherits the already validated ARM64 Spark-vLLM image and adds only the two missing audio decoder distributions from a direct-only hash lock. Repository Compose builds that derivative but keeps the existing model path, command, healthcheck, networks, and GPU settings. A separately named loopback-only candidate is built and tested on the GX10 while production stays healthy; the existing common ASR gateway and protected `asr-companion-de-v1` corpus decide quality and latency eligibility before a narrowly scoped Portainer update.

**Tech Stack:** Docker/Compose, ARM64 NVIDIA GB10, CUDA 13.0, PyTorch 2.11, Spark-vLLM, Qwen3-ASR-0.6B, uv, pytest, Ruff, Portainer, the existing ASR benchmark gateway and protected German corpus.

## Fixed contracts

- Base image: `eugr/spark-vllm@sha256:1d861bef8a6c0851140cec2575ebd32342d55bc0fd28ad4c6ca178269e9d1cff`.
- Expected inherited runtime: Python `3.12.3`, Torch `2.11.0+cu130`, `torch.version.cuda == "13.0"`, vLLM `0.26.1rc1.dev468+g6b5bec7be.d20260807`, Triton `3.6.0`, and FlashInfer `0.6.18`.
- Added distributions: only `soundfile==0.13.1` and `av==17.0.1`, installed with `--require-hashes --no-deps`. Inherited `cffi==2.1.1` is reused.
- Do not install or rebuild Torch, vLLM, Triton, FlashInfer, Flash-Attention, CUDA, CFFI, or a compiler unless a real failing gate proves the exact requirement and the plan is revised before continuing.
- Model snapshot: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0`.
- API: multipart WAV plus `language=de` to `POST /v1/audio/transcriptions`; response is exactly a JSON object with a string `text` field.
- Candidate: image `dgx-qwen3-asr:spark-vllm-test`, container `qwen3-asr-spark-test`, host bind `127.0.0.1:18001:8001`.
- Current production invariant: image `sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3`, size `33,294,535,748`, `running|healthy|0|unless-stopped`.
- Portainer stack is `voice`, stack ID `16`, endpoint ID `1`, API origin `http://192.168.68.41:9000/api`.
- Never print, commit, or copy out raw audio, references, transcripts, caller data, Portainer environments, tokens, or model-cache content.
- Until Task 6 passes, do not stop, recreate, update, or replace production `qwen3-asr`.

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
- response object has exactly `text` and that value is a string;
- response and timing stay in an owner-only temporary directory and are never copied to Git or the report.

- [ ] **Step 4: Correlate candidate PID and nonzero SM with the request**

During a second real request, sample `nvidia-smi --query-compute-apps=pid` and `nvidia-smi dmon -s u`. Require at least one PID belonging to `qwen3-asr-spark-test` and maximum aggregate SM utilization greater than zero. Record only booleans, maximum SM percentage, image ID, and request latency.

- [ ] **Step 5: Verify production and clean the isolated container**

```bash
ssh volsch@192.168.68.41 'docker inspect qwen3-asr --format "{{.Image}}|{{.State.Status}}|{{.State.Health.Status}}|{{.RestartCount}}|{{.HostConfig.RestartPolicy.Name}}"; docker rm -f qwen3-asr-spark-test >/dev/null; test "$(docker ps -aq --filter name=^/qwen3-asr-spark-test$ | wc -l)" -eq 0'
```

Expected: production invariant is unchanged; candidate image remains available.

---

### Task 6: Run the production-versus-candidate German quality and latency A/B

**Files:**
- No repository files changed.
- Protected raw artifacts remain only below `/home/volsch/ai-companion/runtime/asr-image-ab-*` on GX10.
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/performance-ab-report.md`

**Interfaces:**
- Consumes: accepted candidate, current production, common gateway source in `/home/volsch/projekte/ai-companion/.worktrees/asr-baseline-first`, and frozen protected corpus root.
- Produces: a transcript-free comparison with explicit quality, success, CUDA, p50, and p90 gates.

- [ ] **Step 1: Create an owner-only run directory and stage only benchmark code**

On GX10 create a unique mode-`0700` directory below `/home/volsch/ai-companion/runtime`, with `umask 077`. Copy only `scripts/asr_benchmark.py`, `scripts/asr_corpus.py`, and `scripts/asr_score.py` from the local `asr-baseline-first` worktree into a mode-`0700` tools subdirectory; each file must be mode `0600`. Use the frozen root and its `.venv/bin/python`; never copy corpus files off GX10.

- [ ] **Step 2: Build and start two identical common gateways**

Build `asr-benchmark-gateway:spark-image-ab` from:

```bash
docker --host ssh://volsch@192.168.68.41 build \
  --file /home/volsch/projekte/ai-companion/.worktrees/asr-baseline-first/deploy/asr-benchmark-gateway.Dockerfile \
  --tag asr-benchmark-gateway:spark-image-ab \
  /home/volsch/projekte/ai-companion/.worktrees/asr-baseline-first
```

Start production gateway `asr-image-ab-gateway-prod` on `voice_default`, backend `http://qwen3-asr:8001`, loopback port `18110`; start candidate gateway `asr-image-ab-gateway-candidate` with the same image/security options, backend `http://qwen3-asr-spark-test:8001`, loopback port `18111`. Both gateways must report `{"status":"ok"}`. Recreate the Task 5 candidate without its host port but on `voice_default`; keep production running.

- [ ] **Step 3: Run identical quality and load phases in paired order**

Using the staged runner and exact frozen files, run once per side:

```text
production quality: 70 cases, concurrency 1, gateway 18110
candidate quality:  70 cases, concurrency 1, gateway 18111
production load:    frozen 12-case subset, concurrency 4, gateway 18110
candidate load:     frozen 12-case subset, concurrency 4, gateway 18111
```

Invoke `scripts/asr_benchmark.py run --phase quality` with the protected root and manifest, and `run --phase load` with the same root/manifest plus `latency-subset-v1.json`. Every output must be created exclusively with mode `0600`; no raw JSON may be printed.

- [ ] **Step 4: Observe CUDA for each side during its load phase**

Correlate each container's PIDs with NVIDIA compute PIDs while its load phase runs and require nonzero SM utilization. Store only `production_cuda_observed=true` and `candidate_cuda_observed=true` plus bounded aggregate SM values in the protected comparison.

- [ ] **Step 5: Produce and validate a transcript-free comparison inside the protected directory**

Use the staged `asr_score.py` against the protected manifest and each 70-record quality run. The comparator must reject duplicate JSON keys, unknown fields, missing/failed requests, non-finite metrics, case-order differences, corpus/hash drift, and any unexpected response shape. It may read transcripts only inside the owner-only run directory. Its mode-`0600` output contains no records, text, transcript, reference, audio path, error body, or raw PID.

Require all of these gates:

- 70/70 quality and 12/12 load requests succeed for each side;
- corpus version and ordered case IDs are identical;
- candidate WER, CER, macro WER, real-path macro WER, empty-speech count, malformed count, non-speech hallucinated words, language-change cases, non-transcript additions, and command-risk cases are no worse than production;
- candidate entity recall and each non-null number/time/date accuracy are no lower than production;
- candidate load p50 end-to-end latency divided by production p50 is at most `1.05`;
- candidate load nearest-rank p90 divided by production p90 is at most `1.10`;
- both CUDA observations are true.

The safe output must include exact aggregate maps, finite numeric validation, corpus version, hashes of base image, candidate image, model snapshot identifier, runner files, manifest, subset, and gateway image. Validate the output a second time from disk before interpreting it.

- [ ] **Step 6: Apply the single permitted replication rule only if needed**

If and only if one paired series fails solely because one isolated shared-host latency anomaly affects both sides, preserve series 1 and run one complete unchanged replication. Report series 1, series 2, and combined 24-sample load aggregates. Do not delete or selectively rerun cases. A quality, success, CUDA, or repeated latency failure rejects rollout.

- [ ] **Step 7: Clean benchmark containers and report only safe evidence**

Remove only `qwen3-asr-spark-test`, `asr-image-ab-gateway-prod`, and `asr-image-ab-gateway-candidate`. Recheck production `running|healthy|0|unless-stopped`. The report may contain image IDs, sizes, corpus version/hash, aggregate quality metrics, p50/p90 values and ratios, CUDA booleans, and pass/fail; it must contain no raw records or transcripts.

---

### Task 7: Roll out the accepted image with exact rollback protection

**Files:**
- No planned repository changes.
- Protected backups and environments remain in a mode-`0700` local run directory.
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/task-7-report.md`

**Interfaces:**
- Consumes: all-green Task 4-6 evidence and current live stack identity.
- Produces: updated healthy ASR production or automatic restoration of the exact pre-update stack.

- [ ] **Step 1: Re-resolve all identities immediately before mutation**

Require stack `voice`, ID `16`, endpoint `1`; production container healthy with stable zero restart count for 30 seconds; candidate image ID equal to the A/B image; and a clean repository at the accepted commit. If production image changed since Task 6, stop and reassess instead of applying the older candidate over newer production.

- [ ] **Step 2: Create immutable rollback and release tags**

```bash
release_tag="dgx-qwen3-asr:spark-vllm-$(git rev-parse --short=12 HEAD)"
ssh volsch@192.168.68.41 'docker image tag sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3 dgx-qwen3-asr:rollback-38f255cd'
docker --host ssh://volsch@192.168.68.41 image tag dgx-qwen3-asr:spark-vllm-test "$release_tag"
```

If the current production ID differs from the recorded baseline, derive the rollback tag from the newly verified ID and record it; never retag the older image as current production.

- [ ] **Step 3: Back up exact Portainer Compose and environment**

Use `/home/volsch/projekte/ai-companion/.worktrees/asr-baseline-first/scripts/portainer.py` with:

```text
GX10_PORTAINER_TOKEN_FILE=/home/volsch/.gx10_portainer_token
--base-url http://192.168.68.41:9000/api
--endpoint-id 1
backup --name voice
```

Write Compose and environment to new mode-`0600` files and redirect only the helper's redacted metadata stdout to a separate mode-`0600` file. Never print or source the environment.

- [ ] **Step 4: Create a one-line candidate stack revision and prove its diff**

From the exact backup, replace only the current `qwen3-asr` image scalar with the immutable release tag. Parse both YAML documents and assert every value except `services.qwen3-asr.image` is deeply equal. Also assert the textual diff contains exactly one removed image line and one added image line.

- [ ] **Step 5: Apply the exact stack update without pull or prune**

Run the same Portainer helper with `upsert --name voice --compose "$run_dir/voice-candidate.yml" --environment "$run_dir/voice-environment.json" --expected-stack-id 16 --apply`. The helper sets `Prune=false`; the image is already local, so no registry pull is required.

- [ ] **Step 6: Run production acceptance**

Within 300 seconds require:

- `qwen3-asr` uses the accepted candidate image ID and immutable release tag;
- `running|healthy|0|unless-stopped` and remains stable for 30 seconds;
- `/v1/models` reports `qwen3-asr`;
- one private real German 16 kHz WAV from `voice_default` receives HTTP 200 and exact `{text: string}` response shape;
- the production ASR container's PID is observed on CUDA and aggregate SM exceeds zero during that request;
- `dgx-qwen3-asr:rollback-...` still resolves to the exact pre-update image.

- [ ] **Step 7: Roll back automatically on any failed gate**

Use the same Portainer helper to upsert the exact backup Compose plus exact backup environment with stack ID `16`. Then require the pre-update image ID, `running|healthy|0|unless-stopped`, `/v1/models`, a real transcription, and real CUDA proof. Report the candidate as rejected; do not attempt a second rollout in the same task.

---

### Task 8: Run final verification, self-review, and commit the plan record

**Files:**
- Modify only if live evidence invalidated documentation: `docs/superpowers/specs/2026-08-09-spark-vllm-asr-image-design.md`
- Already created: `docs/superpowers/plans/2026-08-09-spark-vllm-asr-image.md`

**Interfaces:**
- Consumes: final implementation and live acceptance result.
- Produces: a clean, reviewable feature branch with exact evidence and no private artifacts.

- [ ] **Step 1: Run fresh complete repository verification**

```bash
pytest -q
ruff check agent tests dgx
ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
```

Expected: all commands pass from fresh processes.

- [ ] **Step 2: Verify the final image and live state one last time**

Re-run Task 4's image identity/platform/import checks and Task 7's live invariant. If rollout was rejected, prove the original production image is restored and healthy; never describe a rejected candidate as deployed.

- [ ] **Step 3: Inspect the entire branch diff for privacy and scope**

```bash
git diff --stat 8ae3a04..HEAD
git diff --check 8ae3a04..HEAD
git status --short --branch
```

Review every changed file. Require no audio, transcript, caller data, token, Portainer environment, raw benchmark output, model cache, generated wheel, or image archive. The expected implementation surface is only ASR locks, ASR Dockerfile, ASR Compose ownership, deployment tests, design, and this plan.

- [ ] **Step 4: Request code review and address only verified findings**

Use `superpowers:requesting-code-review`. For any finding, use `superpowers:receiving-code-review`, reproduce the issue, add a failing test when behavior changes, and rerun the affected plus full gates. Do not broaden the image for speculative compatibility.

- [ ] **Step 5: Commit this implementation plan before execution if not already committed**

```bash
git add docs/superpowers/plans/2026-08-09-spark-vllm-asr-image.md
git commit -m "docs(asr): plan Spark vLLM image rollout"
```

Do not push, open a PR, merge, or delete the branch unless the user separately requests that delivery workflow.
