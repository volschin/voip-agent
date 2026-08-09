# Spark-vLLM ASR Midpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify a reproducible June 2026 Spark-vLLM Qwen3-ASR candidate that bisects the complete runtime change between production and the rejected August nightly.

**Architecture:** A repository-owned build script checks out one immutable eugr source snapshot, applies one audited build-system patch, and builds vLLM and FlashInfer from exact commits under a historical dependency cutoff. The existing ASR Dockerfile adds only the already hash-locked audio decoders to that local midpoint base. The candidate is then tested in isolation with the exact production model snapshot, first on the 21 discriminating cases and only then, if eligible, on the complete 70-quality plus 12-load A/B.

**Tech Stack:** Bash, Docker BuildKit, ARM64 NVIDIA GB10, CUDA 13.0.2, PyTorch 2.11, vLLM 0.23.0, FlashInfer 0.6.12, Transformers 5.12.1, pytest, Ruff, Qwen3-ASR-0.6B.

## Global Constraints

- eugr source snapshot: `b51af15a280d28c2ad9096b3ef581524eddbd0e7`.
- CUDA base: `nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6`; expected ARM64 manifest: `sha256:450d11555d20ac8ebbbc13ebf17589c2bd42869171a90179ce7098b4a5e64c6a`.
- NCCL: tag `v2.30.3-1`, commit `6da422082f910a8dd230f7e42e26ece4dc37bccc`.
- vLLM: tag `v0.23.0`, commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- FlashInfer: tag `v0.6.12`, commit `d768c14e7cf5dd5df45a8a1de78ae815879f108a`.
- Runtime: Torch `2.11.0+cu130`, torchvision `0.26.0+cu130`, torchaudio `2.11.0+cu130`, Transformers `5.12.1`, and FlashInfer `0.6.12`.
- Dependency cutoff: `2026-06-18T23:59:59Z`.
- Qwen3-ASR adapter SHA-256: `e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d`.
- Model snapshot: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0`.
- Audio extension remains exactly the existing hash-locked SoundFile `0.13.1` and PyAV `17.0.1` lock.
- Do not add a vLLM, encoder, prompt, language, or transcription-response source patch.
- Keep `language=de`; additive response fields remain valid.
- Production must remain the currently observed image `sha256:38f255cd9c0b6bac1e9b1aaa72904c25f7d3e3958ef56cefee9531ff65f2cbe3` with `running|healthy|0|unless-stopped` throughout.
- Do not expose protected audio, transcripts, manifest content, process IDs, or private paths in committed evidence.
- This plan builds and qualifies a candidate only. It does not authorize production rollout.

---

### Task 1: Add the reproducible midpoint build contract

**Files:**
- Create: `dgx/asr/build-midpoint-base.sh`
- Create: `dgx/asr/eugr-midpoint.patch`
- Modify: `dgx/asr/Dockerfile`
- Modify: `tests/test_dgx_deployment.py`
- Modify: `dgx/README.md`

**Interfaces:**
- Consumes: the immutable refs in Global Constraints and the existing audio lock.
- Produces: base tag `dgx-spark-vllm:midpoint-v023` and final tag `dgx-qwen3-asr:spark-midpoint-v023-test`.

- [ ] **Step 1: Write failing deployment tests for the midpoint source build**

Add constants for every immutable ref and tests with these exact assertions:

```python
def test_asr_midpoint_build_pins_complete_historical_stack() -> None:
    script = (ROOT / "dgx/asr/build-midpoint-base.sh").read_text(encoding="utf-8")
    patch = (ROOT / "dgx/asr/eugr-midpoint.patch").read_text(encoding="utf-8")

    for value in (
        "b51af15a280d28c2ad9096b3ef581524eddbd0e7",
        "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "d768c14e7cf5dd5df45a8a1de78ae815879f108a",
        "6da422082f910a8dd230f7e42e26ece4dc37bccc",
        "2026-06-18T23:59:59Z",
        "sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6",
    ):
        assert value in script or value in patch
    assert "transformers==5.12.1" in patch
    assert "VLLM_PRS" not in script
    assert "FLASHINFER_PRS" not in script


def test_asr_midpoint_runtime_asserts_adapter_and_versions() -> None:
    dockerfile = (ROOT / "dgx/asr/Dockerfile").read_text(encoding="utf-8")
    assert "ARG SPARK_BASE=dgx-spark-vllm:midpoint-v023" in dockerfile
    assert "0.23.0" in dockerfile
    assert "2.11.0+cu130" in dockerfile
    assert "5.12.1" in dockerfile
    assert "0.6.12" in dockerfile
    assert "e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d" in dockerfile
```

Update the existing ASR base assertion to accept exactly one `ARG SPARK_BASE`
followed by `FROM ${SPARK_BASE}`. Retain the audio-only install guard and the
no-runtime-compiler guard.

Run:

```bash
pytest -q tests/test_dgx_deployment.py -k 'asr and (midpoint or spark or audio or image)'
```

Expected: fail because the script, patch, and midpoint Dockerfile contract do not yet exist.

- [ ] **Step 2: Create the exact upstream build-system patch**

Create a unified patch against eugr commit `b51af15a...` that makes only these
changes to its `Dockerfile`:

```diff
-ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
+ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
+ENV UV_EXCLUDE_NEWER=2026-06-18T23:59:59Z
-RUN git clone -b v2.30u1 https://github.com/NVIDIA/nccl.git && \
+RUN git clone https://github.com/NVIDIA/nccl.git && \
+    cd nccl && git checkout 6da422082f910a8dd230f7e42e26ece4dc37bccc && cd .. && \
-        echo "transformers>=5.0.0" >> /tmp/wheel-override.txt; \
+        echo "transformers==5.12.1" >> /tmp/wheel-override.txt; \
```

Add `UV_EXCLUDE_NEWER` to both the builder base and runner stages. Do not alter
vLLM model source, eugr model patches, attention code, or runtime command code.

- [ ] **Step 3: Implement the build script**

The script must use this structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

EUGR_COMMIT=b51af15a280d28c2ad9096b3ef581524eddbd0e7
VLLM_COMMIT=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
FLASHINFER_COMMIT=d768c14e7cf5dd5df45a8a1de78ae815879f108a
BASE_TAG=dgx-spark-vllm:midpoint-v023
FINAL_TAG=dgx-qwen3-asr:spark-midpoint-v023-test

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(mktemp -d)
trap 'rm -rf -- "$source_root"' EXIT

git clone --filter=blob:none https://github.com/eugr/spark-vllm-docker.git "$source_root/eugr"
git -C "$source_root/eugr" checkout --detach "$EUGR_COMMIT"
git -C "$source_root/eugr" apply --check "$script_dir/eugr-midpoint.patch"
git -C "$source_root/eugr" apply "$script_dir/eugr-midpoint.patch"

(
  cd "$source_root/eugr"
  ./build-and-copy.sh --tag "$BASE_TAG" --gpu-arch 12.1a --build-jobs 4 \
    --tf5 --rebuild-vllm --vllm-ref "$VLLM_COMMIT" \
    --rebuild-flashinfer --flashinfer-ref "$FLASHINFER_COMMIT"
)

docker build --pull=false --build-arg "SPARK_BASE=$BASE_TAG" \
  --tag "$FINAL_TAG" --file "$script_dir/Dockerfile" "$script_dir/.."
```

After each build, add read-only assertions for the base/final image ID,
architecture, labels/build metadata, installed versions, and adapter SHA. The
runtime inventory must identify the historical eugr snapshot's own source
adjustments separately from repository-added patches; the latter must be empty.
The script must not start, stop, rename, update, or remove a container.

- [ ] **Step 4: Change the ASR derivative to the midpoint base**

Replace the current digest `FROM` with:

```dockerfile
ARG SPARK_BASE=dgx-spark-vllm:midpoint-v023
FROM ${SPARK_BASE}
```

Keep the existing locked audio installation. Replace the runtime assertion with
checks for vLLM `0.23.0`, Torch `2.11.0+cu130`, CUDA `13.0`, Transformers
`5.12.1`, FlashInfer `0.6.12`, and SHA-256 of the imported
`vllm/model_executor/models/qwen3_asr.py`. Keep the image unpatched.

- [ ] **Step 5: Document the one-command candidate build**

Add to `dgx/README.md`:

```bash
cd dgx
./asr/build-midpoint-base.sh
```

State that this produces only the two local test tags, uses the June cutoff,
does not deploy them, and takes a source-build amount of time.

- [ ] **Step 6: Make the contract green and commit**

```bash
bash -n dgx/asr/build-midpoint-base.sh
pytest -q tests/test_dgx_deployment.py
ruff check tests/test_dgx_deployment.py
ruff format --check tests/test_dgx_deployment.py
git diff --check
git add dgx/asr/build-midpoint-base.sh dgx/asr/eugr-midpoint.patch \
  dgx/asr/Dockerfile dgx/README.md tests/test_dgx_deployment.py
git commit -m "build(asr): add reproducible midpoint stack"
```

Expected: all commands pass and the commit contains no generated wheels or images.

---

### Task 2: Build and statically validate the midpoint image on GX10

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/midpoint-build-report.md`
- Modify only if a real build defect requires TDD: files from Task 1

**Interfaces:**
- Consumes: Task 1 build recipe and local GX10 Docker.
- Produces: immutable candidate image ID plus complete transcript-free runtime inventory.

- [ ] **Step 1: Recheck production and candidate namespaces**

```bash
ssh volsch@192.168.68.41 'docker inspect qwen3-asr --format "{{.Image}}|{{.State.Status}}|{{.State.Health.Status}}|{{.RestartCount}}|{{.HostConfig.RestartPolicy.Name}}"'
ssh volsch@192.168.68.41 'test -z "$(docker ps -aq --filter name=^/qwen3-asr-midpoint-test$)"'
```

Expected: exact production invariant and no midpoint test container.

- [ ] **Step 2: Transfer only the clean build inputs**

Use `git archive HEAD dgx/asr` and transfer that archive to the owner-only
directory `/home/volsch/asr-midpoint-build-20260809` on GX10. Do not transfer
`.env`, benchmark records, model files, or unrelated worktree content. Create
the parent with mode `0700` and extract only that archive.

- [ ] **Step 3: Run the source build**

```bash
ssh volsch@192.168.68.41 '/home/volsch/asr-midpoint-build-20260809/dgx/asr/build-midpoint-base.sh'
```

Expected: both tags exist, the script's exact version/hash assertions pass, and
production remains healthy with zero restarts. A real failure is debugged with
`superpowers:systematic-debugging`; any source change starts with a failing
regression test and requires a rebuild.

- [ ] **Step 4: Record static image evidence**

Record only image ID, size, ARM64/Linux, creation time, base/vLLM/FlashInfer/
NCCL commits, cutoff, installed package versions, adapter hash, and absence of
local vLLM patches. Verify the final image is smaller than production.

- [ ] **Step 5: Run repository gates after any build correction**

```bash
pytest -q
ruff check agent tests dgx
ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
```

Expected: all pass; unrelated production services are untouched.

---

### Task 3: Prove exact model, API, and CUDA behavior

**Files:**
- Append outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/midpoint-build-report.md`

**Interfaces:**
- Consumes: Task 2 candidate image and read-only model cache.
- Produces: a healthy isolated candidate at `127.0.0.1:18001` with real GPU evidence.

- [ ] **Step 1: Start only the named isolated candidate**

Run the accepted candidate image with container name `qwen3-asr-midpoint-test`,
loopback mapping `127.0.0.1:18001:8001`, read-only Hugging Face cache, offline
environment, one GPU, 4 GiB shared memory, and the exact Compose command/model
snapshot. Use restart policy `no`.

- [ ] **Step 2: Fail closed on model identity**

Wait for `/v1/models`, require exactly served ID `qwen3-asr`, then inspect the
container command and require the exact `5eb144179...` snapshot path. Record
startup time without model-cache filenames beyond the approved revision.

- [ ] **Step 3: Prove the API and real CUDA path**

Send one protected 16 kHz German WAV with `language=de`; require HTTP 200 and a
JSON object with a string `text`. During the request, require the candidate
container's process to appear as a GPU compute owner and observe aggregate SM
utilization above zero. Do not record the transcript or process ID.

- [ ] **Step 4: Stop on any identity or CUDA mismatch**

If model identity, adapter hash, request success, or CUDA proof fails, remove
only `qwen3-asr-midpoint-test`, preserve the image and logs, and do not run the
quality discriminator.

---

### Task 4: Run the 21-case discriminator

**Files:**
- Create only in the protected GX10 directory: raw result JSON
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/midpoint-quality-report.md`

**Interfaces:**
- Consumes: the healthy midpoint endpoint, sealed corpus, historical production aggregates, and the existing `engine-ab.py` selector.
- Produces: transcript-free counts for 21 cases with selection hash `cb4b76a15c7182376ecb426abfd29af26e9c849d2fb3ddf01e5f09f3490fd957`.

- [ ] **Step 1: Revalidate protected inputs without printing content**

Require the owner-only freeze directory mode `0700`, protected files mode
`0600`, manifest count `70`, exact corpus version, historical result counts
`70`, and the fixed 21-case selection hash.

- [ ] **Step 2: Run one complete midpoint discriminator series**

Use the existing protected script against `http://127.0.0.1:18001`, with
`language=de`, original order, concurrency one, and a new owner-only output.
Require `21/21` successes.

- [ ] **Step 3: Apply the predeclared pass gate**

The candidate must meet or exceed historical production's exact discriminator:

```text
entities: 14/22
numbers:   1/6
times:     1/6
dates:     0/4
```

If any category is lower, classify the midpoint as rejected and skip Task 5.
Do not tune, patch, or selectively rerun it.

- [ ] **Step 4: Record safe evidence**

Record candidate image ID, model revision, adapter hash, selection hash,
completion count, aggregate hit counts, latency aggregates, and CUDA result.
Validate that the report contains no transcript-bearing `records` array.

---

### Task 5: Run the complete paired A/B only after discriminator success

**Files:**
- Create only in the protected GX10 directory: full benchmark raw outputs
- Append outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-image/midpoint-quality-report.md`

**Interfaces:**
- Consumes: successful Task 4 candidate and the previously validated normalized common gateway.
- Produces: one complete 70-quality plus 12-load series for production and midpoint.

- [ ] **Step 1: Build one immutable normalized gateway**

Use the already validated gateway source contract: accept an object with string
`text`, ignore additive fields, emit only normalized text. Build it once and
run the exact same gateway image for both endpoints with equal environment and
security settings except backend URL.

- [ ] **Step 2: Execute the fixed series in order**

Run production quality `70/70`, midpoint quality `70/70`, production load
`12/12`, and midpoint load `12/12`. Use identical bytes, `language=de`, order,
concurrency, and timeout. Require real CUDA activity in both load phases.

- [ ] **Step 3: Apply all existing adoption gates**

Require candidate entity recall and each non-null number/time/date accuracy no
lower than production, no non-speech regression, median latency ratio at most
`1.05`, p90 ratio at most `1.10`, all requests successful, and CUDA true.
Do not delete outliers. Use the one allowed complete replication only for an
isolated shared-host latency anomaly, never for a quality failure.

- [ ] **Step 4: Produce one safe decision**

Write `eligible` or `ineligible`, the exact aggregate metrics, image IDs,
runtime inventory, input hashes, gateway image ID, and production invariant.
No raw text or audio paths may leave the protected directory.

---

### Task 6: Clean up, verify, and commit the outcome

**Files:**
- Modify: `docs/superpowers/specs/2026-08-09-spark-vllm-asr-midpoint-design.md`

**Interfaces:**
- Consumes: final build and quality decision.
- Produces: retained immutable candidate image, no candidate container, unchanged production, and committed non-private outcome.

- [ ] **Step 1: Remove only isolated test containers**

Remove `qwen3-asr-midpoint-test` and the two normalized benchmark gateway
containers by exact name. Retain the midpoint candidate image and its base image.

- [ ] **Step 2: Re-prove production and cleanup**

Require production's exact current image, `running|healthy|0|unless-stopped`,
zero midpoint/gateway test containers, and the retained candidate image ID.

- [ ] **Step 3: Run final repository verification**

```bash
pytest -q
ruff check agent tests dgx
ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
```

Expected: every command passes.

- [ ] **Step 4: Record and commit the outcome**

Append the immutable candidate identity, exact runtime stack, discriminator/full
A/B decision, and production invariant to the midpoint design. Commit only
repository-safe files:

```bash
git add docs/superpowers/specs/2026-08-09-spark-vllm-asr-midpoint-design.md
git commit -m "docs(asr): record midpoint candidate result"
```

Do not deploy, push, open a PR, merge, or delete the retained image without a
separate user instruction.
