# Spark-vLLM ASR v0.23.0 Midpoint Reactivation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the exact completed vLLM `0.23.0` midpoint base, finish its ASR derivative, and qualify it once against the frozen ten-case non-speech gate with the production UrocyonF 1.7B model.

**Architecture:** Restore only the previously reviewed midpoint recipe hunks, add an explicit fail-closed `--reuse-base` path bound to the recovered image ID, and build the lightweight audio derivative without repeating the native base build. Run that immutable final image behind the existing normalized gateway, prove real CUDA, and stop after the ten-case gate regardless of pass or fail.

**Tech Stack:** Bash, Docker BuildKit, ARM64 NVIDIA GB10, CUDA 13.0, PyTorch 2.11, vLLM 0.23.0, FlashInfer 0.6.12, Transformers 5.12.1, UrocyonF/Qwen3-ASR-1.7B-NVFP4, pytest, Ruff.

## Global Constraints

- Reusable base tag: `dgx-spark-vllm:midpoint-v023`.
- Reusable base image ID: `sha256:bfa7bcd7c70829e44cd919f22fc68a028816681abe8d4f3b4a2b1ba81e47c134`.
- Final candidate tag: `dgx-qwen3-asr:spark-midpoint-v023-test`.
- eugr source: `b51af15a280d28c2ad9096b3ef581524eddbd0e7`.
- NCCL source: `6da422082f910a8dd230f7e42e26ece4dc37bccc`.
- vLLM source: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`; installed version exactly `0.23.0`.
- FlashInfer source: `d768c14e7cf5dd5df45a8a1de78ae815879f108a`; installed version `0.6.12`.
- Torch: `2.11.0+cu130`; Transformers: `5.12.1`; runtime CUDA: `13.0`.
- Qwen3-ASR adapter SHA-256: `e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d`.
- Final derivative adds only the existing hash-locked SoundFile `0.13.1` and PyAV `17.0.1` packages.
- Model repository: `UrocyonF/Qwen3-ASR-1.7B-NVFP4`; revision `61ad4d533c64e033a750b66c44aad6f18634997e`.
- Served name `qwen3-asr`; `language=de`; `gpu-memory-utilization=0.08`; `max-model-len=8192`; `max-num-seqs=4`.
- Do not modify `dgx/docker-compose.yml`; it already holds the current UrocyonF 1.7B production contract.
- Do not add vLLM, encoder, prompt, language, response, VAD, decoding, or transcript-filter patches.
- Production stays on its current image and command as `running|healthy|0|unless-stopped` with zero restarts.
- Do not commit or report protected paths, audio, raw transcripts, references, PIDs, or raw GPU rows.
- Run the ten non-speech cases once. Do not run the complete 70/12 A/B under this plan.
- No build or test result authorizes rollout.

---

### Task 1: Restore the reproducible midpoint recipe with exact base reuse

**Files:**
- Create: `dgx/asr/build-midpoint-base.sh`
- Create: `dgx/asr/eugr-midpoint.patch`
- Modify: `dgx/asr/Dockerfile`
- Modify: `dgx/README.md`
- Modify: `tests/test_dgx_deployment.py`

**Interfaces:**
- Consumes: exact pre-removal recipe blobs from `cf6f716^`, current audio lock, and recovered base ID.
- Produces: executable `dgx/asr/build-midpoint-base.sh --reuse-base` that rejects every non-exact base and builds `dgx-qwen3-asr:spark-midpoint-v023-test` without a native base rebuild.

- [ ] **Step 1: Write the failing reactivation tests**

Restore the midpoint deployment-contract tests from `cf6f716^` by applying only
their deleted midpoint hunks to the current `tests/test_dgx_deployment.py`.
Preserve every test added after `cf6f716`, and add `import subprocess` beside
the existing standard-library imports. Add these focused tests:

```python
def test_asr_midpoint_reuse_is_bound_to_recovered_base() -> None:
    script = (ROOT / "dgx/asr/build-midpoint-base.sh").read_text(encoding="utf-8")
    assert "--reuse-base" in script
    assert (
        "sha256:bfa7bcd7c70829e44cd919f22fc68a028816681abe8d4f3b4a2b1ba81e47c134"
        in script
    )
    assert 'test "$base_id" = "$REUSABLE_BASE_IMAGE_ID"' in script
    assert 'assert_historical_inventory "$BASE_TAG"' in script


def test_asr_midpoint_help_is_side_effect_free() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "dgx/asr/build-midpoint-base.sh"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--reuse-base" in result.stdout


def test_asr_midpoint_rejects_unknown_argument_before_docker() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "dgx/asr/build-midpoint-base.sh"), "--unknown"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unknown argument" in result.stderr
```

Run:

```bash
.venv/bin/pytest -q tests/test_dgx_deployment.py -k 'asr and midpoint'
```

Expected: FAIL because the recipe files and reuse interface do not yet exist.

- [ ] **Step 2: Restore only the reviewed midpoint recipe hunks**

Use `git show cf6f716^:<path>` as the exact source and `apply_patch` for the
filesystem edits. Restore the full deleted blobs:

```text
dgx/asr/build-midpoint-base.sh
dgx/asr/eugr-midpoint.patch
```

Restore only the midpoint `ARG SPARK_BASE`, `FROM ${SPARK_BASE}`, and runtime
assertion hunk in `dgx/asr/Dockerfile`. Restore only the focused midpoint build
paragraph in `dgx/README.md`. Do not restore the obsolete 0.6B design/plan,
overwrite the current 1.7B Compose contract, or reverse later unrelated tests.

- [ ] **Step 3: Add the exact fail-closed reuse interface**

Place argument parsing before every Docker, Git, network, and temporary-directory
operation:

```bash
REUSABLE_BASE_IMAGE_ID=sha256:bfa7bcd7c70829e44cd919f22fc68a028816681abe8d4f3b4a2b1ba81e47c134
reuse_base=false

usage() {
  printf 'Usage: %s [--reuse-base]\n' "${0##*/}"
}

case ${1-} in
  "") ;;
  --reuse-base) reuse_base=true ;;
  --help|-h) usage; exit 0 ;;
  *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac
test "$#" -le 1
```

Keep the original full-build path as the default. Wrap only its CUDA manifest,
clone, patch, and native build operations:

```bash
if "$reuse_base"; then
  base_id=$(docker image inspect --format '{{.Id}}' "$BASE_TAG")
  test "$base_id" = "$REUSABLE_BASE_IMAGE_ID"
  assert_historical_inventory "$BASE_TAG"
else
  source_root=$(mktemp -d)
  trap 'rm -rf -- "$source_root"' EXIT
  assert_cuda_arm64_manifest
  git clone --filter=blob:none \
    https://github.com/eugr/spark-vllm-docker.git "$source_root/eugr"
  git -C "$source_root/eugr" checkout --detach "$EUGR_COMMIT"
  git -C "$source_root/eugr" apply --check --unidiff-zero \
    "$script_dir/eugr-midpoint.patch"
  git -C "$source_root/eugr" apply --unidiff-zero \
    "$script_dir/eugr-midpoint.patch"
  expected_upstream_changes=$'Dockerfile\nbuild-and-copy.sh'
  test "$(git -C "$source_root/eugr" diff --name-only)" = \
    "$expected_upstream_changes"
  (
    cd "$source_root/eugr"
    ./build-and-copy.sh --tag "$BASE_TAG" --gpu-arch 12.1a --build-jobs 4 \
      --tf5 --rebuild-vllm --vllm-ref "$VLLM_COMMIT" \
      --rebuild-flashinfer --flashinfer-ref "$FLASHINFER_COMMIT"
  )
  assert_historical_inventory "$BASE_TAG"
fi

docker build --pull=false --build-arg "SPARK_BASE=$BASE_TAG" \
  --tag "$FINAL_TAG" --file "$script_dir/Dockerfile" "$script_dir/.."
assert_historical_inventory "$FINAL_TAG"
```

The reuse branch must validate the complete saved-image inventory, adapter hash,
build metadata, platform, inherited CUDA rootfs prefix, labels, and dependency
cutoff before the derivative build. It must not fall through to a base rebuild.

- [ ] **Step 4: Make the recipe contract green**

Run:

```bash
bash -n dgx/asr/build-midpoint-base.sh
.venv/bin/pytest -q tests/test_dgx_deployment.py
.venv/bin/ruff check tests/test_dgx_deployment.py
.venv/bin/ruff format --check tests/test_dgx_deployment.py
git diff --check
git diff --exit-code cf6f716^ -- dgx/asr/eugr-midpoint.patch
```

For the last command, compare the restored patch blob specifically; expected
output is empty. Review the script/Dockerfile/README/test diff separately to
confirm only the reuse interface and preservation of later changes differ from
`cf6f716^`.

- [ ] **Step 5: Commit the reactivated recipe**

```bash
git add dgx/asr/build-midpoint-base.sh dgx/asr/eugr-midpoint.patch \
  dgx/asr/Dockerfile dgx/README.md tests/test_dgx_deployment.py
git commit -m "build(asr): reactivate v0.23 midpoint image"
```

Expected: no image, wheel, cache, model, environment file, or protected artifact
is committed.

### Task 2: Reuse the exact base and finish the ASR derivative

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-midpoint-reactivation/task-2-report.md`
- Modify only after a proven build defect and a TDD red test: Task 1 files

**Interfaces:**
- Consumes: Task 1 committed recipe and recovered base ID.
- Produces: immutable final candidate image with complete static inventory.

- [ ] **Step 1: Capture the pre-build invariant and namespaces**

Read production image, command, state, health, restart count, and policy. Require
no container named `qwen3-asr-midpoint-v023-test` or
`asr-midpoint-v023-gateway`. Inspect the base tag and require its exact ID before
transferring any build input.

Expected: production is `running|healthy|0|unless-stopped`; both names are empty;
base ID is exactly the global constraint.

- [ ] **Step 2: Transfer only committed build inputs**

Create a clean archive from the Task 1 commit:

```bash
git archive HEAD dgx/asr
```

Create the remote directory with mode `0700`:

```bash
remote_build_root=$(ssh volsch@192.168.68.41 \
  'umask 077; mktemp -d /home/volsch/asr-midpoint-reactivation-XXXXXXXX')
git archive HEAD dgx/asr | \
  ssh volsch@192.168.68.41 "tar -x -C '$remote_build_root'"
```

Transfer and extract the archive into that owner-only GX10 build directory.
Do not transfer `.env`, Compose, models, benchmark artifacts, worktree metadata,
or unrelated files. Verify the remote script, patch, Dockerfile, and audio-lock
SHA-256 values against the local committed files without printing the directory.

- [ ] **Step 3: Build only the derivative**

Run the transferred script with:

```bash
ssh volsch@192.168.68.41 \
  "$remote_build_root/dgx/asr/build-midpoint-base.sh --reuse-base"
```

Expected: the script first validates base
`sha256:bfa7bcd7c70829e44cd919f22fc68a028816681abe8d4f3b4a2b1ba81e47c134`,
does not clone or compile eugr/vLLM/FlashInfer, builds the derivative, and passes
its complete final-image inventory.

- [ ] **Step 4: Independently inspect the final image**

Require:

```text
tag: dgx-qwen3-asr:spark-midpoint-v023-test
platform: arm64/linux
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
transformers: 5.12.1
flashinfer-python: 0.6.12
soundfile: 0.13.1
av: 17.0.1
adapter SHA-256: e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d
```

Also require the exact base image as the history/rootfs prefix, no runtime
compiler addition by the derivative, and no installed duplicate Torch or vLLM
distribution. Record the final image ID and byte size.

- [ ] **Step 5: Write the transcript-free build report**

Record exact committed input hashes, base/final identities, version inventory,
adapter/build-metadata proof, whether the native base path ran (`false`), build
duration, and unchanged production invariant. Omit remote paths and raw Docker
archive content.

### Task 3: Prove the 1.7B runtime and run the ten non-speech cases once

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-midpoint-reactivation/task-3-report.md`
- Create outside Git: `.superpowers/sdd/2026-08-09-spark-vllm-asr-midpoint-reactivation/performance-report.md`
- Modify: none

**Interfaces:**
- Consumes: Task 2 final image, immutable normalized gateway image `sha256:f434a9c6533dd050968e56f547206ca4660ab4aa53d2f9d85ee6b5d8f12f473c`, exact 1.7B model revision, owner-only frozen benchmark.
- Produces: one transcript-free non-speech gate decision with real CUDA proof.

- [ ] **Step 1: Start one isolated candidate**

Use exact name `qwen3-asr-midpoint-v023-test`, restart policy `no`, all GPUs,
4 GiB shared memory, network `voice_default`, loopback
`127.0.0.1:18001:8001`, offline flags, read-only production model cache, and:

```bash
docker run -d \
  --name qwen3-asr-midpoint-v023-test \
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
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /home/volsch/.cache/huggingface:/root/.cache/huggingface:ro \
  dgx-qwen3-asr:spark-midpoint-v023-test \
  vllm serve /root/.cache/huggingface/hub/models--UrocyonF--Qwen3-ASR-1.7B-NVFP4/snapshots/61ad4d533c64e033a750b66c44aad6f18634997e \
  --served-model-name qwen3-asr \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.08 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --trust-remote-code
```

Do not add `VLLM_USE_V2_MODEL_RUNNER` or any other tuning variable.

- [ ] **Step 2: Prove bounded readiness, model identity, API, and CUDA**

Poll for at most 300 seconds. Require `/v1/models` to contain exactly
`qwen3-asr`, zero restarts, and the exact image/model command. Start the exact
immutable normalized gateway as `asr-midpoint-v023-gateway` on loopback port
`18002`, pointing only to the candidate.

Send one authorized hash-validated German WAV with `language=de`. Require HTTP
`200`, normalized response keys exactly `text`, a string value without printing
it, candidate PID intersection with the GPU compute process, and nonzero
aggregate SM during the request.

- [ ] **Step 3: Validate the frozen discriminator contract**

Require these exact safe identities before inference:

```text
corpus version: asr-companion-de-v1+4519e93d3f8f6b99
manifest: 44a3d411e0adcd15a740a9f59ee6aaf52d2946fc3956be08908ab9c516982663
ordered ten cases: 2eb9ea534ecee8c84bc27b6d6bd3011b88609bea52bd3e3ab189d4c3c8a50db7
runner: 091c57013d1422c09bacf72060b95a593a91a8171661d4f6ebe80d29c218cdb1
corpus validator: 1f1279e8c5d836a885aa27632364ceec36615a40443ae83484fc5e1d7c6876a1
scorer: e4d655c39e56ff9c09bd7362753b9c3770a0ef29e2281171d8aacb6331c41bf2
request contract: d5164cfa9c2898f0adacb666a69e771f58dfb8955dfdcd01f0238f83c7c150a9
```

Validate owner-only modes and every selected audio hash without printing paths
or content.

- [ ] **Step 4: Run all ten non-speech cases exactly once**

Send `non_speech-01` through `non_speech-10` in established order, concurrency
`1`, through `http://127.0.0.1:18002/v1/audio/transcriptions` with
`language=de`. Write raw results once to an owner-only `0600` artifact and score
only after all ten requests finish. A valid run is `10/10`; do not retry or
inspect text before scoring.

Require:

```text
completed == 10
failed == 0
non_speech_hallucinated_words <= 5
malformed_responses == 0
normalized_nondeterministic_cases == []
non_transcript_addition_cases == []
language_change_cases == []
command_risk_cases == []
candidate_cuda == true
```

The result is terminal for this plan. Do not run the complete A/B even on pass.

- [ ] **Step 5: Validate and report without protected content**

Validate the safe-result schema recursively against forbidden keys and types,
finite numeric bounds, exact case map, corpus/model/image hashes, raw artifact
hash, and `0600`/`0700` modes. Record immutable identities, aggregate and
per-case word counts, request/CUDA gates, artifact hashes, and PASS/FAIL only.

### Task 4: Clean up and record the candidate outcome

**Files:**
- Modify: `docs/superpowers/specs/2026-08-09-spark-vllm-asr-midpoint-reactivation-design.md`
- Modify: `docs/superpowers/plans/2026-08-09-spark-vllm-asr-midpoint-reactivation.md`
- Test: complete repository surface

**Interfaces:**
- Consumes: Task 3 terminal decision.
- Produces: zero test containers, retained midpoint images, unchanged production, historical non-executable outcome, green branch.

- [ ] **Step 1: Remove only the two named test containers**

Remove `asr-midpoint-v023-gateway` and `qwen3-asr-midpoint-v023-test`. Require
zero exact-name matches afterward. Retain base and final candidate images; do
not prune any image or build cache.

- [ ] **Step 2: Prove final production and image retention**

Compare full production image/command/state/health/restarts/policy to Task 2
preflight and require byte equality. Inspect both retained midpoint image IDs
and final platform/size.

- [ ] **Step 3: Append the exact historical outcome**

Mark both design and plan completed historical records. Record build identity,
whether base reuse succeeded, ten-case counts, all safety gates, final decision,
cleanup counts, retained images, and production invariant. Replace the
agentic-worker header with a prominent
`COMPLETED HISTORICAL RECORD — DO NOT EXECUTE` banner, state that every command
is historical evidence only, and mark
every conditional step that did not run explicitly `NOT EXECUTED`.

- [ ] **Step 4: Run fresh full verification**

```bash
bash -n dgx/asr/build-midpoint-base.sh
.venv/bin/pytest -q
.venv/bin/ruff check agent tests dgx
.venv/bin/ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
git status --short
```

Expected: every command passes and only the two intended outcome documents are
modified after the Task 1 recipe commit.

- [ ] **Step 5: Commit the evidenced outcome**

```bash
git add \
  docs/superpowers/specs/2026-08-09-spark-vllm-asr-midpoint-reactivation-design.md \
  docs/superpowers/plans/2026-08-09-spark-vllm-asr-midpoint-reactivation.md
git commit -m "docs(asr): record v0.23 midpoint non-speech gate"
```

Expected: owner-only reports remain ignored and the tracked worktree is clean.
