# Spark-vLLM ASR FlashInfer Wheel Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Build an otherwise identical vLLM 0.23 ASR candidate with the official Spark-vLLM FlashInfer 0.6.18 wheels, run the frozen non-speech gate, and conditionally run the full same-model benchmark.

**Architecture:** Add a candidate-only hash-verified wheel overlay on immutable production image `0fadf01c...`; leave production Compose and its Dockerfile unchanged. Prove the package/image delta statically, then use the existing normalized gateway and frozen benchmark in the mandated non-speech-first order.

**Tech Stack:** Bash, Docker BuildKit, ARM64 NVIDIA GB10, CUDA 13.0, PyTorch 2.11, vLLM 0.23.0, FlashInfer 0.6.18, UrocyonF/Qwen3-ASR-1.7B-NVFP4, pytest, Ruff.

## Global Constraints

- Baseline tag: `dgx-qwen3-asr:vllm023-615e858c`.
- Baseline image: `sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930`.
- Candidate tag: `dgx-qwen3-asr:vllm023-flashinfer0618-test`.
- Replace exactly `flashinfer-python`, `flashinfer-jit-cache`, and `flashinfer-cubin` 0.6.12 with the three 0.6.18 assets and hashes in the design.
- Use `pip install --no-deps --force-reinstall`; do not resolve or modify any other distribution.
- Preserve vLLM 0.23.0, Torch 2.11.0+cu130, CUDA 13.0, Transformers 5.12.1, SoundFile 0.13.1, PyAV 17.0.1, and the Qwen3-ASR adapter hash.
- Production, `dgx/asr/Dockerfile`, and `dgx/docker-compose.yml` are immutable throughout the experiment.
- Model repository, revision, command, request bytes, `language=de`, ordering, gateway, runner, scorer, and runtime environment remain unchanged.
- Run the ten non-speech probes exactly once; run the 70/70 quality and 12/12 load comparison only if the discriminator passes.
- Never commit or report audio, raw transcripts, protected paths, caller data, raw GPU rows, model data, wheel files, or image archives.
- No benchmark result authorizes rollout.

---

### Task 1: Add the fail-closed candidate recipe

**Files:**
- Create: `dgx/asr/Dockerfile.flashinfer-wheel-candidate`
- Create: `dgx/asr/build-flashinfer-wheel-candidate.sh`
- Create: `dgx/asr/verify_flashinfer_wheel_candidate.py`
- Modify: `tests/test_dgx_deployment.py`
- Modify: `dgx/README.md`

**Interfaces:**
- Consumes: exact local baseline tag and the three public release assets.
- Produces: `dgx-qwen3-asr:vllm023-flashinfer0618-test` with a verified single-package-family delta.

- [ ] **Step 1: Write focused failing deployment-contract tests**

Add behavior tests that execute the verifier against hand-checked image
inventories and file bytes, including altered package/config/rootfs/adapter/
FlashInfer/duplicate cases. Execute the build script with fake external
commands to prove side-effect-free `--help`, unknown-argument rejection, and
rejection of a wrong base before download. Add the focused static deployment
contract for the exact three upstream release assets and dependency-closed
Dockerfile.

- [ ] **Step 2: Run the focused tests and observe the expected missing-file failures**

```bash
/home/volsch/projekte/voip-agent/venv/bin/pytest -q \
  tests/test_dgx_deployment.py -k flashinfer_wheel_candidate
```

Expected: FAIL because the candidate Dockerfile and script do not exist.

- [ ] **Step 3: Implement the minimal Dockerfile and build script**

The Dockerfile accepts only `ARG QUALIFIED_ASR_BASE`, inherits it, copies the
three wheels, installs their exact filenames in one `python3 -m pip install
--no-cache-dir --no-deps --force-reinstall` instruction, removes the wheel
payloads, and asserts the exact runtime inventory.

The Bash script parses only `--help`, checks the exact base image before any
download, downloads the three exact assets to `mktemp -d`, verifies bytes and
SHA-256, builds with `--pull=false`, then uses the Python verifier to capture
and compare real rootfs, image configuration, adapter, and installed
distribution inventories fail closed.

- [ ] **Step 4: Run focused and repository deployment tests**

```bash
bash -n dgx/asr/build-flashinfer-wheel-candidate.sh
/home/volsch/projekte/voip-agent/venv/bin/pytest -q \
  tests/test_dgx_deployment.py -k flashinfer_wheel_candidate
/home/volsch/projekte/voip-agent/venv/bin/pytest -q tests/test_dgx_deployment.py
/home/volsch/projekte/voip-agent/venv/bin/ruff check tests/test_dgx_deployment.py
/home/volsch/projekte/voip-agent/venv/bin/ruff format --check tests/test_dgx_deployment.py
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit the recipe**

```bash
git add dgx/asr/Dockerfile.flashinfer-wheel-candidate \
  dgx/asr/build-flashinfer-wheel-candidate.sh dgx/README.md \
  tests/test_dgx_deployment.py \
  docs/superpowers/specs/2026-08-10-spark-vllm-asr-flashinfer-wheel-design.md \
  docs/superpowers/plans/2026-08-10-spark-vllm-asr-flashinfer-wheel.md
git commit -m "build(asr): add FlashInfer wheel candidate"
```

### Task 2: Build and verify the immutable candidate on GX10

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-10-spark-vllm-asr-flashinfer-wheel/task-2-report.md`
- Modify only after a proven recipe defect and a new red test: Task 1 files

**Interfaces:**
- Consumes: committed Task 1 recipe and current live production invariant.
- Produces: immutable candidate image plus transcript-free static/runtime provenance.

- [ ] **Step 1: Capture production and namespace preflight**

Require exact production image/command, `running|healthy|0|unless-stopped`, the
exact baseline tag/image, the normalized gateway image, and zero containers
named `qwen3-asr-flashinfer0618-test`, `asr-flashinfer0618-gateway`, or
`asr-flashinfer0618-production-gateway`.

- [ ] **Step 2: Transfer only committed recipe inputs**

Use `git archive HEAD dgx/asr` into a mode-0700 remote temporary directory.
Verify the four committed recipe/input SHA-256 values without printing the
remote path. Do not transfer `.env`, models, benchmark data, or worktree state.

- [ ] **Step 3: Run the build script once**

Execute `build-flashinfer-wheel-candidate.sh` from the transferred archive.
Require all three download digests/sizes, exact base ID, exact rootfs prefix,
unchanged configuration and non-FlashInfer distribution map, FlashInfer
0.6.18 triple, ARM64/Linux, and the retained ASR adapter hash.

- [ ] **Step 4: Independently inspect the candidate**

Repeat the image ID, platform, rootfs prefix, configuration, package inventory,
and import checks outside the build script. Record the candidate image ID and
size. Any mismatch rejects the build without changing production.

- [ ] **Step 5: Write the transcript-free build report**

Record only committed input hashes, baseline/candidate IDs, release asset
identities, package equality proof, adapter hash, platform/size, duration, and
the unchanged production invariant.

### Task 3: Run the ten-case non-speech discriminator

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-10-spark-vllm-asr-flashinfer-wheel/task-3-report.md`
- Create outside Git: `.superpowers/sdd/2026-08-10-spark-vllm-asr-flashinfer-wheel/performance-report.md`
- Modify: none

**Interfaces:**
- Consumes: Task 2 image, exact 1.7B snapshot, immutable normalized gateway, frozen benchmark artifact.
- Produces: one transcript-free PASS/FAIL and authorization decision for Task 4.

- [ ] **Step 1: Validate the protected benchmark artifact and hashes**

Require exactly one owner-only artifact with the established same-model label,
mode `0600`, exact corpus/ten-case/order/runner/scorer/request hashes, fixed ten
case IDs, and `language=de`. Do not print its path or content.

- [ ] **Step 2: Start and prove the isolated candidate and gateway**

Start `qwen3-asr-flashinfer0618-test` and `asr-flashinfer0618-gateway` with the
exact runtime contract. Require loopback-only ports, restart policy `no`, exact
image/model readiness, normalized text-only response shape, candidate PID
correlation, and nonzero CUDA SM during one real request.

- [ ] **Step 3: Send the exact ten non-speech probes once**

Run the protected runner once against the candidate gateway with fixed bytes,
IDs, order, timeout, scorer, and `language=de`. Require `10/10` successful
responses and a transcript-free result artifact. Do not retry or tune.

- [ ] **Step 4: Apply the fixed discriminator**

Require hallucinated words `<= 5`, zero malformed/addition/language-change/
command-risk/nondeterminism cases, and candidate CUDA true. On failure, mark
Task 4 skipped and continue to Task 5. On success, proceed to Task 4.

- [ ] **Step 5: Write the transcript-free discriminator report**

Record exact identities and hashes, request count, per-case word counts,
aggregate count, safety map, CUDA evidence, and the full-test decision.

### Task 4: Run the conditional complete same-model comparison

**Files:**
- Append outside Git: `.superpowers/sdd/2026-08-10-spark-vllm-asr-flashinfer-wheel/performance-report.md`
- Modify: none

**Interfaces:**
- Consumes: Task 3 PASS, current production, ready candidate, identical normalized gateways.
- Produces: complete 70/70 quality and 12/12 load eligibility result.

- [ ] **Step 1: Revalidate both exact model/image identities and gateway equality**

Start the production gateway only after Task 3 passes. Require both gateways
use the same immutable image and normalization contract and both backends serve
the exact 1.7B revision.

- [ ] **Step 2: Run the fixed series in the prescribed order**

Run production quality 70, candidate quality 70, production load 12, and
candidate load 12. Require every request to succeed and fresh container-bound
CUDA evidence for both sides.

- [ ] **Step 3: Apply the complete gate**

Require no quality/non-speech regression, p50 ratio `<= 1.05`, p90 ratio
`<= 1.10`, all request/safety/provenance gates true, and no outlier deletion or
selective repetition.

- [ ] **Step 4: Append the transcript-free full result**

Record quality aggregates, non-speech counts, latency aggregates/ratios, CUDA
booleans/maxima, request counts, exact provenance, and final eligibility.

### Task 5: Clean up, verify, and record the final outcome

**Files:**
- Modify: `docs/superpowers/specs/2026-08-10-spark-vllm-asr-flashinfer-wheel-design.md`
- Modify: `docs/superpowers/plans/2026-08-10-spark-vllm-asr-flashinfer-wheel.md`
- Test: repository verification commands

**Interfaces:**
- Consumes: Task 3 and conditional Task 4 evidence.
- Produces: zero benchmark containers, unchanged production, tracked final report, green repository.

- [ ] **Step 1: Remove only the three exact benchmark container names**

Remove candidate and candidate gateway, plus production gateway only if Task 4
created it. Require all three exact-name counts zero. Retain the candidate image.

- [ ] **Step 2: Prove the final production invariant byte-for-byte**

Compare image, command, state, health, restart count, policy, and model identity
with Task 2 preflight. Require exact equality.

- [ ] **Step 3: Append the exact execution outcome to design and plan**

Record candidate ID/size, wheel provenance, discriminator metrics, whether the
full series ran, full metrics when applicable, cleanup counts, production
invariant, and eligibility. Keep the report transcript-free.

- [ ] **Step 4: Run fresh full repository verification**

```bash
/home/volsch/projekte/voip-agent/venv/bin/pytest -q
/home/volsch/projekte/voip-agent/venv/bin/ruff check agent tests dgx
/home/volsch/projekte/voip-agent/venv/bin/ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
git status --short
```

Expected: all pass and only intended recipe, tests, README, design, and plan are
modified.

- [ ] **Step 5: Commit the evidenced outcome**

```bash
git add docs/superpowers/specs/2026-08-10-spark-vllm-asr-flashinfer-wheel-design.md \
  docs/superpowers/plans/2026-08-10-spark-vllm-asr-flashinfer-wheel.md
git commit -m "docs(asr): record FlashInfer wheel benchmark"
```

Expected: owner-only reports, wheels, images, and benchmark artifacts remain
untracked.
