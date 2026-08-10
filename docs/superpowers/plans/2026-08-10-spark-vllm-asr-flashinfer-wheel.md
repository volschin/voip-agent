# Spark-vLLM ASR FlashInfer Wheel Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Build an otherwise identical vLLM 0.23 ASR candidate with the official Spark-vLLM FlashInfer 0.6.18 wheels, run the frozen non-speech gate, and conditionally run the full same-model benchmark.

**Architecture:** Add a candidate-only hash-verified wheel overlay on immutable production image `0fadf01c...`; leave production Compose and its Dockerfile unchanged. Prove the package/image delta statically, then use the existing normalized gateway and frozen benchmark in the mandated non-speech-first order.

**Tech Stack:** Bash, Docker BuildKit, ARM64 NVIDIA GB10, CUDA 13.0, PyTorch 2.11, vLLM 0.23.0, FlashInfer 0.6.18, UrocyonF/Qwen3-ASR-1.7B-NVFP4, pytest, Ruff.

**Status:** Completed 2026-08-10 UTC. This checked plan is a historical
execution record and no longer authorizes commands or rollout.

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

- [x] **Step 1: Write focused failing deployment-contract tests**

Add behavior tests that execute the verifier against hand-checked image
inventories and file bytes, including altered package/config/rootfs/adapter/
FlashInfer/duplicate cases. Execute the build script with fake external
commands to prove side-effect-free `--help`, unknown-argument rejection, and
rejection of a wrong base before download. Add the focused static deployment
contract for the exact three upstream release assets and dependency-closed
Dockerfile.

- [x] **Step 2: Run the focused tests and observe the expected missing-file failures**

```bash
/home/volsch/projekte/voip-agent/venv/bin/pytest -q \
  tests/test_dgx_deployment.py -k flashinfer_wheel_candidate
```

Expected: FAIL because the candidate Dockerfile and script do not exist.

- [x] **Step 3: Implement the minimal Dockerfile and build script**

The Dockerfile accepts only `ARG QUALIFIED_ASR_BASE`, inherits it, mounts the
verified wheel context read-only for one `python3 -m pip install --no-cache-dir
--no-deps --force-reinstall` instruction, and appends exactly one layer without
retaining wheel payloads.

The Bash script parses only `--help`, checks the exact base image before any
download, verifies the exact release and asset metadata, downloads the three
exact assets to `mktemp -d`, verifies bytes and SHA-256, creates a random
temporary local tag from the immutable base image ID and checks it before and
after an untagged `--pull=false` build, then uses the Python verifier to capture
and compare the exact base-ID rootfs, image
configuration, adapter, and installed distribution inventories fail closed.
Only a verified image is promoted to the documented candidate tag.

- [x] **Step 4: Run focused and repository deployment tests**

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

- [x] **Step 5: Commit the recipe**

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

- [x] **Step 1: Capture production and namespace preflight**

Require exact production image/command, `running|healthy|0|unless-stopped`, the
exact baseline tag/image, the normalized gateway image, and zero containers
named `qwen3-asr-flashinfer0618-test`, `asr-flashinfer0618-gateway`, or
`asr-flashinfer0618-production-gateway`.

- [x] **Step 2: Transfer only committed recipe inputs**

Use `git archive HEAD dgx/asr` into a mode-0700 remote temporary directory.
Verify the four committed recipe/input SHA-256 values without printing the
remote path. Do not transfer `.env`, models, benchmark data, or worktree state.

- [x] **Step 3: Run the build script once**

Execute `build-flashinfer-wheel-candidate.sh` from the transferred archive.
Require all three download digests/sizes, exact base ID, exact rootfs prefix,
unchanged configuration and non-FlashInfer distribution multiset (including
any inherited duplicate metadata), each FlashInfer distribution exactly once
at 0.6.18, ARM64/Linux, and the retained ASR adapter hash.

- [x] **Step 4: Independently inspect the candidate**

Repeat the image ID, platform, rootfs prefix, configuration, package inventory,
and import checks outside the build script. Record the candidate image ID and
size. Any mismatch rejects the build without changing production.

- [x] **Step 5: Write the transcript-free build report**

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

- [x] **Step 1: Validate the protected benchmark artifact and hashes**

Require exactly one owner-only artifact with the established same-model label,
mode `0600`, exact corpus/ten-case/order/runner/scorer/request hashes, fixed ten
case IDs, and `language=de`. Do not print its path or content.

- [x] **Step 2: Start and prove the isolated candidate and gateway**

Start `qwen3-asr-flashinfer0618-test` and `asr-flashinfer0618-gateway` with the
exact runtime contract. Require loopback-only ports, restart policy `no`, exact
image/model readiness, normalized text-only response shape, candidate PID
correlation, and nonzero CUDA SM during one real request.

- [x] **Step 3: Send the exact ten non-speech probes once**

Run the protected runner once against the candidate gateway with fixed bytes,
IDs, order, timeout, scorer, and `language=de`. Require `10/10` successful
responses and a transcript-free result artifact. Do not retry or tune.

- [x] **Step 4: Apply the fixed discriminator**

Require hallucinated words `<= 5`, zero malformed/addition/language-change/
command-risk/nondeterminism cases, and candidate CUDA true. On failure, mark
Task 4 skipped and continue to Task 5. On success, proceed to Task 4.

- [x] **Step 5: Write the transcript-free discriminator report**

Record exact identities and hashes, request count, per-case word counts,
aggregate count, safety map, CUDA evidence, and the full-test decision.

### Task 4: Run the conditional complete same-model comparison

**Files:**
- Append outside Git: `.superpowers/sdd/2026-08-10-spark-vllm-asr-flashinfer-wheel/performance-report.md`
- Modify: none

**Interfaces:**
- Consumes: Task 3 PASS, current production, ready candidate, identical normalized gateways.
- Produces: complete 70/70 quality and 12/12 load eligibility result.

- [x] **Step 1: Revalidate both exact model/image identities and gateway equality**

Start the production gateway only after Task 3 passes. Require both gateways
use the same immutable image and normalization contract and both backends serve
the exact 1.7B revision.

- [x] **Step 2: Run the fixed series in the prescribed order**

Run production quality 70, candidate quality 70, production load 12, and
candidate load 12. Require every request to succeed and fresh container-bound
CUDA evidence for both sides.

- [x] **Step 3: Apply the complete gate**

Require no quality/non-speech regression, p50 ratio `<= 1.05`, p90 ratio
`<= 1.10`, all request/safety/provenance gates true, and no outlier deletion or
selective repetition.

- [x] **Step 4: Append the transcript-free full result**

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

- [x] **Step 1: Remove only the three exact benchmark container names**

Remove candidate and candidate gateway, plus production gateway only if Task 4
created it. Require all three exact-name counts zero. Retain the candidate image.

- [x] **Step 2: Prove the final production invariant byte-for-byte**

Compare image, command, state, health, restart count, policy, and model identity
with Task 2 preflight. Require exact equality.

- [x] **Step 3: Append the exact execution outcome to design and plan**

Record candidate ID/size, wheel provenance, discriminator metrics, whether the
full series ran, full metrics when applicable, cleanup counts, production
invariant, and eligibility. Keep the report transcript-free.

- [x] **Step 4: Run fresh full repository verification**

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

- [x] **Step 5: Commit the evidenced outcome**

```bash
git add docs/superpowers/specs/2026-08-10-spark-vllm-asr-flashinfer-wheel-design.md \
  docs/superpowers/plans/2026-08-10-spark-vllm-asr-flashinfer-wheel.md
git commit -m "docs(asr): record FlashInfer wheel benchmark"
```

Expected: owner-only reports, wheels, images, and benchmark artifacts remain
untracked.

## Historical Execution Result

- Initial image `sha256:1891cdd1...ed46a` and its positive benchmark were
  superseded after review found optimized-Python, release-identity,
  tag-promotion, base-binding, and multi-layer verification gaps.
- Those findings were fixed with explicit checks and optimized-mode tests,
  exact release metadata, checked ephemeral base tagging from the immutable
  ID, verify-before-promote, and a one-layer BuildKit bind mount.
- Final candidate:
  `sha256:864acad23c3c55738f50ed546a2a175767e8835fa12ddd3fc4a58e3a9ded56d1`,
  ARM64/Linux, `25,265,654,356` bytes, exact rootfs layer count `21 + 1`.
- Static delta: only the three FlashInfer distributions changed from 0.6.12
  to 0.6.18; all other distribution multiplicities, rootfs ancestry, image
  configuration, and adapter bytes matched the baseline.
- Non-speech gate: PASS, `10/10`, five words with per-case counts
  `1,1,1,0,0,0,0,1,0,1`, no structural or safety failures, CUDA true,
  maximum SM `72%`, safe SHA-256 `2e17fb7f...84bd`.
- Complete comparison: FAIL, although both sides completed `70/70` quality
  and `12/12` load with CUDA.
- Quality: production WER/CER/macro WER
  `0.04330708661417323/0.03785245268443414/0.05783146591970121`;
  candidate `0.045275590551181105/0.038238702201622246/0.061799719887955185`.
- Load: production p50/p90 `383.4179179975763/544.1918869037181 ms`;
  candidate `404.35268403962255/637.96085503418 ms`; ratios
  `1.0546003852698886/1.1723086477161173` exceeded both limits.
- CUDA: both true, both maximum SM `96%`.
- Replication: forbidden because quality also failed; series count remained one.
- Complete safe result:
  `ea78f048cea1dfe1e7963d3ff768e3b23446f81f65129388918a9b84635bf5c9`.
- Cleanup: all three exact benchmark-container counts zero; candidate image
  retained; production unchanged as `running|healthy|0|unless-stopped` on
  `sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930`.
- Decision: final candidate rejected; no rollout was executed or authorized.
