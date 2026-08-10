# vLLM 0.24 ASR Wheel Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Build an otherwise identical Qwen3-ASR candidate with official vLLM 0.24.0 and humming-kernels 0.1.6 wheels, run the frozen non-speech gate, and conditionally run the complete same-model benchmark.

**Architecture:** Add a candidate-only, hash-verified two-wheel overlay on immutable production image `sha256:0fadf01c...`; leave production Compose and Dockerfile unchanged. Prove the complete package/image delta statically, then use the existing normalized gateway and protected frozen benchmark in the mandatory non-speech-first order.

**Tech Stack:** Bash, Python 3.12, Docker BuildKit, ARM64 NVIDIA GB10, CUDA 13.0, PyTorch 2.11, vLLM 0.24.0, humming-kernels 0.1.6, UrocyonF/Qwen3-ASR-1.7B-NVFP4, pytest, Ruff.

## Global Constraints

- Baseline: `dgx-qwen3-asr:vllm023-615e858c` at `sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930`.
- Candidate: `dgx-qwen3-asr:vllm024-pypi-test`.
- Replace exactly `vllm` 0.23.0 with 0.24.0 and `humming-kernels` 0.1.4 with 0.1.6 using the approved official wheel assets and hashes.
- Use `pip install --no-cache-dir --no-deps --force-reinstall`; no index, resolver, network, source, sdist, compiler, or production-component rebuild.
- Preserve all other distribution multiplicities, the production rootfs prefix, image configuration, model, runtime, gateway, request bytes, scorer, and ordering.
- Require vLLM 0.24 adapter SHA `639d3691fae9195ed38e17306a29b04bc60025e1119d0090443ec7d935eceffd`; no patching.
- Production, `dgx/asr/Dockerfile`, and `dgx/docker-compose.yml` remain immutable.
- Run ten non-speech probes exactly once. Run 70/70 quality and 12/12 load only after PASS.
- Never commit or expose audio, transcripts, protected paths, caller data, raw GPU rows, models, wheels, archives, or owner-only reports.
- No result authorizes rollout, merge, push, or PR.

---

### Task 1: Add the fail-closed candidate recipe

**Files:**
- Create: `dgx/asr/Dockerfile.vllm024-wheel-candidate`
- Create: `dgx/asr/build-vllm024-wheel-candidate.sh`
- Create: `dgx/asr/verify_vllm024_wheel_candidate.py`
- Modify: `tests/test_dgx_deployment.py`
- Modify: `dgx/README.md`

**Interfaces:** Exact baseline plus two official wheels in; verified candidate tag out.

- [ ] **Step 1: Write focused failing deployment-contract tests**

Add behavior tests for exact inventory/release success and wrong package,
version, config, platform, rootfs, adapter, duplicates, multiplicity, or release
fields. Exercise the builder with fake commands to prove safe help/argument
handling, wrong-base rejection before download, verify-before-promote, and
base-drift rejection. Check the exact two assets, one read-only bind-mounted
`RUN`, no `COPY`, no compiler, and no build network or resolver.

- [ ] **Step 2: Observe the intended RED state**

```bash
/home/volsch/projekte/voip-agent/venv/bin/pytest -q \
  tests/test_dgx_deployment.py -k vllm024_wheel_candidate
```

Expected: FAIL because the three recipe files do not exist.

- [ ] **Step 3: Implement the minimal recipe**

The Dockerfile accepts only `ARG QUALIFIED_ASR_BASE`, inherits it, bind-mounts
the verified wheel directory read-only, and performs one dependency-closed pip
replacement. The verifier provides `verify-file PATH SIZE SHA256`,
`verify-release PACKAGE PATH`, `capture-image IMAGE OUTPUT`, and
`verify-images BASE_JSON CANDIDATE_JSON`. Checks remain active with
`PYTHONOPTIMIZE=1`, preserve duplicate multiplicity, require exactly the two
version changes, exact adapter hashes, ARM64/Linux, rootfs `21 + 1`, equal
configuration, and successful isolated `pip check`.

The builder recognizes only `--help`; checks the immutable base before any
download; verifies exact PyPI JSON and bytes in mode-0700 temporary storage;
binds a random temporary tag to the exact base ID; builds untagged with
`DOCKER_BUILDKIT=1`, `--network=none`, and `--pull=false`; verifies inventories;
rechecks the base; and only then promotes the image. Cleanup is exact.

- [ ] **Step 4: Verify locally**

```bash
bash -n dgx/asr/build-vllm024-wheel-candidate.sh
/home/volsch/projekte/voip-agent/venv/bin/pytest -q \
  tests/test_dgx_deployment.py -k vllm024_wheel_candidate
/home/volsch/projekte/voip-agent/venv/bin/pytest -q tests/test_dgx_deployment.py
/home/volsch/projekte/voip-agent/venv/bin/ruff check tests/test_dgx_deployment.py \
  dgx/asr/verify_vllm024_wheel_candidate.py
/home/volsch/projekte/voip-agent/venv/bin/ruff format --check \
  tests/test_dgx_deployment.py dgx/asr/verify_vllm024_wheel_candidate.py
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit the recipe**

```bash
git add dgx/asr/Dockerfile.vllm024-wheel-candidate \
  dgx/asr/build-vllm024-wheel-candidate.sh \
  dgx/asr/verify_vllm024_wheel_candidate.py dgx/README.md \
  tests/test_dgx_deployment.py docs/superpowers/specs/2026-08-10-vllm024-asr-wheel-candidate-design.md \
  docs/superpowers/plans/2026-08-10-vllm024-asr-wheel-candidate.md
git commit -m "build(asr): add vLLM 0.24 wheel candidate"
```

### Task 2: Build and verify the candidate on GX10

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-10-vllm024-asr-wheel-candidate/task-2-report.md`
- Modify Task 1 files only after a proven defect and new red regression test.

**Interfaces:** Committed recipe and live invariant in; immutable candidate and safe provenance out.

- [ ] **Step 1: Capture preflight**

Require exact production image/command/model and
`running|healthy|0|unless-stopped`, normalized gateway identity, and zero exact
test containers `qwen3-asr-vllm024-test`, `asr-vllm024-gateway`, and
`asr-vllm024-production-gateway`.

- [ ] **Step 2: Transfer committed inputs only**

Stream `git archive HEAD dgx/asr` into a mode-0700 GX10 temporary directory and
verify recipe hashes without exposing its path. Transfer no environment,
models, benchmark data, or worktree state.

- [ ] **Step 3: Build once and verify independently**

Run the builder once. Then independently recheck image ID, platform, size,
rootfs prefix, configuration, exact distribution multiset/delta, adapter,
`pip check`, imports, and versions. Any mismatch rejects the image and leaves
production untouched.

- [ ] **Step 4: Write the owner-only build report**

Record committed input hashes, base/candidate IDs, wheel identities, equality
proof, adapter hashes, platform/size/duration, and unchanged production only.

### Task 3: Run the ten-case non-speech discriminator

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-10-vllm024-asr-wheel-candidate/task-3-report.md`
- Create outside Git: `.superpowers/sdd/2026-08-10-vllm024-asr-wheel-candidate/performance-report.md`

**Interfaces:** Verified image and frozen benchmark in; safe PASS/FAIL and full-test decision out.

- [ ] **Step 1: Validate the protected artifact**

Use the established owner-only validator. Require mode `0600`, exact corpus,
ten-case/order/runner/scorer/request hashes, fixed case IDs, and `language=de`.
Print neither path nor content.

- [ ] **Step 2: Start and prove candidate plus gateway**

Start `qwen3-asr-vllm024-test` and `asr-vllm024-gateway` with the exact runtime.
Require loopback-only ports, restart `no`, exact image/model, readiness,
normalized text-only response, zero restarts, PID-correlated CUDA, and nonzero
SM activity during a real request.

- [ ] **Step 3: Run the exact ten cases once and gate**

Require `10/10`, words `<=5`, zero malformed/addition/language-change/
command-risk/nondeterminism cases, and CUDA true. Do not inspect, retry, or
tune. Write safe identities, hashes, per-case counts, aggregate/safety/CUDA,
and decision. On FAIL skip Task 4; on PASS continue.

### Task 4: Run the conditional complete comparison

**Files:**
- Append outside Git: `.superpowers/sdd/2026-08-10-vllm024-asr-wheel-candidate/performance-report.md`

**Interfaces:** Task 3 PASS and identical gateways in; 70/70 plus 12/12 eligibility result out.

- [ ] **Step 1: Revalidate identities and gateway equality**

Only after PASS start `asr-vllm024-production-gateway`; require identical
normalization and exact model revision on both sides.

- [ ] **Step 2: Run the prescribed series once**

Run production quality 70, candidate quality 70, production load 12, candidate
load 12. Every request succeeds and both sides have fresh container-bound CUDA.

- [ ] **Step 3: Gate and report safely**

Require no quality/entity/number/time/date/non-speech regression, p50 ratio
`<=1.05`, p90 ratio `<=1.10`, and all provenance/safety gates. Replicate only
under the established shared-host latency-only rule, never after quality or
safety failure. Record only safe aggregates, ratios, CUDA maxima, counts,
provenance, and decision.

### Task 5: Clean up, verify, and record outcome

**Files:**
- Modify: `docs/superpowers/specs/2026-08-10-vllm024-asr-wheel-candidate-design.md`
- Modify: `docs/superpowers/plans/2026-08-10-vllm024-asr-wheel-candidate.md`

**Interfaces:** Benchmark evidence in; zero test containers, unchanged production, safe tracked outcome out.

- [ ] **Step 1: Clean up exact containers and prove production invariant**

Remove only the three exact test container names if present, require all counts
zero, and retain the candidate image. Compare production image, command, model,
state, health, restart count, and policy exactly with preflight.

- [ ] **Step 2: Append the safe execution outcome**

Record candidate ID/size, wheel provenance, static delta, discriminator, whether
the full series ran, complete safe metrics when applicable, cleanup, production
invariant, and eligibility. Include no protected content or paths.

- [ ] **Step 3: Run fresh full verification**

```bash
/home/volsch/projekte/voip-agent/venv/bin/pytest -q
/home/volsch/projekte/voip-agent/venv/bin/ruff check agent tests dgx
/home/volsch/projekte/voip-agent/venv/bin/ruff format --check agent tests dgx
docker compose --env-file dgx/.env.example -f dgx/docker-compose.yml config --quiet
git diff --check
git status --short
```

- [ ] **Step 4: Commit the safe outcome**

```bash
git add docs/superpowers/specs/2026-08-10-vllm024-asr-wheel-candidate-design.md \
  docs/superpowers/plans/2026-08-10-vllm024-asr-wheel-candidate.md
git commit -m "docs(asr): record vLLM 0.24 benchmark"
```

Expected: owner-only data remains outside Git. No rollout, merge, push, or PR.
