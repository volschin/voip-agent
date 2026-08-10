# Qualifying AI Candidates Skill Design

**Date:** 2026-08-10
**Status:** Approved for implementation
**Skill destination:** `~/.codex/skills/qualifying-ai-candidates`

## Goal

Create a personal Codex skill that applies the reusable lessons from prior
GX10 A/B tests and candidate benchmarks to future AI candidates. The skill must
make candidate qualification faster without weakening comparison integrity,
production safety, privacy, or modality-specific acceptance.

It covers ASR, TTS, LLM, speech-to-speech, embeddings, rerankers, OCR, vision,
and generative-media candidates. It must prefer the smallest sufficient
evidence path and reuse current repository tooling where it already provides
hardened, candidate-specific checks.

## Non-goals

- Do not create a universal benchmark or orchestration engine.
- Do not replace repository-owned runners, comparators, evidence validators,
  recovery procedures, or deployment contracts.
- Do not treat vendor benchmarks, model cards, or successful API probes as
  local GX10 acceptance evidence.
- Do not install a newer version, mutate production, roll out a candidate, or
  infer deployment authorization from benchmark eligibility.
- Do not copy secrets, private audio, transcripts, prompts, caller data, model
  caches, raw GPU rows, or protected evidence into the skill.

## Skill Architecture

Use a compact `SKILL.md` for the common workflow and progressive disclosure
for detailed knowledge:

```text
qualifying-ai-candidates/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── lessons-and-failures.md
│   ├── modality-gates.md
│   └── evidence-and-tooling.md
└── assets/
    ├── candidate-brief.md
    └── candidate-result.md
```

`SKILL.md` defines the evidence-first decision flow and routes the agent to the
relevant reference. `lessons-and-failures.md` retains reusable errors and their
countermeasures without narrating individual sessions. `modality-gates.md`
defines specialized acceptance surfaces. `evidence-and-tooling.md` explains
how to discover and reuse the current repository scripts by responsibility.
The two assets are short output contracts copied and completed for a new
candidate.

Do not add a generic validator script initially. The existing verifiers are
deliberately bound to exact candidates, artifacts, consumers, and production
states. A new abstraction would repeat the earlier guard-layer ratchet without
a demonstrated repeated need.

## Qualification Workflow

### 1. Establish the current contract

Read the applicable repository instructions, current branch and worktree,
consumer implementation, deployed contract, live state when authorized, and
existing benchmark artifacts. Historical documents are discovery inputs, not
proof of current state.

State the candidate claim before choosing tests. Classify the comparison as an
image-only A/B, one-variable diagnostic, tuned recipe, different-model
comparison, subjective candidate selection, or end-to-end architectural
replacement. Do not attribute a result to one delta when more than one relevant
variable changed.

### 2. Validate upstream state online

For work that may depend on a newer release, current artifact, compatibility
claim, or known limitation, verify the current state through primary sources:

- official release and source repositories;
- OCI registry manifests and immutable image digests;
- PyPI release APIs and exact non-yanked artifacts;
- Hugging Face model repositories, revisions, files, licenses, and model cards;
- official hardware and runtime compatibility documentation.

Record the verification time, primary source, version or revision, platform,
license constraints, artifact filename, size, and digest when available. Check
ARM64/Linux and GB10 compatibility rather than relying on generic support
tables. If online verification is unavailable, report that the latest state is
not currently verified and do not claim otherwise.

An online discovery may propose a new candidate. It never silently substitutes
a newer version into an approved candidate and never proves local performance.

### 3. Audit reusable artifacts

Treat local images, wheels, model snapshots, templates, manifests, protected
packages, and historical results as reusable only after exact identity checks.
A mutable tag, cache presence, filename, or old success label is insufficient.

Require the applicable combination of image ID or registry digest, model
revision and tree manifest, byte hash and size, platform, installed-distribution
multiset, adapter or template hash, measurement-path identity, and production
contract. Distinguish reusable build input from comparable result evidence.
Historical results remain comparable only when measurement inputs and relevant
operating conditions are still identical.

### 4. Freeze the candidate brief

Complete `assets/candidate-brief.md` with:

- candidate claim and comparison class;
- online validation and reusable-artifact audit;
- exact baseline and candidate identities;
- one explicit allowed-delta set and all invariants;
- consumer and measurement-path contracts;
- hard candidate gates, metric-local gates, and adoption-only gates;
- the cheapest sufficient discriminator;
- privacy, production, recovery, and authorization boundaries.

If the intended delta cannot be proven statically, stop before an expensive or
live benchmark.

### 5. Use staged evidence

Run static provenance and compatibility gates before model load. Then run the
smallest discriminator that can reject the candidate reliably. Execute the
complete quality, load, latency, startup, memory, and end-to-end program only
after the discriminator passes.

Freeze corpus, ordering, gateway, normalization, scoring, request contract,
sampling, concurrency, timeout, and thresholds before seeing candidate output.
Use one unchanged consumer-facing gateway across different native APIs. Hash
the shared measurement path and require the same identity across comparable
candidates.

Do not retry individual cases, tune after seeing output, remove outliers, or
change gates. Permit a predeclared complete replication only for its narrowly
defined failure class. A quality or safety failure is terminal when the brief
says so.

### 6. Isolate and seal each candidate

Run one candidate per isolated lifecycle. A candidate receives its own run ID,
source, backups, boots when required, protected raw evidence, acceptance
marker, and safe aggregate. A later candidate failure must not invalidate an
already sealed result.

Separate load-bearing candidate failures from metric-local evidence failures.
Wrong identity, unhealthy runtime, restart, CPU or cloud fallback, consumer
contract failure, privacy failure, absent real hardware execution, or failed
restoration rejects the candidate. An unavailable but optional startup event or
auxiliary runtime counter invalidates only that metric when readiness and the
measured workload are independently proven.

### 7. Restore and accept production

Snapshot the current production state immediately before authorized mutation,
prepare recovery, and change only the intended component. After success or
failure, remove only named benchmark resources, restore the exact prior state,
and verify image, command, model, health, restart count, policy, topology,
consumer request, and unrelated services as applicable.

Eligibility never authorizes rollout. Adoption requires a separate instruction
and fresh modality-specific live acceptance.

### 8. Classify the result precisely

Complete `assets/candidate-result.md` and use exactly one terminal status:

- `invalid`: comparison or evidence cannot support the claimed conclusion;
- `ineligible`: valid benchmark evidence failed a mandatory gate;
- `eligible`: all benchmark gates passed;
- `recommended`: eligible and preferred by the declared decision rule;
- `adopted`: separately authorized rollout completed and passed live acceptance.

Record executed steps, actual identities, evidence hashes, gate results,
skipped phases, waivers, cleanup, production state, remaining uncertainty, and
the next authorized action. Never convert a restored rollback into a successful
adoption claim.

## Modality Gates

All candidates require exact provenance, an explicit delta, the actual consumer
contract, complete requests, real target-hardware execution, privacy, isolation,
and restoration. Add the relevant modality gates:

- **ASR:** German real-path and telephone audio, WER/CER, entity and
  number/time/date handling, non-speech hallucinations and command risk,
  latency, queueing, concurrency, and cancellation.
- **TTS:** German intelligibility and prosody, stable speaker identity,
  clipping/glitches/repetition/truncation/reference leakage, codec correctness,
  first-audio latency, cancellation, and subjective telephone listening as a
  final gate. Real-speaker references require explicit rights and private,
  read-only handling.
- **LLM:** German and character fidelity, valid tools and structured output,
  streaming protocol, TTFT, decode rate, concurrent load, UMA, startup, and
  exact prompt, template, thinking, parser, sampling, context, and model
  identity.
- **Speech-to-speech:** German telephone quality, full-duplex turn-taking,
  barge-in, self-talk and loops, stable voice, end-to-end latency, tool and
  consent contracts, and real PJSIP call acceptance.
- **Embeddings and rerankers:** unchanged retrieval corpus, Recall, MRR, nDCG,
  dimensional and normalization contracts, latency, concurrency, and memory.
- **OCR, vision, and generative media:** real consumer data, task-specific
  accuracy plus required visual or auditory review, deterministic provenance,
  latency, VRAM or UMA, privacy, and output-contract validation.

For a new modality, derive gates from the real consumer path and failure cost.
Do not substitute a generic leaderboard metric.

## Reusable Lessons and Tooling

The references must preserve at least these non-obvious lessons:

- A mixed model/runtime change invalidates an image-only conclusion.
- Exact-key API checks can reject valid additive fields; validate the actual
  consumer contract.
- Artifact bytes, revision, size, release identity, platform, and full delta
  matter; names and mutable tags do not.
- Python `assert` is not a fail-closed production verifier because optimized
  mode removes it.
- Verify before promoting a candidate tag and preserve distribution
  multiplicities, rootfs ancestry, image configuration, and adapter bytes.
- Log strings are unstable evidence. Prefer structured configuration and make
  optional runtime counters metric-local.
- Guard code can consume the experiment. Stop hardening when it no longer
  protects a load-bearing contract; measure and seal one candidate first.
- Prime and measure candidates independently. Retained GX10 unified memory can
  require a fresh boot before a fair memory or startup comparison.
- An aborted preparation without workload evidence is not a measurement.
- A focused pass can justify a documented waiver for a known unrelated timing
  flake; never alter the benchmark to hide it.
- HTTP success, ASR roundtrip, or GPU health cannot replace subjective TTS
  speaker and telephone acceptance.
- Exact request counts can be invalid semantic acceptance gates when sentence
  segmentation varies; require the meaningful minimum behavior instead.

`references/evidence-and-tooling.md` routes agents to current equivalents of:

- `voip-agent/dgx/asr/build-*-candidate.sh` and
  `verify_*_candidate.py` for exact candidate construction and static delta
  proof;
- `voip-agent/tests/test_dgx_deployment.py` for mutation, optimized-mode,
  verify-before-promote, and dependency-closure tests;
- `ai-companion/scripts/benchmark_measurement_path.py` for shared measurement
  identity;
- `benchmark_candidate.py`, `benchmark_evidence.py`, and
  `benchmark_compare.py` for candidate-local sealing, protected evidence, safe
  aggregates, and comparisons;
- `benchmark_transaction.sh` for authorized GX10 transaction and exact restore
  boundaries.

The skill must instruct the agent to inspect the current files and help output
before use. Historical plans marked completed or non-executable are evidence,
not runbooks.

## Validation Strategy

Follow skill TDD without touching production:

1. Run realistic artifact-analysis scenarios without the skill and capture the
   baseline decisions and rationalizations.
2. Create only the guidance needed to correct observed failures.
3. Repeat the same scenarios with the skill and compare decisions.
4. Tighten the skill only for demonstrated gaps and re-run affected scenarios.

The scenarios cover:

- a mixed model/runtime A/B that must be classified `invalid`;
- current online release discovery plus a misleading local cache artifact;
- an optional startup/event failure that must remain metric-local;
- a TTS result with HTTP, CUDA, and objective audio evidence but no subjective
  speaker/telephone acceptance.

Forward tests receive raw artifacts and a user-like task, not the expected
answer. They are read-only and must not call Portainer, mutate Docker, reboot
GX10, or touch production. Validate the finished folder with the Skill Creator
validator, verify `agents/openai.yaml` against `SKILL.md`, inspect word and line
counts, and run any bundled testable resource. No generic script is expected in
the initial version.

## Acceptance

The design is implemented when:

- the skill is discoverable under `~/.codex/skills/qualifying-ai-candidates`;
- its metadata triggers for candidate comparison, A/B benchmark, model/runtime
  upgrade, online latest-version validation, and artifact-reuse questions;
- the body remains concise and routes detailed knowledge progressively;
- the two assets produce complete pre-run and terminal contracts;
- the four forward scenarios improve the observed baseline behavior;
- validation and metadata generation pass with fresh evidence;
- no production or private data was modified or copied.
