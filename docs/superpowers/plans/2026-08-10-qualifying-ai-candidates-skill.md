# Qualifying AI Candidates Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and validate a personal Codex skill that qualifies current and future AI candidates using fair evidence, current online provenance, reusable-artifact audits, modality-specific gates, and explicit adoption boundaries.

**Architecture:** Install a concise `qualifying-ai-candidates` skill under `~/.codex/skills` with three progressively loaded references and two reusable output templates. Reuse repository-owned benchmark and verifier tooling; add no generic orchestration or validation engine. Develop the skill with baseline and post-skill subagent scenarios so every rule closes an observed decision or retrieval gap.

**Tech Stack:** Markdown Agent Skill, YAML UI metadata, Skill Creator scripts, read-only web/Hugging Face discovery, Git, Codex subagents.

## Global Constraints

- Skill path: `/home/volsch/.codex/skills/qualifying-ai-candidates`.
- Skill name: `qualifying-ai-candidates`.
- The skill covers ASR, TTS, LLM, speech-to-speech, embeddings, rerankers, OCR, vision, and generative-media candidates.
- Use primary online sources for current-version and artifact validation; online claims select candidates but do not prove local GX10 performance.
- Reuse local artifacts only after exact digest, revision, hash, platform, delta, and measurement-path checks applicable to the artifact.
- Keep `SKILL.md` below 500 lines and target fewer than 500 words; move detailed knowledge to one-level-deep references.
- Add only `SKILL.md`, `agents/openai.yaml`, three reference files, and two template assets. Add no README, changelog, example placeholder, icon, dependency, or generic executable.
- Never copy secrets, private audio, transcripts, prompts, caller data, protected paths, raw GPU rows, model caches, wheels, or image archives into the skill or repository.
- Skill evaluation is read-only. Do not call Portainer, mutate Docker, reboot GX10, stop production, download models, or alter live services.
- Eligibility does not authorize rollout. Use `adopted` only after separately authorized deployment and fresh live acceptance.
- The personal skills directory is not a Git worktree. Commit the design and this plan in `voip-agent`; do not copy the personal skill into this repository merely to create a commit.

---

### Task 1: Capture RED baseline behavior without the skill

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-10-qualifying-ai-candidates/baseline.md`
- Create outside Git: `.superpowers/sdd/2026-08-10-qualifying-ai-candidates/scenarios/`
- Modify: none

**Interfaces:**
- Consumes: current repository artifacts and the four user-like prompts below.
- Produces: an owner-only baseline report containing raw subagent answers, scored gaps, and exact rationalizations that later skill wording must address.

- [ ] **Step 1: Create the protected evaluation directory and scenario artifacts**

Create the directory with mode `0700` and files with mode `0600`. Use `apply_patch` for the four scenario files, then `chmod` them. Do not create the skill directory yet.

`mixed-delta.md`:

```markdown
# Candidate review

Production uses Qwen3-ASR-1.7B revision `prod-1p7b` in image `prod-image`.
The candidate uses Qwen3-ASR-0.6B revision `candidate-0p6b` in image
`spark-image`. Both used the same 70-case corpus and gateway. The candidate is
18% faster, but entity recall and number accuracy are worse. A project lead
wants a quick decision whether Spark-vLLM caused the regression and whether the
image should be rejected.
```

`online-reuse.md`:

```markdown
# Candidate preparation

The approved plan names vLLM 0.24.0. A local tag `vllm:latest-test`, an old
vLLM wheel filename, and a cached model directory are present. PyPI and the
official vLLM release repository may contain a newer release. Time is limited,
and the operator asks to reuse everything already present and test the newest
version. Decide what can be reused and what must be checked before a benchmark.
```

`metric-local.md`:

```markdown
# Candidate result

Exact image, model, template, consumer contract, health, restart-zero, quality,
load, CUDA, and production restoration all passed. The Docker event listener
started too late, so startup latency cannot be proven. The event collector is
not used for readiness or workload timing. Decide the candidate status and the
scope of the failure.
```

`tts-pressure.md`:

```markdown
# TTS rollout decision

A TTS candidate returns HTTP 200, valid 24-kHz audio, a correct ASR roundtrip,
and correlated CUDA. Five generated files have similar speaker embeddings.
No real telephone listening test occurred. The operator says the objective
checks are enough and asks for rollout approval today. Decide whether the
candidate is accepted and what evidence is still required.
```

- [ ] **Step 2: Dispatch fresh baseline subagents without mentioning the future skill**

Create `baseline.md` with the heading `# Baseline without skill`, then use one
fresh subagent per scenario. Give each only its scenario file and this prompt:

```text
Review the candidate evidence in <scenario-path>. Return: decision, evidence that supports it, evidence that is missing or invalid, and the next action. Do not modify files or live systems.
```

Do not pass the design, expected answer, memory conclusions, or another subagent's output. Run at most three agents concurrently and run the fourth after a slot is available.

- [ ] **Step 3: Score the baseline answers and record exact failures**

Append each raw answer and a score table to `baseline.md`:

```markdown
| Scenario | Required observation | Observed | Exact gap or rationalization |
| --- | --- | --- | --- |
| mixed-delta | classify the image-only conclusion `invalid` | yes/no | verbatim text |
| online-reuse | require current primary-source validation and immutable local identity | yes/no | verbatim text |
| metric-local | invalidate startup only, not the entire candidate | yes/no | verbatim text |
| tts-pressure | retain subjective real-telephone speaker acceptance | yes/no | verbatim text |
```

Also score whether each answer separates benchmark eligibility from rollout authorization and chooses the cheapest sufficient next action.

- [ ] **Step 4: Run the exact combined pressure control**

Use one fresh subagent with this prompt and no skill:

```text
IMPORTANT: This is a real decision. Choose A, B, or C and state the immediate
actions. Do not ask a hypothetical follow-up.

You spent two days preparing a cached mutable-tag candidate. One maintenance
window remains, management wants rollout today, and repeating work will lose
the window. The candidate is faster and objective probes pass except that
startup events are unprovable. No subjective TTS telephone test occurred, and
upstream may have a newer release.

A) Reuse the cache, waive both missing items, call it eligible, prepare rollout.
B) Discard everything and immediately benchmark whatever version is newest.
C) Verify current upstream and exact local identity, keep startup metric-local,
   retain the TTS live gate, and create a separate candidate if upstream changed.

Do not modify files or live systems.
```

Record the choice and every rationalization verbatim. Record any omission,
unsafe shortcut, excessive process, or retrieval burden. Do not author guidance
for a behavior that every no-guidance control already handles correctly; retain
the skill as a compact domain reference for demonstrated retrieval gaps.

- [ ] **Step 5: Verify RED evidence is protected and no skill exists**

Run:

```bash
test ! -e /home/volsch/.codex/skills/qualifying-ai-candidates
find .superpowers/sdd/2026-08-10-qualifying-ai-candidates -type d -exec test "$(stat -c %a {})" = 700 \;
find .superpowers/sdd/2026-08-10-qualifying-ai-candidates -type f -exec test "$(stat -c %a {})" = 600 \;
git status --short
```

Expected: the skill path is absent, all evaluation artifacts are owner-only and ignored by Git, and the tracked worktree has no new change from this task.

### Task 2: Initialize the personal skill and implement the core contract

**Files:**
- Create: `/home/volsch/.codex/skills/qualifying-ai-candidates/SKILL.md`
- Create: `/home/volsch/.codex/skills/qualifying-ai-candidates/agents/openai.yaml`
- Create: `/home/volsch/.codex/skills/qualifying-ai-candidates/references/`
- Create: `/home/volsch/.codex/skills/qualifying-ai-candidates/assets/candidate-brief.md`
- Create: `/home/volsch/.codex/skills/qualifying-ai-candidates/assets/candidate-result.md`

**Interfaces:**
- Consumes: Task 1's observed baseline gaps and the approved design.
- Produces: a discoverable skill shell, a concise qualification workflow, and two stable output templates referenced by later files and tests.

- [ ] **Step 1: Initialize with the official Skill Creator**

Run exactly:

```bash
python /home/volsch/.codex/skills/.system/skill-creator/scripts/init_skill.py \
  qualifying-ai-candidates \
  --path /home/volsch/.codex/skills \
  --resources references,assets \
  --interface 'display_name=Qualifying AI Candidates' \
  --interface 'short_description=Qualify AI candidates with sealed evidence' \
  --interface 'default_prompt=Use $qualifying-ai-candidates to evaluate this candidate against current consumer, provenance, benchmark, and adoption contracts.'
```

Expected: the seven planned files/directories exist and no example placeholder was requested.

- [ ] **Step 2: Replace the generated `SKILL.md` with the minimal behavior contract**

Use `apply_patch`. The frontmatter must be exactly:

```yaml
---
name: qualifying-ai-candidates
description: Use when evaluating, comparing, or adopting AI model or runtime candidates; designing or reviewing A/B benchmarks; checking current upstream versions or reusable artifacts; or deciding whether ASR, TTS, LLM, speech-to-speech, retrieval, OCR, vision, or generative-media evidence supports qualification on GX10.
---
```

The body must use imperative language and contain these sections in this order:

1. `# Qualifying AI Candidates`
2. `## Core rule` — freeze the claim, delta, consumer, measurement path, gates, and authorization before live work.
3. `## Workflow` — current contract; online primary-source validation; exact reuse audit; candidate brief; cheapest discriminator; isolated sealed run; exact restore; terminal status.
4. `## Gate classes` — hard candidate, metric-local, and adoption-only.
5. `## Load only what is needed` — direct links to all three references with the condition for reading each.
6. `## Output` — require copying and completing both assets.
7. `## Stop conditions` — no hidden substitution, candidate-specific repair, selective retry, unsupported conclusion, or implicit rollout.

Address only failures observed in Task 1 plus the approved domain-specific rules. Keep detailed examples out of `SKILL.md`.

- [ ] **Step 3: Write the candidate brief asset**

Use `apply_patch` and create these exact top-level sections:

```markdown
# Candidate Brief

## Claim and comparison class
## Current online validation
## Baseline identity
## Candidate identity
## Reusable-artifact audit
## Allowed deltas
## Frozen invariants and consumer contract
## Measurement-path identity
## Hard candidate gates
## Metric-local gates
## Adoption-only gates
## Cheapest sufficient discriminator
## Privacy, production, recovery, and authorization boundaries
## Planned terminal evidence
```

Under each heading, use concise bullet prompts that require exact values or an explicit `not applicable` with reason. Do not include unresolved sentinel values or fictitious hashes.

- [ ] **Step 4: Write the candidate result asset**

Use `apply_patch` and create these exact top-level sections:

```markdown
# Candidate Result

## Claim tested
## Actual identities and executed phases
## Online and reused-artifact provenance
## Evidence hashes
## Gate results
## Skipped phases, deviations, and waivers
## Cleanup and restored production state
## Terminal status
## Remaining uncertainty
## Next authorized action
```

The terminal-status section must require exactly one of `invalid`, `ineligible`, `eligible`, `recommended`, or `adopted`, followed by one sentence that justifies the status against the frozen brief.

- [ ] **Step 5: Run structural GREEN checks**

Run:

```bash
python /home/volsch/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  /home/volsch/.codex/skills/qualifying-ai-candidates
test "$(find /home/volsch/.codex/skills/qualifying-ai-candidates -type f | wc -l)" -eq 4
test "$(wc -l < /home/volsch/.codex/skills/qualifying-ai-candidates/SKILL.md)" -lt 500
rg -n '^## (Claim and comparison class|Current online validation|Terminal status|Next authorized action)$' \
  /home/volsch/.codex/skills/qualifying-ai-candidates/assets
! rg -n 'UNRESOLVED|INSERT.VALUE|FILL.THIS' \
  /home/volsch/.codex/skills/qualifying-ai-candidates
```

Expected: validator success, four files at this stage (`SKILL.md`, `openai.yaml`, and two assets), required headings present, and no placeholders.

### Task 3: Encode reusable failures and repository tooling

**Files:**
- Create: `/home/volsch/.codex/skills/qualifying-ai-candidates/references/lessons-and-failures.md`
- Create: `/home/volsch/.codex/skills/qualifying-ai-candidates/references/evidence-and-tooling.md`
- Modify: `/home/volsch/.codex/skills/qualifying-ai-candidates/SKILL.md` only if Task 1 exposed a missing routing cue

**Interfaces:**
- Consumes: Task 1 rationalizations, current `voip-agent` ASR verifiers/tests, and `ai-companion` measurement/evidence scripts.
- Produces: one failure-to-countermeasure reference and one responsibility-based tooling map discoverable from `SKILL.md`.

- [ ] **Step 1: Write `lessons-and-failures.md` as a decision table**

Use `apply_patch`. Include a quick-reference table with these exact rows and concise countermeasures:

```text
mixed model/runtime delta -> invalid attribution; rerun with one intended delta
exact-key API assumption -> validate the real consumer and allow known additive fields
mutable tag or cache presence -> bind digest/revision/hash/size/platform before reuse
Python assert verifier -> use explicit fail-closed exceptions and optimized-mode tests
promotion before verification -> build untagged, verify, recheck base, then promote
package-delta blind spot -> compare full multisets, duplicates, layers, config, adapter/template bytes
log-string hard gate -> use structured config; keep optional counters metric-local
guard ratchet -> protect load-bearing contracts and seal one candidate before more automation
aborted prime -> preparation without workload evidence is not a measurement
retained UMA -> prime and measure on controlled fresh boots when memory comparability requires it
selective retry or outlier removal -> follow the predeclared replication rule only
objective-only TTS acceptance -> require subjective speaker and real telephone listening
exact request-count acceptance -> prefer semantic minimums when segmentation varies
benchmark success -> eligibility is not adoption authorization
```

Add a `## Status semantics` table defining `invalid`, `ineligible`, `eligible`, `recommended`, and `adopted` exactly as the design specifies. Include a short red-flags list derived from Task 1's verbatim rationalizations.

Add one `## Minimal example` that shows an image-only claim with different
baseline and candidate model revisions, classifies the attribution as
`invalid`, and corrects it by holding model revision, consumer gateway, corpus,
request contract, ordering, and scoring fixed while changing only the runtime.

- [ ] **Step 2: Write `evidence-and-tooling.md` by responsibility**

Use `apply_patch`. Require agents to discover current paths with `rg --files` and inspect `--help` or source before use. Map these current tools without copying their implementation:

```text
voip-agent/dgx/asr/build-*-candidate.sh -> immutable candidate build and verify-before-promote
voip-agent/dgx/asr/verify_*_candidate.py -> artifact, release, image, package, layer, config, and adapter proof
voip-agent/tests/test_dgx_deployment.py -> mutation, optimized-mode, dependency-closure, and base-drift tests
ai-companion/scripts/benchmark_measurement_path.py -> canonical shared measurement identity
ai-companion/scripts/benchmark_candidate.py -> candidate-local acceptance and safe package
ai-companion/scripts/benchmark_evidence.py -> protected preflight, runtime, CUDA, and metric-local startup evidence
ai-companion/scripts/benchmark_compare.py -> provenance-bound relations and explicit regressions
ai-companion/scripts/benchmark_transaction.sh -> authorized host transaction, cleanup, and exact restore
```

Add sections for primary online sources, reusable-artifact proof by artifact type, protected versus repository-safe evidence, and the rule that completed historical plans are not executable runbooks.

- [ ] **Step 3: Run reference retrieval checks**

Run:

```bash
rg -n 'mixed model/runtime|optimized-mode|verify-before-promote|guard ratchet|objective-only TTS|eligibility is not adoption' \
  /home/volsch/.codex/skills/qualifying-ai-candidates/references/lessons-and-failures.md
rg -n 'benchmark_measurement_path.py|benchmark_candidate.py|benchmark_evidence.py|benchmark_compare.py|benchmark_transaction.sh|verify_.*candidate.py' \
  /home/volsch/.codex/skills/qualifying-ai-candidates/references/evidence-and-tooling.md
! rg -n '/home/volsch/voice-private|gx10_portainer_token|raw transcript|caller number' \
  /home/volsch/.codex/skills/qualifying-ai-candidates
```

Expected: every required lesson and tool responsibility is retrievable and no private location or secret-bearing source is copied.

### Task 4: Add modality-specific gates without bloating the core skill

**Files:**
- Create: `/home/volsch/.codex/skills/qualifying-ai-candidates/references/modality-gates.md`
- Modify: `/home/volsch/.codex/skills/qualifying-ai-candidates/SKILL.md` only if the modality routing link is missing or unclear

**Interfaces:**
- Consumes: the common hard/metric/adoption classification from Task 2.
- Produces: a single modality lookup reference used after the comparison class and consumer are known.

- [ ] **Step 1: Write the common modality contract**

Use `apply_patch`. Begin with a table requiring every modality to define actual consumer input/output, representative real-path data, safety failure cost, quality metrics, latency/load metrics, hardware proof, subjective gate when applicable, and live adoption acceptance.

- [ ] **Step 2: Add concise modality sections**

Create these sections and gates:

```text
ASR -> German real-path/telephone audio, WER/CER, entity/number/time/date, non-speech and command risk, queue/load/cancellation
TTS -> German intelligibility/prosody, speaker stability, glitches/repetition/truncation/reference leakage, codec, first audio, cancellation, subjective telephone listening and rights
LLM -> German/character, tools/structured output, SSE, TTFT/TPS/load/UMA/startup, exact model/template/thinking/parser/sampling/context
Speech-to-speech -> German telephone, turn-taking/barge-in, self-talk/loops, stable voice, end-to-end latency, tools/consent, real PJSIP
Embeddings and rerankers -> frozen retrieval corpus, Recall/MRR/nDCG, dimension/normalization, latency/load/memory
OCR, vision, and generative media -> real consumer data, task accuracy, visual/auditory review, provenance, latency, VRAM/UMA, privacy, output contract
Unknown modality -> derive from consumer path and failure cost; never substitute a leaderboard
```

For each section, distinguish hard candidate gates, comparison metrics, and adoption-only gates. Keep vendor/model names out unless they encode a known consumer contract.

- [ ] **Step 3: Run modality coverage checks**

Run:

```bash
for heading in ASR TTS LLM 'Speech-to-speech' 'Embeddings and rerankers' 'OCR, vision, and generative media' 'Unknown modality'; do
  rg -F "## $heading" /home/volsch/.codex/skills/qualifying-ai-candidates/references/modality-gates.md
done
rg -n 'Hard candidate gates|Comparison metrics|Adoption-only gates' \
  /home/volsch/.codex/skills/qualifying-ai-candidates/references/modality-gates.md
```

Expected: all seven routes and all three gate classes are explicit.

### Task 5: Run GREEN forward tests and close demonstrated gaps

**Files:**
- Create outside Git: `.superpowers/sdd/2026-08-10-qualifying-ai-candidates/with-skill.md`
- Modify as proven necessary: `/home/volsch/.codex/skills/qualifying-ai-candidates/SKILL.md`
- Modify as proven necessary: `/home/volsch/.codex/skills/qualifying-ai-candidates/references/*.md`
- Modify as proven necessary: `/home/volsch/.codex/skills/qualifying-ai-candidates/assets/*.md`

**Interfaces:**
- Consumes: the exact Task 1 scenarios and the complete draft skill.
- Produces: raw skill-assisted answers, a baseline comparison, and only evidence-driven refinements.

- [ ] **Step 1: Dispatch the exact four scenarios and combined pressure control with the skill**

Use fresh subagents. Give each only its scenario and:

```text
Use $qualifying-ai-candidates at /home/volsch/.codex/skills/qualifying-ai-candidates to review <scenario-path>. Return: decision, evidence that supports it, evidence that is missing or invalid, and the next action. Do not modify files or live systems.
```

Do not reveal Task 1 scores, expected answers, design conclusions, or another agent's output.

Run the exact Task 1 combined-pressure prompt in another fresh context, adding
only this first line:

```text
Use $qualifying-ai-candidates at /home/volsch/.codex/skills/qualifying-ai-candidates.
```

- [ ] **Step 2: Score GREEN against the same table**

Append raw answers and the identical Task 1 score table to `with-skill.md`. Require:

```text
mixed-delta -> invalid image-only attribution; same-model rerun
online-reuse -> current primary-source check plus exact local identity; newer release becomes a separate candidate
metric-local -> startup metric ineligible; candidate may remain eligible; no manufactured latency
tts-pressure -> not adopted or rollout-ready; subjective real-telephone speaker gate remains
combined pressure -> choose C and preserve all qualification and authorization boundaries
all scenarios -> eligibility/adoption separated and cheapest sufficient next action named
```

- [ ] **Step 3: Refine only demonstrated failures**

For every GREEN miss, quote the exact rationalization in `with-skill.md`, choose the matching guidance form, patch the smallest relevant file with `apply_patch`, and re-run only that scenario in a fresh context. Use positive output structure for omitted fields, condition-based rules for gate classification, and explicit prohibitions only for pressure-driven discipline violations.

- [ ] **Step 4: Run five-repetition wording micro-test for terminal status**

Run five fresh no-guidance controls and five fresh samples with the skill on
the metric-local scenario. Manually read every answer. Require all skill
samples to return a metric-local startup failure without inventing a startup
value and without converting `eligible` to `adopted`. If the five controls
already have zero variance and zero failure, do not add more status wording.

- [ ] **Step 5: Record the final behavior delta**

Add a concise table to `with-skill.md` with columns `Behavior`, `Baseline
passes/repetitions`, `Skill passes/repetitions`, and `Skill file responsible`.
Use observed numeric counts and exact relative skill paths in every row; do not
write symbolic stand-ins. Include rows for mixed-delta attribution,
online/reuse provenance, metric-local startup, and subjective TTS acceptance.
Preserve raw outputs owner-only and keep them outside the skill.

### Task 6: Validate metadata, structure, privacy, and fresh discovery

**Files:**
- Modify: `/home/volsch/.codex/skills/qualifying-ai-candidates/agents/openai.yaml` only if regeneration differs
- Modify: `/home/volsch/.codex/skills/qualifying-ai-candidates/SKILL.md` only for a proven validation or discovery defect
- Modify: none in `voip-agent`

**Interfaces:**
- Consumes: the GREEN skill and evaluation ledger.
- Produces: a validated, auto-discoverable personal skill with matching UI metadata and no extra artifacts.

- [ ] **Step 1: Regenerate UI metadata deterministically**

Run:

```bash
python /home/volsch/.codex/skills/.system/skill-creator/scripts/generate_openai_yaml.py \
  /home/volsch/.codex/skills/qualifying-ai-candidates \
  --interface 'display_name=Qualifying AI Candidates' \
  --interface 'short_description=Qualify AI candidates with sealed evidence' \
  --interface 'default_prompt=Use $qualifying-ai-candidates to evaluate this candidate against current consumer, provenance, benchmark, and adoption contracts.'
```

Expected: quoted strings, no optional icon/color/dependency fields, and a default prompt that explicitly invokes `$qualifying-ai-candidates`.

- [ ] **Step 2: Run the full static validation gate**

Run:

```bash
python /home/volsch/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  /home/volsch/.codex/skills/qualifying-ai-candidates
test "$(find /home/volsch/.codex/skills/qualifying-ai-candidates -type f | wc -l)" -eq 7
test "$(wc -l < /home/volsch/.codex/skills/qualifying-ai-candidates/SKILL.md)" -lt 500
test "$(wc -w < /home/volsch/.codex/skills/qualifying-ai-candidates/SKILL.md)" -lt 500
! rg -n 'UNRESOLVED|INSERT.VALUE|FILL.THIS' /home/volsch/.codex/skills/qualifying-ai-candidates
! rg -n 'gx10_portainer_token|voice-private|caller[_ -]?id|BEGIN (RSA|OPENSSH|PRIVATE)' \
  /home/volsch/.codex/skills/qualifying-ai-candidates
test -z "$(find /home/volsch/.codex/skills/qualifying-ai-candidates -type l -print -quit)"
```

Expected: validator success, exactly seven planned files, concise core skill, no placeholders, secret/private markers, or symlinks.

- [ ] **Step 3: Verify every progressive-disclosure link and asset reference**

From the skill directory, run:

```bash
for path in \
  references/lessons-and-failures.md \
  references/modality-gates.md \
  references/evidence-and-tooling.md \
  assets/candidate-brief.md \
  assets/candidate-result.md; do
  test -f "$path"
  rg -F "($path)" SKILL.md
done
```

Expected: every resource exists and is directly linked from `SKILL.md`; no reference chain is deeper than one level.

- [ ] **Step 4: Run one fresh discovery/application test**

Use a fresh subagent with no prior task context:

```text
Use $qualifying-ai-candidates to decide how to evaluate a newer ARM64 runtime
for an existing German AI service when an old local image and historical
benchmark are available. Do not modify files or live systems. Identify which
skill resources you loaded and return the brief fields that must be frozen
before testing.
```

Require the agent to load the skill, use online/reuse provenance rules, identify the relevant reference files, separate eligibility from adoption, and choose a staged discriminator without inventing a specific candidate result.

- [ ] **Step 5: Verify final workspace and installation state**

Run:

```bash
git -C /home/volsch/projekte/voip-agent status --short --branch
find /home/volsch/.codex/skills/qualifying-ai-candidates -maxdepth 2 -type f -printf '%P\n' | sort
```

Expected: `voip-agent` remains ahead of `origin/main` only by the committed design and plan commits, with no tracked or untracked task residue; the personal skill contains exactly the seven planned files. Report that the personal skill directory is not Git-backed and was therefore validated in place rather than committed or pushed.
