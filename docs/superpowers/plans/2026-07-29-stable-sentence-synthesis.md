# Stable Sentence Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every VoIP response sentence through the stable whole-WAV Qwen
TTS operation used by the clean greeting while preserving ordered sentence
playback, bounded prefetch, cancellation, and direct 16 kHz PCM output.

**Architecture:** LLM tokens continue to be segmented and queued with depth two.
The single TTS consumer synthesizes one complete sentence at a time through
`VoicePipeline.synthesize_pcm16()`, then yields that sentence to the existing
PCM playback queue. The experimental raw TTS streaming client remains
available, but is no longer part of the VoIP response path.

**Tech Stack:** Python 3.12+, asyncio, NumPy, pytest/pytest-asyncio, Ruff,
Docker Compose, PJSUA2, Qwen3-TTS HTTP API

## Global Constraints

- Do not change the selected voice profile, Qwen model, or shared TTS server
  contract.
- Keep LLM token streaming, German sentence segmentation, and the bounded
  two-entry sentence prefetch queue.
- Run only one TTS model generation at a time and preserve source order.
- Keep the 300 ms PJSIP PCM prebuffer and two-second maximum-ahead bound.
- Keep the production output contract signed-16-bit little-endian mono 16 kHz
  PCM and preserve the ARI 8 kHz A-law rollback adapter.
- Barge-in must cancel and discard the local turn promptly even if a
  synchronous server-side model call finishes later.
- The real call and user auditory confirmation remain mandatory acceptance
  gates.

---

## File map

| File | Responsibility |
|---|---|
| `tests/test_pipeline.py` | Regression tests proving stable whole-WAV response synthesis, order, empty output, recovery, and cancellation |
| `agent/pipeline.py` | Select stable per-sentence synthesis for the live response path |
| `README.md` | Describe the production VoIP TTS path and its latency semantics |
| `docs/pjsip-migration.md` | Keep the direct-PJSIP operational contract current |

No changes are required in `agent/tts.py`, `agent/main.py`, `dgx/tts/`, the
PJSIP media sink, or the ARI adapter.

---

### Task 1: Pin the stable response-synthesis contract

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `agent/pipeline.py`

**Interfaces:**
- Consumes: `synthesize_pcm16(text: str) -> bytes`, which calls the injected
  whole-WAV `tts` callable and returns 16 kHz PCM16 bytes.
- Produces: `_tts_pcm16_chunks(text: str) -> AsyncIterator[bytes]`, yielding
  zero or one complete stable sentence; `process_turn_stream(...)` continues to
  yield these sentence blobs in queue order.

- [ ] **Step 1: Replace the streaming-resampler unit test with a failing stable-call test**

Replace `test_tts_pcm16_chunks_preserve_resampler_state` with:

```python
async def test_tts_pcm16_chunks_use_stable_whole_wav_synthesis():
    source = np.arange(-3000, 3000, dtype=np.int16)
    tts = AsyncMock(return_value=_wav_samples(source))
    tts_stream = AsyncMock()
    pipe = VoicePipeline(
        stt=AsyncMock(),
        llm=AsyncMock(),
        tts=tts,
        tts_stream=tts_stream,
    )

    chunks = [chunk async for chunk in pipe._tts_pcm16_chunks("Hallo")]

    assert chunks == [resample_pcm16(source, 24_000, 16_000)]
    tts.assert_awaited_once_with("Hallo")
    tts_stream.assert_not_called()
```

- [ ] **Step 2: Add a failing empty-output regression test**

Add:

```python
async def test_tts_pcm16_chunks_emit_nothing_when_stable_synthesis_fails():
    pipe = VoicePipeline(
        stt=AsyncMock(),
        llm=AsyncMock(),
        tts=AsyncMock(return_value=b"not a wav"),
        tts_stream=AsyncMock(),
    )

    chunks = [chunk async for chunk in pipe._tts_pcm16_chunks("Kaputt")]

    assert chunks == []
```

- [ ] **Step 3: Convert the ordered sentence test to stable WAV synthesis**

In `test_sentence_prefetch_is_bounded_and_preserves_order`, replace
`tts_stream` with a stable callable:

```python
    async def tts(text):
        tts_calls.append(text)
        if text == "Eins.":
            first_tts_started.set()
            await release_first.wait()
        samples = np.full(240, len(tts_calls), dtype=np.int16)
        return _wav_samples(samples)

    forbidden_stream = AsyncMock()
    pipe = VoicePipeline(
        stt=stt,
        llm=None,
        tts=tts,
        llm_stream=llm_stream,
        tts_stream=forbidden_stream,
    )
```

Retain the existing bounded-consumption and exact-order assertions, then add:

```python
    assert len(output) == 6
    assert forbidden_stream.call_count == 0
```

- [ ] **Step 4: Convert the live-turn, filler, recovery, and cancellation fixtures**

For response tests that currently define `tts_stream`, inject whole-WAV
callables instead:

```python
    async def tts(_text):
        return _wav(2400)
```

For the cancellation test, use:

```python
    async def tts(_text):
        try:
            tts_started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()
```

The cancellation assertion remains:

```python
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await stream.aclose()
    assert closed.is_set()
```

For the filler test, collect the stable callable's text and return `_wav(2400)`.
For the recovery test, collect calls and assert the last one is
`FALLBACK_RECOVERY`; import that constant from `agent.pipeline`.

- [ ] **Step 5: Add exact once-only response-order assertions**

Extend the ordered test with:

```python
    assert tts_calls == ["Eins.", "Zwei.", "Drei.", "Vier.", "Fünf.", "Sechs."]
    assert len(tts_calls) == len(set(tts_calls))
```

This test data contains unique markers, so the second assertion proves no
sentence was requested more than once.

- [ ] **Step 6: Run the focused tests and confirm the expected failure**

Run:

```bash
venv/bin/pytest tests/test_pipeline.py -k \
  "tts_pcm16_chunks or sentence_prefetch or process_turn_stream or cancelled_turn" \
  -v
```

Expected: failures show `_tts_pcm16_chunks` attempts to iterate
`tts_stream` instead of awaiting the stable `tts` callable.

- [ ] **Step 7: Implement the minimal stable per-sentence adapter**

In `agent/pipeline.py`, remove `StreamingPcm16Resampler` from the import:

```python
from agent.audio import resample_pcm16
```

Replace `_tts_pcm16_chunks` with:

```python
    async def _tts_pcm16_chunks(self, text):
        """Synthesize one stable sentence and yield 16 kHz mono PCM16."""
        pcm_16k = await self.synthesize_pcm16(text)
        if pcm_16k:
            yield pcm_16k
```

Do not change sentence queueing, history mutation, recovery behavior, or the
constructor signature. Keeping `tts_stream` in the constructor preserves the
public wiring contract while ensuring the production pipeline no longer
invokes it.

- [ ] **Step 8: Run focused tests and confirm they pass**

Run:

```bash
venv/bin/pytest tests/test_pipeline.py -v
```

Expected: all `tests/test_pipeline.py` tests pass.

- [ ] **Step 9: Run static checks for the changed Python files**

Run:

```bash
venv/bin/ruff check agent/pipeline.py tests/test_pipeline.py
venv/bin/ruff format --check agent/pipeline.py tests/test_pipeline.py
```

Expected: both commands exit zero.

- [ ] **Step 10: Commit the implementation**

```bash
git add agent/pipeline.py tests/test_pipeline.py
git commit -m "fix(tts): synthesize responses in stable sentence mode"
```

---

### Task 2: Align operator documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/pjsip-migration.md`

**Interfaces:**
- Consumes: stable per-sentence behavior implemented in Task 1.
- Produces: operational documentation that distinguishes LLM streaming from
  experimental TTS codec streaming.

- [ ] **Step 1: Update the README live-turn description**

Replace the paragraph that describes
`/v1/audio/speech/stream` with:

```markdown
The live turn streams LLM tokens into the German sentence segmenter. Each
completed sentence is synthesized sequentially through the stable
`/v1/audio/speech` whole-WAV endpoint, converted once from 24 to 16 kHz PCM,
then fed through the 300 ms PCM prebuffer to PJSUA2. Up to two completed
sentence segments may wait ahead; generation and playback overlap across
sentences. The experimental `/v1/audio/speech/stream` codec path is retained
for diagnostics but is not used for VoIP responses.
```

Keep the existing barge-in paragraph, adding one sentence:

```markdown
A server-side full-sentence model call may finish after client cancellation;
its late result is discarded and never reaches playback.
```

- [ ] **Step 2: Update the latency section**

Replace claims about the first streaming TTS chunk with:

```markdown
On the live response path, time-to-first-audio includes stable synthesis of the
first completed sentence. Subsequent LLM segmentation, sentence synthesis, and
playback overlap while preserving sentence order. Intelligibility and
once-only ordering take precedence over raw codec-stream time-to-first-chunk.
```

- [ ] **Step 3: Update the PJSIP migration audio path**

In `docs/pjsip-migration.md`, state that each completed LLM sentence is
synthesized through `/v1/audio/speech`, WAV-decoded, converted once from 24 to
16 kHz PCM, and written to the existing bounded playback queue. State that
`/v1/audio/speech/stream` is not used by production VoIP playback.

- [ ] **Step 4: Check documentation consistency**

Run:

```bash
rg -n "/v1/audio/speech/stream|stateful 24-to-16|streaming TTS chunk" \
  README.md docs/pjsip-migration.md
git diff --check
```

Expected: any remaining `/stream` mention explicitly identifies it as
diagnostic/non-production; `git diff --check` exits zero.

- [ ] **Step 5: Commit the documentation**

```bash
git add README.md docs/pjsip-migration.md
git commit -m "docs(tts): describe stable sentence playback"
```

---

### Task 3: Review and repository verification

**Files:**
- Review: `agent/pipeline.py`
- Review: `tests/test_pipeline.py`
- Review: `README.md`
- Review: `docs/pjsip-migration.md`

**Interfaces:**
- Consumes: Tasks 1 and 2 commits.
- Produces: independently reviewed, repository-green repair candidate.

- [ ] **Step 1: Request an independent code review**

Use `superpowers:requesting-code-review`. The reviewer must inspect the diff
from `dbc6ee2` through `HEAD` for:

- accidental use of `tts_stream` in the VoIP response path;
- duplication or order regressions;
- cancellation or history-state regressions;
- empty-output and recovery behavior;
- stale production documentation.

- [ ] **Step 2: Resolve every Critical, Important, and Minor finding**

For each concrete finding, reproduce it with a focused test when applicable,
apply the smallest correction, rerun the focused test, and commit with a
Conventional Commit subject.

- [ ] **Step 3: Run the full unit suite**

```bash
venv/bin/pytest -q
```

Expected: all tests pass; baseline before this repair was 248 tests.

- [ ] **Step 4: Run repository lint and format checks**

```bash
venv/bin/ruff check agent tests dgx/tts
venv/bin/ruff format --check agent tests dgx/tts
```

Expected: both commands exit zero.

- [ ] **Step 5: Validate both Compose configurations**

Run from the feature worktree while using the deployment project's `.env`:

```bash
docker compose -p voip-agent \
  --project-directory /home/volsch/projekte/voip-agent \
  -f compose.yml config --quiet
docker compose -p voip-agent \
  --project-directory /home/volsch/projekte/voip-agent \
  -f compose.asterisk.yml config --quiet
```

Expected: both commands exit zero without printing secrets.

- [ ] **Step 6: Confirm the candidate diff and worktree state**

```bash
git diff --check dbc6ee2..HEAD
git status --short
git log --oneline dbc6ee2..HEAD
```

Expected: no whitespace errors, clean worktree, and only repair-related commits.

---

### Task 4: Build, deploy, and perform live acceptance

**Files:**
- Deploy source: current worktree commit
- Runtime verification: Portainer/Docker deployment and service logs

**Interfaces:**
- Consumes: repository-green exact Git commit from Task 3.
- Produces: deployed exact image plus automated and human auditory evidence.

- [ ] **Step 1: Record rollback and pre-deployment state**

Record without printing secrets:

```bash
git rev-parse HEAD
docker inspect voip-agent --format \
  '{{.Image}} restart={{.RestartCount}} started={{.State.StartedAt}}'
docker image inspect voip-agent-voip-agent:rollback-pre-continuous-pcm \
  --format '{{.Id}}'
```

Expected: exact candidate commit and both current/rollback image IDs are
captured in the progress ledger.

- [ ] **Step 2: Build an image from the exact worktree state**

Use the existing feature Compose deployment command:

```bash
docker compose -p voip-agent \
  --project-directory /home/volsch/projekte/voip-agent \
  -f /home/volsch/projekte/.worktrees/voip-agent-shared-ai-traefik/compose.yml \
  build voip-agent
```

Expected: build exits zero. Record the resulting immutable image ID.

- [ ] **Step 3: Replace only the VoIP agent service**

```bash
docker compose -p voip-agent \
  --project-directory /home/volsch/projekte/voip-agent \
  -f /home/volsch/projekte/.worktrees/voip-agent-shared-ai-traefik/compose.yml \
  up -d --no-deps voip-agent
```

Expected: `voip-agent` is recreated from the candidate image; shared AI
services are not restarted.

- [ ] **Step 4: Verify container and SIP continuity**

Inspect health, restart count, recent logs, and SIP registration. Expected:

- running container image equals the recorded candidate image;
- restart count is zero;
- SIP registration is `200 OK`;
- no queue, playback, PCM, media, traceback, or TTS errors appear.

- [ ] **Step 5: Verify the live endpoint selection**

Exercise one marked multi-sentence pipeline response or inspect request logs.
Expected:

- each sentence produces one `POST /v1/audio/speech`;
- no VoIP response produces `POST /v1/audio/speech/stream`;
- sentence markers appear exactly once and in source order;
- playback begins only after stable first-sentence synthesis.

- [ ] **Step 6: Verify cancellation and recovery**

Start a long first-sentence request, cancel the local response task, and begin a
new marked request. Expected:

- local PCM playback clears promptly;
- no late audio from the cancelled response is played;
- the next request completes after any bounded server model-lock delay;
- logs contain no stranded queue/task or session-state error.

- [ ] **Step 7: Perform the real call**

The call must cover:

1. greeting;
2. a multi-sentence answer;
3. one interruption/barge-in;
4. a following answer;
5. clean hangup.

Expected machine evidence: SIP remains registered, restart count stays zero,
requests use the stable endpoint, and no runtime media errors occur.

Expected human evidence: the user confirms the responses are intelligible,
ordered, and free of repeated fragments. Without this confirmation, the repair
is not complete.

- [ ] **Step 8: Record final state**

Record exact commit, image ID, container restart count, SIP state, automated
checks, real-call timestamps, and any still-open auditory defect. Do not mark
the goal complete unless the auditory gate passes.
