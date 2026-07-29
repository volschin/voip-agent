# Post-Merge Residual Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two load-bearing findings left by the final scoped review:
bound whole-WAV TTS admission from HTTP arrival and enforce the exact
two-second PCM backlog across every in-flight handoff.

**Architecture:** The async stable-speech handler owns one absolute admission
deadline beginning before `asyncio.to_thread()` scheduling. It polls operation
completion and client disconnect without a separate monitor task; timeout or
disconnect sets the runtime cancellation event and cancels the async
thread-wrapper so an unscheduled queued job cannot later run. PJSIP reserves
three 20 ms blocks outside its byte-limited buffer: sink-held, queue-held, and
producer-held.

**Tech Stack:** Python 3.14, asyncio, FastAPI/Starlette, NumPy, pytest,
PJSUA2, Ruff, Docker Compose

## Global Constraints

- Preserve stable whole-WAV VoIP synthesis and the accepted quality-first
  latency trade-off.
- Only a model generation that already owns the runtime model lock may finish
  after HTTP cancellation.
- Admission timeout includes default-executor queue delay and runtime lock
  delay.
- Preserve the 300 ms prebuffer and enforce at most 64,000 aggregate bytes,
  exactly two seconds of 16 kHz mono PCM16.
- Preserve direct PJSIP PCM, ARI rollback, sentence ordering, barge-in,
  priority lifecycle, voice profile, and secret contracts.
- Do not deploy until tests, scoped review, image builds, and machine
  acceptance pass.
- Final completion still requires a fresh real-call auditory gate.

---

### Task 1: Bound stable TTS admission from request arrival

**Files:**
- Modify: `dgx/tts/api.py`
- Modify: `tests/test_tts_api.py`

**Interfaces:**
- Consumes: `runtime.synthesize(..., cancel_event, lock_timeout)`.
- Produces: `_run_stable_synthesis(...)` with an absolute request-arrival
  deadline and no leaked disconnect-monitor task.

- [ ] **Step 1: Add a failing saturated-executor timeout test**

Occupy every worker in the loop's default executor with blocking functions,
then call `_run_stable_synthesis()` with a `0.05` second admission timeout.
Assert within `0.20` seconds that:

```python
with pytest.raises(SynthesisAdmissionTimeout):
    await asyncio.wait_for(operation, timeout=0.20)
assert runtime.calls == []
```

Release and await all blocker jobs in `finally`, then wait another event-loop
turn and assert `runtime.calls` remains empty.

- [ ] **Step 2: Add a failing saturated-executor disconnect test**

With the same saturated executor, make the request report disconnected before
a worker is available. Assert the handler raises `_ClientDisconnected` within
`0.20` seconds, the queued runtime callable never starts after worker release,
and no background task remains.

- [ ] **Step 3: Add a monitor-finalization regression**

Run successful, timeout, disconnect, and caller-cancel cases. Snapshot
`asyncio.all_tasks()` before and after each case and assert no
`_wait_for_disconnect` or stable-synthesis helper remains when the handler
returns.

- [ ] **Step 4: Run the new tests and prove the current implementation fails**

Run:

```bash
venv/bin/pytest tests/test_tts_api.py -k \
  "saturated_executor or stable_monitor_finalization" -v
```

Expected before the fix: timeout/disconnect remains pending beyond `0.20`
seconds or a monitor/helper task survives.

- [ ] **Step 5: Implement one polling owner with an absolute deadline**

In `_run_stable_synthesis()`:

- capture `deadline = asyncio.get_running_loop().time() +
  admission_timeout_seconds` before creating the `to_thread` task;
- keep the runtime `cancel_event` and runtime `lock_timeout`;
- wait for the operation in short bounded intervals;
- between waits, check `request.is_disconnected()`;
- on absolute deadline, set the cancel event, cancel the operation wrapper,
  and raise `SynthesisAdmissionTimeout`;
- on disconnect, set the event, cancel the wrapper, and raise
  `_ClientDisconnected`;
- on caller cancellation, set the event, cancel the wrapper, and re-raise;
- do not await a worker that has not started or a model generation already
  running;
- consume/settle only already-completed task results;
- remove `_wait_for_disconnect` when it is no longer needed.

- [ ] **Step 6: Run focused and complete TTS tests**

```bash
venv/bin/pytest tests/test_tts_api.py tests/test_tts_clone_runtime.py -v
venv/bin/ruff check dgx/tts/api.py tests/test_tts_api.py
venv/bin/ruff format --check dgx/tts/api.py tests/test_tts_api.py
```

Expected: all commands exit zero and no task-leak warning appears.

- [ ] **Step 7: Commit**

```bash
git add dgx/tts/api.py tests/test_tts_api.py
git commit -m "fix(tts): bound admission from request arrival"
```

---

### Task 2: Reserve every in-flight PCM block

**Files:**
- Modify: `agent/pjsip.py`
- Modify: `tests/test_conversation.py`
- Modify: `dgx/README.md`

**Interfaces:**
- Consumes: 640-byte PCM blocks from the pipeline and the byte-bounded
  `_PcmByteQueue`.
- Produces: a maximum aggregate backlog of 64,000 bytes across buffer,
  sink-held, queued, and producer-held audio.

- [ ] **Step 1: Strengthen the failing aggregate-budget test**

Instrument the blocked playback path so the test observes:

- `buffer.buffered_bytes`;
- the 640-byte block held by the sink after queue removal;
- `queue.queued_bytes`;
- the 640-byte block held by the producer while awaiting `put()`.

Assert:

```python
assert (
    buffer.buffered_bytes
    + sink_held_bytes
    + queue.queued_bytes
    + producer_held_bytes
    <= PjsipAudioSink.MAX_AHEAD_BYTES
)
```

The pre-fix measurement must reproduce 64,640 bytes against the 64,000-byte
limit.

- [ ] **Step 2: Run the strengthened test and prove it fails**

```bash
venv/bin/pytest \
  tests/test_conversation.py::test_blocked_pjsip_playback_never_exceeds_two_second_aggregate_backlog \
  -v
```

Expected before the fix: assertion reports 64,640 bytes.

- [ ] **Step 3: Reserve three external blocks**

Change:

```python
STREAM_BUFFER_MAX_BYTES = MAX_AHEAD_BYTES - 3 * PCM_BLOCK_BYTES
```

The invariant becomes:

```text
62,080 buffer + 640 sink-held + 640 queued + 640 producer-held = 64,000 bytes
```

Keep queue capacity one block, 20 ms pipeline rechunking, and the 300 ms
prebuffer unchanged.

- [ ] **Step 4: Clarify DGX provisioning order**

Move the mandatory private-profile provisioning section before the first
`docker compose up` instruction in `dgx/README.md`. Preserve the documented
ownership, modes, manifest/WAV, profile ID, pinned model, validation, and
rollback details.

- [ ] **Step 5: Run focused media checks**

```bash
venv/bin/pytest tests/test_conversation.py tests/test_pjsip.py tests/test_pipeline.py -v
venv/bin/ruff check agent/pjsip.py tests/test_conversation.py
venv/bin/ruff format --check agent/pjsip.py tests/test_conversation.py
git diff --check
```

Expected: aggregate backlog never exceeds 64,000 bytes and all commands exit
zero.

- [ ] **Step 6: Commit**

```bash
git add agent/pjsip.py tests/test_conversation.py dgx/README.md
git commit -m "fix(audio): account for producer-held PCM"
```

---

### Task 3: Verify, review, deploy, and accept

**Files:**
- Review all Task 1-2 changes.
- Operational evidence belongs in this plan's ignored SDD workspace.

**Interfaces:**
- Consumes: reviewed Tasks 1-2.
- Produces: exact deployed image and fresh auditory acceptance.

- [ ] **Step 1: Run complete repository verification**

```bash
venv/bin/pytest -q
venv/bin/ruff check agent tests dgx/tts
venv/bin/ruff format --check agent tests dgx/tts
docker compose -p voip-agent \
  --project-directory /home/volsch/projekte/voip-agent \
  -f compose.yml config --quiet
docker compose -p voip-agent \
  --project-directory /home/volsch/projekte/voip-agent \
  -f compose.pjsip-poc.yml config --quiet
docker compose --project-directory dgx \
  -f dgx/docker-compose.yml config --quiet
```

- [ ] **Step 2: Build and import-smoke both Python 3.14 images**

Build from the exact worktree, not the main checkout. Verify network-disabled
imports for Python 3.14, `webrtcvad`, `pjsua2`, and the appropriate entry
module.

- [ ] **Step 3: Request one independent scoped code review**

The reviewer must verify both residual findings, task cleanup, 64,000-byte
arithmetic, tests, and absence of new Critical/Important defects.

- [ ] **Step 4: Deploy only the VoIP agent from the exact reviewed commit**

Preserve and record the current rollback image. Tag the exact built image as
the Compose service image, then recreate only `voip-agent` with `--no-deps`.

- [ ] **Step 5: Repeat machine acceptance**

Prove exact image identity, SIP `200 OK`, restart zero, stable endpoint only,
ordered responses, cancellation/recovery, priority acquire/renew/release,
error-free logs, and reproducible admission/backlog probes.

- [ ] **Step 6: Repeat real-call acceptance**

Exercise greeting, multi-sentence response, audible interruption, following
response, clean hangup, and immediate subsequent call. Record the expected
larger stable-synthesis reaction time separately from audio correctness.
