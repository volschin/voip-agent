# Smart Turn v2 Turn Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add semantic, prosody-aware end-of-turn detection (Smart Turn v2 on DGX) so the agent waits through mid-utterance pauses instead of talking over the caller, without regressing barge-in and degrading safely when the model is slow/unavailable.

**Architecture:** A new `TurnDetectorClient` calls a DGX HTTP service that classifies a buffered waveform as `complete`/`incomplete`. In LISTENING the VAD floor is lowered (~200 ms) to propose endpoint candidates fast; the classifier confirms flush-now vs keep-listening. Barge-in (SPEAKING/PROCESSING) keeps the legacy 800 ms VAD via a second per-call buffer. Behind feature flag `turn_detection_enabled` (default off) the legacy single-buffer path is unchanged.

**Tech Stack:** Python 3.12+, pydantic-settings, httpx (async), webrtcvad, numpy, pytest-asyncio (`asyncio_mode=auto`), respx + `unittest.mock.AsyncMock`.

**Spec:** `docs/superpowers/specs/2026-06-13-smart-turn-v2-turn-detection-design.md`

---

## File Structure

- **Create** `agent/turn_detector.py` — `TurnDetectorClient` (HTTP client to DGX `/v1/turn/classify`).
- **Modify** `agent/config.py` — five `turn_*` settings.
- **Modify** `agent/audio.py` — `VadBuffer.add_frame_candidate`, `continue_speech`, `at_cap` (existing `add_frame`/`_flush`/`reset` untouched).
- **Modify** `agent/ari.py` — constructor `turn_detector` param, `_bargein_buffers` dict, `_turn_active`, two-buffer `_setup_call`, `_teardown_call` cleanup, rewritten `_on_audio`, new `_gate_turn_end`.
- **Modify** `agent/main.py` — construct + inject `TurnDetectorClient`.
- **Create** `dgx/SMART_TURN.md` — pinned endpoint contract for the (separately built) DGX container.
- **Modify** `CLAUDE.md` — document the new turn-detection invariant.
- **Test (create)** `tests/test_turn_detector.py`.
- **Test (modify)** `tests/test_config.py`, `tests/test_audio.py`, `tests/test_ari.py`.

Run all tests with `venv/bin/pytest -v`.

---

## Task 1: Config fields

**Files:**
- Modify: `agent/config.py:73-83` (after the Tool / LLM safety block)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_turn_detection_defaults(settings):
    assert settings.turn_detection_enabled is False
    assert settings.turn_detector_url == "http://dgx-spark:8004"
    assert settings.turn_complete_threshold == 0.5
    assert settings.turn_classify_timeout_ms == 150
    assert settings.turn_vad_silence_ms == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_config.py::test_turn_detection_defaults -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'turn_detection_enabled'`

- [ ] **Step 3: Add the settings**

In `agent/config.py`, immediately after the `trusted_caller_set` property (line 83), before `# Agent behaviour`:

```python
    # Turn detection (Smart Turn v2 on DGX). Fail closed: off by default so the
    # legacy single-buffer 800ms silence path is unchanged until explicitly
    # enabled AND German precision has been verified live.
    turn_detection_enabled: bool = False
    turn_detector_url: str = "http://dgx-spark:8004"
    turn_complete_threshold: float = 0.5  # prob >= this => caller's turn is complete
    turn_classify_timeout_ms: int = 150  # latency budget; exceed => degrade to flush
    turn_vad_silence_ms: int = 200  # endpoint-candidate floor for the turn-end VAD
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_config.py::test_turn_detection_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat(config): add turn detection settings (flag off)"
```

---

## Task 2: VadBuffer candidate API

Adds a non-destructive endpoint-candidate path. `add_frame_candidate` returns the buffered speech (a fresh concatenated array) when the silence floor or the hard cap is reached, **without** resetting. `continue_speech` clears only the silence counter (retains frames) so a later pause re-triggers. `at_cap` exposes the hard limit. Existing `add_frame`/`force_flush`/`reset`/`_flush` are untouched (legacy + barge-in keep using them).

**Files:**
- Modify: `agent/audio.py:24-68` (inside `VadBuffer`)
- Test: `tests/test_audio.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_audio.py` (add `import numpy as np` and `from agent.audio import VadBuffer` at top if not present):

```python
class _FakeVad:
    """Scripted webrtcvad replacement: is_speech returns the next scripted bool."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def is_speech(self, _frame_bytes, _sample_rate):
        v = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return v


def test_add_frame_candidate_returns_without_reset():
    vad = VadBuffer(silence_threshold_ms=40, frame_ms=20)  # _silence_threshold = 2
    vad._vad = _FakeVad([True, True, False, False])
    f = np.zeros(320, dtype=np.int16)
    assert vad.add_frame_candidate(f) is None  # speech 1
    assert vad.add_frame_candidate(f) is None  # speech 2
    assert vad.add_frame_candidate(f) is None  # silence 1
    cand = vad.add_frame_candidate(f)          # silence 2 -> threshold
    assert cand is not None
    assert len(cand) == 320 * 4                # all frames buffered, no reset
    assert vad._in_speech is True              # buffer retained


def test_continue_speech_retains_buffer():
    vad = VadBuffer(silence_threshold_ms=40, frame_ms=20)  # _silence_threshold = 2
    vad._vad = _FakeVad([True, False, False, True, False, False])
    f = np.ones(320, dtype=np.int16)
    vad.add_frame_candidate(f)          # speech 1
    vad.add_frame_candidate(f)          # silence 1
    cand1 = vad.add_frame_candidate(f)  # silence 2 -> candidate (3 frames)
    assert cand1 is not None and len(cand1) == 320 * 3
    vad.continue_speech()               # keep listening
    vad.add_frame_candidate(f)          # speech 2 (frame 4)
    vad.add_frame_candidate(f)          # silence 1
    cand2 = vad.add_frame_candidate(f)  # silence 2 -> candidate (6 frames)
    assert cand2 is not None and len(cand2) == 320 * 6  # earlier frames retained


def test_at_cap_forces_candidate_without_silence():
    vad = VadBuffer(silence_threshold_ms=10000, frame_ms=20, max_speech_ms=60)  # cap = 3
    vad._vad = _FakeVad([True, True, True])
    f = np.zeros(320, dtype=np.int16)
    assert vad.at_cap is False
    assert vad.add_frame_candidate(f) is None
    assert vad.add_frame_candidate(f) is None
    cand = vad.add_frame_candidate(f)  # 3rd frame -> at cap
    assert cand is not None
    assert vad.at_cap is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_audio.py -k "candidate or continue_speech or at_cap" -v`
Expected: FAIL with `AttributeError: 'VadBuffer' object has no attribute 'add_frame_candidate'`

- [ ] **Step 3: Implement the methods**

In `agent/audio.py`, add to `VadBuffer` after `add_frame` (line 53), before `force_flush`:

```python
    @property
    def at_cap(self) -> bool:
        return len(self._speech_frames) >= self._max_speech_frames

    def add_frame_candidate(self, frame: np.ndarray) -> np.ndarray | None:
        # Like add_frame, but returns a COPY of the buffered speech without
        # resetting, so the caller (a turn detector) can decide whether the
        # turn is really over. Returns the candidate on silence floor OR hard
        # cap; the cap guarantees the continue_speech loop terminates.
        is_speech = self._vad.is_speech(frame.astype(np.int16).tobytes(), self._sample_rate)
        if is_speech:
            self._speech_frames.append(frame)
            self._silence_count = 0
            self._in_speech = True
        elif self._in_speech:
            self._speech_frames.append(frame)
            self._silence_count += 1
        if not self._in_speech:
            return None
        if self._silence_count >= self._silence_threshold or self.at_cap:
            return np.concatenate(self._speech_frames)
        return None

    def continue_speech(self) -> None:
        # Keep listening after an "incomplete" verdict: drop the silence count
        # but retain buffered frames so a later pause re-proposes a candidate.
        self._silence_count = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_audio.py -k "candidate or continue_speech or at_cap" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/audio.py tests/test_audio.py
git commit -m "feat(audio): add VadBuffer candidate/continue_speech for turn gating"
```

---

## Task 3: TurnDetectorClient

**Files:**
- Create: `agent/turn_detector.py`
- Test: `tests/test_turn_detector.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_turn_detector.py`:

```python
import httpx
import numpy as np
import pytest
import respx

from agent.turn_detector import TurnDetectorClient


@respx.mock
async def test_classify_complete_above_threshold():
    respx.post("http://td:8004/v1/turn/classify").mock(
        return_value=httpx.Response(200, json={"complete": True, "prob": 0.9})
    )
    c = TurnDetectorClient(base_url="http://td:8004", threshold=0.5)
    assert await c.classify(np.zeros(1600, dtype=np.int16)) is True
    await c.aclose()


@respx.mock
async def test_classify_incomplete_below_threshold():
    respx.post("http://td:8004/v1/turn/classify").mock(
        return_value=httpx.Response(200, json={"complete": False, "prob": 0.2})
    )
    c = TurnDetectorClient(base_url="http://td:8004", threshold=0.5)
    assert await c.classify(np.zeros(1600, dtype=np.int16)) is False
    await c.aclose()


@respx.mock
async def test_classify_raises_on_server_error():
    respx.post("http://td:8004/v1/turn/classify").mock(return_value=httpx.Response(500))
    c = TurnDetectorClient(base_url="http://td:8004")
    with pytest.raises(httpx.HTTPStatusError):
        await c.classify(np.zeros(1600, dtype=np.int16))
    await c.aclose()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_turn_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.turn_detector'`

- [ ] **Step 3: Implement the client**

Create `agent/turn_detector.py`:

```python
import httpx
import numpy as np


class TurnDetectorClient:
    """Calls a DGX Smart Turn v2 service to classify end-of-turn.

    Mirrors SttClient's ownership model: reuse an injected long-lived client
    (and its pool); only close one we created ourselves.
    """

    def __init__(
        self,
        base_url: str,
        threshold: float = 0.5,
        timeout_ms: int = 150,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._threshold = threshold
        self._timeout_s = timeout_ms / 1000.0
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def classify(self, pcm_16k: np.ndarray) -> bool:
        """True = caller's turn is complete. Raises on timeout / HTTP error."""
        body = pcm_16k.astype(np.int16).tobytes()
        resp = await self._client.post(
            f"{self._base_url}/v1/turn/classify",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        return float(resp.json()["prob"]) >= self._threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_turn_detector.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/turn_detector.py tests/test_turn_detector.py
git commit -m "feat(turn): add TurnDetectorClient for DGX Smart Turn v2"
```

---

## Task 4: AriClient — constructor, buffers, setup/teardown

Wires the optional detector and the second (barge-in) buffer. No gating logic yet.

**Files:**
- Modify: `agent/ari.py` — imports, `__init__` (43-63), `_setup_call` (143), `_teardown_call` (226)
- Test: `tests/test_ari.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ari.py` (top already imports `AsyncMock, MagicMock`; add `import numpy as np` and `import httpx` near the other imports):

```python
@pytest.fixture
def ari_td(settings):
    settings.turn_detection_enabled = True
    pipeline = AsyncMock(return_value=b"\xd5" * 160)
    td = AsyncMock()
    ari = AriClient(settings=settings, pipeline=pipeline, turn_detector=td)
    return ari, td


def test_turn_active_reflects_flag_and_detector(ari_td, settings):
    ari, _td = ari_td
    assert ari._turn_active() is True
    # No detector => inactive even with the flag on.
    settings.turn_detection_enabled = True
    ari_no_td = AriClient(settings=settings, pipeline=AsyncMock(), turn_detector=None)
    assert ari_no_td._turn_active() is False


async def test_teardown_pops_bargein_buffer(ari):
    session = CallSession(
        call_id="ch-1", caller_id="+49", history=[], created_at=datetime.now(timezone.utc)
    )
    ari._sessions["ch-1"] = session
    ari._bargein_buffers["ch-1"] = MagicMock()
    await ari._teardown_call("ch-1")
    assert "ch-1" not in ari._bargein_buffers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_ari.py -k "turn_active or bargein_buffer" -v`
Expected: FAIL — `AriClient.__init__() got an unexpected keyword argument 'turn_detector'`

- [ ] **Step 3: Add the import**

In `agent/ari.py`, near the other `from agent...` imports (around line 11), add:

```python
from agent.turn_detector import TurnDetectorClient
```

- [ ] **Step 4: Extend the constructor**

In `agent/ari.py`, change the signature at line 43 and add fields. Replace:

```python
    def __init__(self, settings: Settings, pipeline: VoicePipeline) -> None:
        self._s = settings
        self._pipeline = pipeline
        self._sessions: dict[str, CallSession] = {}
        self._rtp_servers: dict[str, RtpServer] = {}
        self._vad_buffers: dict[str, VadBuffer] = {}
```

with:

```python
    def __init__(
        self,
        settings: Settings,
        pipeline: VoicePipeline,
        turn_detector: TurnDetectorClient | None = None,
    ) -> None:
        self._s = settings
        self._pipeline = pipeline
        self._turn_detector = turn_detector
        self._sessions: dict[str, CallSession] = {}
        self._rtp_servers: dict[str, RtpServer] = {}
        self._vad_buffers: dict[str, VadBuffer] = {}
        # Second buffer used only when turn detection is active: barge-in
        # (SPEAKING/PROCESSING) keeps the legacy 800ms floor here while the
        # primary buffer runs the lowered turn-end floor.
        self._bargein_buffers: dict[str, VadBuffer] = {}
```

- [ ] **Step 5: Add the `_turn_active` helper**

In `agent/ari.py`, add right after `_client` (after line 68):

```python
    def _turn_active(self) -> bool:
        return self._turn_detector is not None and self._s.turn_detection_enabled
```

- [ ] **Step 6: Two-buffer setup**

In `agent/ari.py`, replace line 143 (`self._vad_buffers[channel_id] = VadBuffer()`) with:

```python
            if self._turn_active():
                self._vad_buffers[channel_id] = VadBuffer(
                    silence_threshold_ms=self._s.turn_vad_silence_ms
                )
                self._bargein_buffers[channel_id] = VadBuffer()
            else:
                self._vad_buffers[channel_id] = VadBuffer()
```

- [ ] **Step 7: Teardown cleanup**

In `agent/ari.py` `_teardown_call`, after the `self._vad_buffers.pop(channel_id, None)` line (line 226), add:

```python
        self._bargein_buffers.pop(channel_id, None)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_ari.py -k "turn_active or bargein_buffer" -v`
Expected: PASS (2 passed)

- [ ] **Step 9: Commit**

```bash
git add agent/ari.py tests/test_ari.py
git commit -m "feat(ari): wire turn detector + barge-in buffer (setup/teardown)"
```

---

## Task 5: `_on_audio` gating + `_gate_turn_end`

The behavioral core. In LISTENING with turn detection active, route through the candidate path + classifier. Otherwise (flag off, or barge-in in SPEAKING/PROCESSING) keep the legacy `add_frame` path. The lock/generation/dispatch block is unchanged.

**Files:**
- Modify: `agent/ari.py:306-357` (`_on_audio`) + new `_gate_turn_end`, new class constant
- Test: `tests/test_ari.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ari.py`:

```python
def _listening_session():
    s = CallSession(
        call_id="ch-1", caller_id="+49123", history=[], created_at=datetime.now(timezone.utc)
    )
    s.transition(SessionState.LISTENING)
    return s


async def test_turn_gate_complete_dispatches(ari_td):
    ari, td = ari_td
    td.classify = AsyncMock(return_value=True)
    ari._sessions["ch-1"] = _listening_session()
    ari._generation["ch-1"] = 0
    ari._turn_locks["ch-1"] = asyncio.Lock()
    vad = MagicMock()
    vad.add_frame_candidate = MagicMock(return_value=np.ones(2000, dtype=np.int16))
    vad.at_cap = False
    ari._vad_buffers["ch-1"] = vad
    ari._play_stream = AsyncMock()

    await ari._on_audio("ch-1", b"\xd5" * 160)
    await asyncio.sleep(0)

    td.classify.assert_awaited_once()
    ari._play_stream.assert_awaited_once()
    assert ari._generation["ch-1"] == 1


async def test_turn_gate_incomplete_keeps_listening(ari_td):
    ari, td = ari_td
    td.classify = AsyncMock(return_value=False)
    session = _listening_session()
    ari._sessions["ch-1"] = session
    ari._turn_locks["ch-1"] = asyncio.Lock()
    vad = MagicMock()
    vad.add_frame_candidate = MagicMock(return_value=np.ones(2000, dtype=np.int16))
    vad.at_cap = False
    ari._vad_buffers["ch-1"] = vad
    ari._play_stream = AsyncMock()

    await ari._on_audio("ch-1", b"\xd5" * 160)
    await asyncio.sleep(0)

    td.classify.assert_awaited_once()
    vad.continue_speech.assert_called_once()
    ari._play_stream.assert_not_awaited()
    assert session.state == SessionState.LISTENING


async def test_turn_gate_degrades_to_flush_on_error(ari_td):
    ari, td = ari_td
    td.classify = AsyncMock(side_effect=httpx.ConnectError("boom"))
    ari._sessions["ch-1"] = _listening_session()
    ari._generation["ch-1"] = 0
    ari._turn_locks["ch-1"] = asyncio.Lock()
    vad = MagicMock()
    vad.add_frame_candidate = MagicMock(return_value=np.ones(2000, dtype=np.int16))
    vad.at_cap = False
    ari._vad_buffers["ch-1"] = vad
    ari._play_stream = AsyncMock()

    await ari._on_audio("ch-1", b"\xd5" * 160)
    await asyncio.sleep(0)

    ari._play_stream.assert_awaited_once()  # flushed despite error


async def test_turn_gate_cap_flushes_without_classify(ari_td):
    ari, td = ari_td
    td.classify = AsyncMock(return_value=False)
    ari._sessions["ch-1"] = _listening_session()
    ari._generation["ch-1"] = 0
    ari._turn_locks["ch-1"] = asyncio.Lock()
    vad = MagicMock()
    vad.add_frame_candidate = MagicMock(return_value=np.ones(2000, dtype=np.int16))
    vad.at_cap = True
    ari._vad_buffers["ch-1"] = vad
    ari._play_stream = AsyncMock()

    await ari._on_audio("ch-1", b"\xd5" * 160)
    await asyncio.sleep(0)

    td.classify.assert_not_awaited()
    ari._play_stream.assert_awaited_once()


async def test_turn_gate_discards_if_state_changed_during_await(ari_td):
    ari, td = ari_td
    session = _listening_session()

    async def _classify(_pcm):
        session.transition(SessionState.PROCESSING)  # state moves mid-await
        return True

    td.classify = AsyncMock(side_effect=_classify)
    ari._sessions["ch-1"] = session
    ari._generation["ch-1"] = 0
    ari._turn_locks["ch-1"] = asyncio.Lock()
    vad = MagicMock()
    vad.add_frame_candidate = MagicMock(return_value=np.ones(2000, dtype=np.int16))
    vad.at_cap = False
    ari._vad_buffers["ch-1"] = vad
    ari._play_stream = AsyncMock()

    await ari._on_audio("ch-1", b"\xd5" * 160)
    await asyncio.sleep(0)

    ari._play_stream.assert_not_awaited()


async def test_bargein_skips_turn_detector(ari_td):
    ari, td = ari_td
    td.classify = AsyncMock(return_value=True)
    session = _listening_session()
    session.transition(SessionState.PROCESSING)  # barge-in window
    ari._sessions["ch-1"] = session
    ari._generation["ch-1"] = 1
    ari._turn_locks["ch-1"] = asyncio.Lock()
    bvad = MagicMock()
    bvad.add_frame = MagicMock(return_value=np.ones(800, dtype=np.int16))
    ari._bargein_buffers["ch-1"] = bvad
    ari._play_stream = AsyncMock()

    await ari._on_audio("ch-1", b"\xd5" * 160)
    await asyncio.sleep(0)

    td.classify.assert_not_awaited()
    ari._play_stream.assert_awaited_once()
    assert ari._generation["ch-1"] == 2


async def test_turn_detection_disabled_uses_legacy_add_frame(ari):
    # ari fixture: turn_detection_enabled False, no turn_detector => legacy path.
    ari._sessions["ch-1"] = _listening_session()
    ari._generation["ch-1"] = 0
    ari._turn_locks["ch-1"] = asyncio.Lock()
    vad = MagicMock()
    vad.add_frame = MagicMock(return_value=np.ones(800, dtype=np.int16))
    ari._vad_buffers["ch-1"] = vad
    ari._play_stream = AsyncMock()

    await ari._on_audio("ch-1", b"\xd5" * 160)
    await asyncio.sleep(0)

    vad.add_frame.assert_called_once()
    ari._play_stream.assert_awaited_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_ari.py -k "turn_gate or bargein_skips or disabled_uses_legacy" -v`
Expected: FAIL (the gating branch + `_gate_turn_end` don't exist yet; e.g. `add_frame_candidate` never called, classify not awaited, `_MIN_CLASSIFY_SAMPLES` missing).

- [ ] **Step 3: Add the class constant**

In `agent/ari.py`, alongside the other class constants near `_EXT_PREFIX` (line 32), add:

```python
    # Below this many samples a candidate is too short to be worth a classify
    # round-trip; flush it (fail toward responding). 1600 = 100 ms at 16 kHz.
    _MIN_CLASSIFY_SAMPLES = 1600
```

- [ ] **Step 4: Rewrite `_on_audio`**

In `agent/ari.py`, replace the whole body of `_on_audio` (lines 306-357) with:

```python
    async def _on_audio(self, channel_id: str, alaw_payload: bytes) -> None:
        session = self._sessions.get(channel_id)
        if not session:
            return

        if session.state not in self._INTERRUPTIBLE_STATES:
            return

        pcm_8k = alaw_decode(alaw_payload)
        pcm_16k = resample_8k_to_16k(pcm_8k)

        if self._turn_active() and session.state == SessionState.LISTENING:
            # Turn-end path: low VAD floor proposes a candidate, Smart Turn v2
            # confirms. _gate_turn_end returns the speech to flush, or None to
            # keep listening (incomplete / state changed mid-classify).
            vad = self._vad_buffers.get(channel_id)
            if vad is None:
                return
            candidate = vad.add_frame_candidate(pcm_16k)
            if candidate is None:
                return
            speech = await self._gate_turn_end(channel_id, vad, candidate)
            if speech is None:
                return
        else:
            # Legacy / barge-in path: unchanged 800ms silence flush. When turn
            # detection is active the barge-in buffer is separate so SPEAKING/
            # PROCESSING interrupts keep their original timing.
            vad = (
                self._bargein_buffers.get(channel_id)
                if self._turn_active()
                else self._vad_buffers.get(channel_id)
            )
            if vad is None:
                return
            speech = vad.add_frame(pcm_16k)
            if speech is None:
                return

        lock = self._turn_locks.get(channel_id)
        if lock is None:
            return
        async with lock:
            # State may have advanced while we waited for the lock.
            if session.state not in self._INTERRUPTIBLE_STATES:
                return

            # Bump the generation: any in-flight playback for the prior
            # generation will now no-op on completion, and the turn we start
            # below tags its own audio with this id.
            gen = self._generation.get(channel_id, 0) + 1
            self._generation[channel_id] = gen

            # Tear down the in-flight turn (streaming producer chain or a
            # greeting playback). The bumped generation already neutralizes any
            # late chunk; the cancel stops the work immediately.
            task = self._playback_tasks.pop(channel_id, None)
            if task and not task.done():
                task.cancel()

            # Move back to LISTENING before the new turn enters PROCESSING.
            # Both SPEAKING and the now-interruptible PROCESSING are valid
            # sources for this transition.
            if session.state in (SessionState.SPEAKING, SessionState.PROCESSING):
                session.transition(SessionState.LISTENING)
            vad.reset()

            # Streaming turn: process_turn_stream owns PROCESSING entry,
            # _play_stream drives SPEAKING (first chunk) -> LISTENING. The
            # stale-generation guard now lives inside _play_stream.
            task = asyncio.create_task(self._play_stream(channel_id, session, gen, speech))
            self._playback_tasks[channel_id] = task

    async def _gate_turn_end(self, channel_id, vad, candidate):
        # Returns the speech buffer to flush, or None to keep listening.
        # Hard cap first: never let the incomplete loop run past max_speech.
        if vad.at_cap:
            return candidate
        # Too short to classify: flush (fail toward responding).
        if len(candidate) < self._MIN_CLASSIFY_SAMPLES:
            return candidate
        try:
            complete = await self._turn_detector.classify(candidate)
        except Exception:
            # Degrade: model slow/unavailable -> flush on the candidate.
            log.warning("Turn classify failed for %s; flushing (degrade)", channel_id)
            return candidate
        # State may have changed while we awaited the verdict (e.g. hangup).
        session = self._sessions.get(channel_id)
        if session is None or session.state != SessionState.LISTENING:
            return None
        if complete:
            return candidate
        # Incomplete: caller paused mid-thought. Keep the buffer, keep listening.
        vad.continue_speech()
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_ari.py -k "turn_gate or bargein_skips or disabled_uses_legacy" -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Run the full ARI + audio suites (no regression)**

Run: `venv/bin/pytest tests/test_ari.py tests/test_audio.py -v`
Expected: PASS (all, including the pre-existing `test_bargein_during_processing_cancels_and_starts_stream`)

- [ ] **Step 7: Commit**

```bash
git add agent/ari.py tests/test_ari.py
git commit -m "feat(ari): gate turn-end on Smart Turn v2, keep barge-in legacy"
```

---

## Task 6: Wire into main.py

**Files:**
- Modify: `agent/main.py:12-15` (imports), `agent/main.py:37-59` (construction)

- [ ] **Step 1: Add the import**

In `agent/main.py`, after `from agent.tts import TtsClient` (line 15), add:

```python
from agent.turn_detector import TurnDetectorClient
```

- [ ] **Step 2: Construct and inject the detector**

In `agent/main.py`, after the `tts = TtsClient(...)` line (line 38), add:

```python
    turn_detector = TurnDetectorClient(
        base_url=s.turn_detector_url,
        threshold=s.turn_complete_threshold,
        timeout_ms=s.turn_classify_timeout_ms,
        client=http_client,
    )
```

Then change the `ari = AriClient(...)` line (line 59) to:

```python
    ari = AriClient(settings=s, pipeline=pipeline, turn_detector=turn_detector)
```

(No extra `aclose` needed: `turn_detector` reuses the injected `http_client`, which `main()` already closes in its `finally`.)

- [ ] **Step 3: Import sanity check**

Run: `venv/bin/python -c "from agent.main import main; print('OK')"`
Expected: prints `OK`

- [ ] **Step 4: Commit**

```bash
git add agent/main.py
git commit -m "feat(main): construct and inject TurnDetectorClient"
```

---

## Task 7: Pin the DGX endpoint contract

The inference container is built separately; pin the wire contract so the consumer and producer can't drift.

**Files:**
- Create: `dgx/SMART_TURN.md`

- [ ] **Step 1: Write the contract doc**

Create `dgx/SMART_TURN.md`:

```markdown
# Smart Turn v2 — DGX service contract

Consumer: `agent/turn_detector.py` (`TurnDetectorClient`). Model:
`pipecat-ai/smart-turn-v2` (wav2vec2-based end-of-turn classifier, multilingual
incl. German). Build/deploy of this container is tracked separately; this file
pins the wire contract the agent depends on.

## Endpoint

`POST /v1/turn/classify`

- Request body: raw little-endian **int16 PCM, mono, 16 kHz** (no WAV header).
  Content-Type `application/octet-stream`. Represents the caller's buffered
  speech since their last endpoint candidate.
- Response: `200` JSON `{"complete": bool, "prob": float}` where `prob` is the
  probability in `[0,1]` that the turn is complete. The client compares `prob`
  against `turn_complete_threshold` (default 0.5); `complete` is advisory.
- Latency: must answer within the client budget (`turn_classify_timeout_ms`,
  default 150 ms) or the client times out and degrades to a silence flush.

## Default port

`8004` (see `turn_detector_url` default `http://dgx-spark:8004`). Add the
container to `dgx/docker-compose.yml` analogous to the TTS service, with a
health check.

## Rollout gate

`turn_detection_enabled` stays `False` until German precision is verified live
against this service (multilingual != verified for German — cf. the TTS
`language` full-names lesson).
```

- [ ] **Step 2: Commit**

```bash
git add dgx/SMART_TURN.md
git commit -m "docs(dgx): pin Smart Turn v2 endpoint contract"
```

---

## Task 8: Document the invariant in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (Architecture invariants + Key design decisions)

- [ ] **Step 1: Add a turn-detection bullet to "Architecture invariants"**

In `CLAUDE.md`, after the **Barge-in** bullet, add:

```markdown
- **Turn detection (Smart Turn v2, opt-in):** When `turn_detection_enabled` and a `TurnDetectorClient` is injected, the LISTENING turn-end runs a lowered VAD floor (`turn_vad_silence_ms`, default 200 ms) via `VadBuffer.add_frame_candidate` (returns a candidate **without** resetting) and confirms with a DGX classify call. `complete` (or hard cap `at_cap`, or candidate < `_MIN_CLASSIFY_SAMPLES`, or a classify error → **degrade**) flushes and dispatches; `incomplete` calls `continue_speech()` and keeps listening (bounded by `max_speech_ms`). Barge-in is **unchanged**: SPEAKING/PROCESSING use a separate `_bargein_buffers` VadBuffer at the legacy 800 ms floor. Flag off → single VadBuffer at 800 ms for every state (legacy path verbatim). `classify` runs **before** the turn lock; if state leaves LISTENING during the await, the verdict is discarded.
```

- [ ] **Step 2: Add a key design decision**

In `CLAUDE.md` under "Key design decisions", add:

```markdown
- **Smart Turn v2 fail-closed:** turn detection is off by default. German precision is unverified (multilingual ≠ verified-for-German — see the TTS language lesson); verify live before enabling. The DGX endpoint contract is pinned in `dgx/SMART_TURN.md`; tests mock the client/server boundary, so wire-verify the real container before flipping the flag.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): document Smart Turn v2 turn-detection invariant"
```

---

## Task 9: Full suite + lint gate

- [ ] **Step 1: Run the entire test suite**

Run: `venv/bin/pytest -v`
Expected: PASS (all tests, including the new turn-detection ones; legacy ARI/audio tests unchanged).

- [ ] **Step 2: Format check (CI gate)**

Run: `venv/bin/ruff format --check agent/ tests/` then `venv/bin/ruff check agent/ tests/`
Expected: no changes needed / no lint errors. If `format --check` reports files, run `venv/bin/ruff format agent/ tests/` and commit:

```bash
git add -A
git commit -m "style: ruff format turn-detection changes"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** topology (Task 6 + 7 record the DGX choice), gating low-floor+confirm (Task 5), degrade + hard cap + min-frames (Task 5 `_gate_turn_end`), barge-in unchanged via second buffer (Tasks 4-5), feature flag default off (Task 1), all error/edge cases and the full test matrix from the spec (Tasks 2-5). German-precision caveat documented (Tasks 7-8).
- **Type/name consistency:** `add_frame_candidate`, `continue_speech`, `at_cap`, `_turn_active`, `_gate_turn_end`, `_MIN_CLASSIFY_SAMPLES`, `_bargein_buffers`, `TurnDetectorClient.classify(threshold/timeout_ms)` used identically across tasks and tests.
- **DGX container build** is intentionally out of scope (contract pinned in Task 7).
```
