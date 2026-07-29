# Continuous PCM Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace chunk-reset A-law playback with continuous 24-to-16 kHz PCM streaming, bounded prebuffering, and sentence prefetch while preserving barge-in and the legacy ARI rollback path.

**Architecture:** `VoicePipeline` becomes transport-neutral at signed-16-bit mono 16 kHz PCM. A stateful `audioop.ratecv` adapter preserves conversion state across arbitrary TTS network chunks; PJSIP writes that PCM directly, while ARI alone converts it to stateful 8 kHz A-law. A 300 ms PJSIP prebuffer and a two-entry sentence queue absorb generation jitter without running simultaneous model inference.

**Tech Stack:** Python 3.12 asyncio, NumPy, audioop-lts, PJSUA2, scipy reference tests, pytest, Ruff, Docker Compose.

## Global Constraints

- Work only in `/home/volsch/projekte/.worktrees/voip-agent-shared-ai-traefik` on `feat/shared-ai-traefik`.
- Follow strict red-green-refactor: every production behavior starts with a test that fails for the expected reason.
- Production PJSIP output is little-endian signed-16-bit mono 16 kHz PCM.
- One resampler instance belongs to exactly one logical utterance or ARI stream.
- PJSIP playback prebuffer is 300 ms, exactly 9600 PCM bytes.
- Sentence prefetch depth is two completed segments and preserves text/audio order.
- No simultaneous TTS model generations, new dependency, voice-profile change, inbound-audio change, or SIP configuration change.
- Barge-in and teardown cancel all producer work and clear all buffered PCM.
- The legacy ARI/RTP boundary remains 8 kHz G.711 A-law and keeps its existing pacing contract.
- Never modify or print secrets, caller IDs, private reference audio, or private profile metadata.

---

### Task 1: Stateful PCM16 resampler

**Files:**
- Modify: `agent/audio.py`
- Modify: `tests/test_audio.py`

**Interfaces:**
- Consumes: one-dimensional NumPy `int16` chunks and explicit integer sample rates.
- Produces: `StreamingPcm16Resampler(input_rate: int, output_rate: int)`, `process(samples: np.ndarray) -> bytes`, `close() -> None`, and `resample_pcm16(samples, input_rate, output_rate) -> bytes`.

- [ ] **Step 1: Write failing split-invariance and lifecycle tests**

Add literal fixtures to `tests/test_audio.py`:

```python
def test_streaming_pcm_resampler_matches_one_shot_across_arbitrary_splits():
    samples = np.arange(-5000, 5000, dtype=np.int16)
    one_shot = resample_pcm16(samples, 24_000, 16_000)
    stream = StreamingPcm16Resampler(24_000, 16_000)
    chunks = [samples[:1], samples[1:101], samples[101:4097], samples[4097:]]
    chunked = b"".join(stream.process(chunk) for chunk in chunks)
    stream.close()
    assert chunked == one_shot
    assert len(chunked) == 13_334


def test_streaming_pcm_resampler_rejects_invalid_or_closed_input():
    stream = StreamingPcm16Resampler(24_000, 16_000)
    with pytest.raises(ValueError, match="one-dimensional int16"):
        stream.process(np.zeros((2, 2), dtype=np.int16))
    stream.close()
    with pytest.raises(RuntimeError, match="closed"):
        stream.process(np.zeros(3, dtype=np.int16))
```

The production mutation these tests catch is resetting `ratecv` state for each chunk or accepting ambiguous PCM layout.

- [ ] **Step 2: Run the tests and verify red**

Run:

```bash
venv/bin/pytest tests/test_audio.py -q
```

Expected: import failure because `StreamingPcm16Resampler` and `resample_pcm16` do not exist.

- [ ] **Step 3: Implement the minimal stateful adapter**

In `agent/audio.py`, retain the existing inbound and legacy helpers and add:

```python
class StreamingPcm16Resampler:
    def __init__(self, input_rate: int, output_rate: int) -> None:
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("sample rates must be positive")
        self._input_rate = input_rate
        self._output_rate = output_rate
        self._state = None
        self._closed = False

    def process(self, samples: np.ndarray) -> bytes:
        if self._closed:
            raise RuntimeError("resampler is closed")
        if samples.ndim != 1 or samples.dtype != np.int16:
            raise ValueError("PCM must be one-dimensional int16")
        output, self._state = audioop.ratecv(
            samples.astype("<i2", copy=False).tobytes(),
            2,
            1,
            self._input_rate,
            self._output_rate,
            self._state,
        )
        return output

    def close(self) -> None:
        self._closed = True
        self._state = None


def resample_pcm16(samples: np.ndarray, input_rate: int, output_rate: int) -> bytes:
    stream = StreamingPcm16Resampler(input_rate, output_rate)
    try:
        return stream.process(samples)
    finally:
        stream.close()
```

- [ ] **Step 4: Verify green and existing audio behavior**

Run:

```bash
venv/bin/pytest tests/test_audio.py -q
ruff check agent/audio.py tests/test_audio.py
ruff format --check agent/audio.py tests/test_audio.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent/audio.py tests/test_audio.py
git commit -m "feat(audio): add stateful PCM resampling"
```

---

### Task 2: Convert the voice pipeline contract to 16 kHz PCM

**Files:**
- Modify: `agent/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: TTS WAV or streamed NumPy PCM at 24 kHz.
- Produces: `synthesize_pcm16(text: str) -> bytes`, `process_turn(...) -> bytes`, `_tts_pcm16_chunks(text) -> AsyncIterator[bytes]`, and `process_turn_stream(...) -> AsyncIterator[bytes]`, all outbound values signed-16-bit mono 16 kHz PCM.

- [ ] **Step 1: Write failing PCM contract and continuity tests**

Replace A-law expectations with hand-derived PCM behavior:

```python
def _wav_samples(samples: np.ndarray, rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2", copy=False).tobytes())
    return buffer.getvalue()


async def test_synthesize_pcm16_returns_direct_16k_pcm():
    source = np.arange(24_000, dtype=np.int16)
    pipeline = VoicePipeline(
        stt=AsyncMock(),
        llm=AsyncMock(),
        tts=AsyncMock(return_value=_wav_samples(source)),
    )
    result = await pipeline.synthesize_pcm16("Hallo")
    assert result == resample_pcm16(source, 24_000, 16_000)


async def test_tts_pcm16_chunks_preserve_resampler_state():
    source = np.arange(-3000, 3000, dtype=np.int16)

    async def tts_stream(_text):
        yield source[:101]
        yield source[101:4097]
        yield source[4097:]

    pipeline = VoicePipeline(
        stt=AsyncMock(),
        llm=AsyncMock(),
        tts=AsyncMock(),
        tts_stream=tts_stream,
    )
    result = b"".join([chunk async for chunk in pipeline._tts_pcm16_chunks("Hallo")])
    assert result == resample_pcm16(source, 24_000, 16_000)
```

The production mutations caught are restoring per-chunk `resample_poly`, A-law encoding, or losing conversion state.

- [ ] **Step 2: Run focused tests and verify red**

```bash
venv/bin/pytest tests/test_pipeline.py -q
```

Expected: `synthesize_pcm16` and `_tts_pcm16_chunks` are absent or existing A-law assertions fail.

- [ ] **Step 3: Implement the PCM pipeline**

- Rename `synthesize_alaw` to `synthesize_pcm16`.
- Convert complete 24 kHz WAV samples with `resample_pcm16`.
- Replace `_tts_alaw_chunks` with `_tts_pcm16_chunks`; create one
  `StreamingPcm16Resampler(24_000, 16_000)` before the `async for`, yield only
  non-empty bytes, and close it in `finally`.
- Keep `process_turn` and every fallback on the new PCM contract.
- Keep `process_turn_stream` orchestration and state transitions unchanged in
  this task; only replace A-law variable names and calls.
- Remove outbound `alaw_encode` and `resample_24k_to_8k` imports from
  `agent/pipeline.py`.

- [ ] **Step 4: Verify green**

```bash
venv/bin/pytest tests/test_pipeline.py -q
ruff check agent/pipeline.py tests/test_pipeline.py
ruff format --check agent/pipeline.py tests/test_pipeline.py
```

Expected: all pipeline tests pass and no test asserts on a mock instead of returned PCM.

- [ ] **Step 5: Commit**

```bash
git add agent/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): emit continuous PCM16"
```

---

### Task 3: Direct PJSIP PCM playback with a 300 ms prebuffer

**Files:**
- Modify: `agent/pjsip.py`
- Modify: `agent/conversation.py`
- Modify: `tests/test_pjsip.py`
- Modify: `tests/test_conversation.py`

**Interfaces:**
- Consumes: even-length 16 kHz PCM byte chunks from `VoicePipeline`.
- Produces: `AudioSink.play_pcm16(pcm: bytes)` and `AudioSink.play_pcm16_chunks(queue)`, direct PJSIP buffer writes, and a 9600-byte initial prebuffer.

- [ ] **Step 1: Write failing direct-write and prebuffer tests**

Add tests using the real `PcmPlaybackBuffer`:

```python
async def test_pjsip_sink_writes_pcm_without_alaw_roundtrip():
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    pcm = np.arange(320, dtype="<i2").tobytes()
    playback = asyncio.create_task(sink.play_pcm16(pcm))
    await asyncio.sleep(0)
    assert buffer.read(len(pcm)) == pcm
    await playback


async def test_stream_playback_waits_for_300ms_prebuffer():
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    queue = asyncio.Queue()
    task = asyncio.create_task(sink.play_pcm16_chunks(queue))
    await queue.put(b"\x01\x00" * 4_799)
    await asyncio.sleep(0)
    assert buffer.buffered_bytes == 0
    await queue.put(b"\x02\x00")
    await asyncio.sleep(0)
    assert buffer.buffered_bytes == 9_600
    buffer.read(9_600)
    await queue.put(None)
    await task


async def test_stream_playback_releases_short_response_at_end_of_stream():
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    queue = asyncio.Queue()
    await queue.put(b"\x01\x00" * 1_000)
    await queue.put(None)
    task = asyncio.create_task(sink.play_pcm16_chunks(queue))
    await asyncio.sleep(0)
    assert buffer.buffered_bytes == 2_000
    buffer.read(2_000)
    await task
```

Add an odd-byte rejection test. Update the conversation fake sink to expose
`play_pcm16` and `play_pcm16_chunks`; assert cancellation clears it.

- [ ] **Step 2: Run focused tests and verify red**

```bash
venv/bin/pytest tests/test_pjsip.py tests/test_conversation.py -q
```

Expected: missing PCM sink methods and old A-law expansion behavior.

- [ ] **Step 3: Implement direct playback and prebuffer**

- Remove `_alaw_to_pcm16` from `agent/pjsip.py`.
- Add `PREBUFFER_BYTES = 16_000 * 2 * 300 // 1000`.
- Validate every PCM write has an even byte length.
- Rename sink methods to `play_pcm16` and `play_pcm16_chunks`.
- In `play_pcm16_chunks`, accumulate bytes until `PREBUFFER_BYTES` or the
  `None` sentinel, write the accumulated bytes once, then continue with
  existing backpressure and drain behavior.
- Rename the `AudioSink` protocol and `ConversationManager` calls/variables to
  PCM16. Call `pipeline.synthesize_pcm16` for the greeting.
- Preserve cancellation-driven `clear()` behavior.

- [ ] **Step 4: Verify green**

```bash
venv/bin/pytest tests/test_pjsip.py tests/test_conversation.py -q
ruff check agent/pjsip.py agent/conversation.py tests/test_pjsip.py tests/test_conversation.py
ruff format --check agent/pjsip.py agent/conversation.py tests/test_pjsip.py tests/test_conversation.py
```

- [ ] **Step 5: Commit**

```bash
git add agent/pjsip.py agent/conversation.py tests/test_pjsip.py tests/test_conversation.py
git commit -m "fix(pjsip): prebuffer direct PCM playback"
```

---

### Task 4: Bound sentence prefetch and cancellation

**Files:**
- Modify: `agent/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_conversation.py`

**Interfaces:**
- Consumes: the existing LLM token stream and tool-round callback.
- Produces: an internal `asyncio.Queue(maxsize=2)` of ordered sentence/filler events and one sequential TTS consumer.

- [ ] **Step 1: Write failing ordering, look-ahead, and cancellation tests**

Use a real async LLM generator and a controlled TTS generator:

```python
async def test_sentence_prefetch_is_bounded_and_preserves_order():
    first_tts_started = asyncio.Event()
    release_first = asyncio.Event()
    consumed = []
    tts_calls = []
    tokens = ["Eins. ", "Zwei. ", "Drei. ", "Vier. ", "Fünf. ", "Sechs."]

    async def llm_stream(*_args, **_kwargs):
        for token in tokens:
            consumed.append(token)
            yield token

    async def tts_stream(text):
        tts_calls.append(text)
        if text == "Eins.":
            first_tts_started.set()
            await release_first.wait()
        yield np.full(240, len(tts_calls), dtype=np.int16)

    pipeline = VoicePipeline(
        stt=AsyncMock(return_value="frage"),
        llm=None,
        tts=None,
        llm_stream=llm_stream,
        tts_stream=tts_stream,
    )
    session = _strm_session()
    session.transition(SessionState.LISTENING)
    stream = pipeline.process_turn_stream(session, _pcm_zero())
    first = asyncio.create_task(anext(stream))
    await first_tts_started.wait()
    await asyncio.sleep(0)
    assert tts_calls == ["Eins."]
    assert len(consumed) <= 4
    assert len(consumed) < len(tokens)
    release_first.set()
    output = [await first] + [chunk async for chunk in stream]
    assert tts_calls == ["Eins.", "Zwei.", "Drei.", "Vier.", "Fünf.", "Sechs."]
    assert b"".join(output)


async def test_cancelled_turn_closes_prefetch_and_tts_tasks():
    closed = asyncio.Event()

    async def llm_stream(*_args, **_kwargs):
        yield "Eins."

    async def tts_stream(_text):
        try:
            while True:
                await asyncio.sleep(1)
                yield np.ones(240, dtype=np.int16)
        finally:
            closed.set()

    pipeline = VoicePipeline(
        stt=AsyncMock(return_value="frage"),
        llm=None,
        tts=None,
        llm_stream=llm_stream,
        tts_stream=tts_stream,
    )
    session = _strm_session()
    session.transition(SessionState.LISTENING)
    stream = pipeline.process_turn_stream(session, _pcm_zero())
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await stream.aclose()
    assert closed.is_set()
```

The mutations caught are unbounded sentence accumulation, concurrent/out-of-order TTS, and leaked work after barge-in.

- [ ] **Step 2: Run focused tests and verify red**

```bash
venv/bin/pytest tests/test_pipeline.py tests/test_conversation.py -q
```

Expected: the current token consumer blocks segmentation during TTS or cancellation leaves a producer alive.

- [ ] **Step 3: Implement the bounded producer**

- Add `SENTENCE_PREFETCH_DEPTH = 2`.
- Run LLM token consumption and sentence segmentation in one producer task.
- Put `("text", sentence)` entries into `asyncio.Queue(maxsize=2)`.
- Put exactly one filler entry when the tool-round callback fires.
- Finish with a `done` sentinel; translate producer failure to an `error`
  sentinel.
- The existing async generator remains the sole TTS consumer, so model
  generations remain ordered and sequential.
- In `finally`, cancel and await the segment producer and any pending filler
  task. Closing the current `_tts_pcm16_chunks` generator must close the HTTP
  stream through `TtsClient`.

- [ ] **Step 4: Verify green and mutation coverage**

```bash
venv/bin/pytest tests/test_pipeline.py tests/test_conversation.py -q
ruff check agent/pipeline.py tests/test_pipeline.py tests/test_conversation.py
ruff format --check agent/pipeline.py tests/test_pipeline.py tests/test_conversation.py
```

Manually confirm the tests fail if queue depth becomes unbounded, TTS tasks are
started concurrently, or cleanup is removed.

- [ ] **Step 5: Commit**

```bash
git add agent/pipeline.py tests/test_pipeline.py tests/test_conversation.py
git commit -m "fix(pipeline): prefetch bounded sentence audio"
```

---

### Task 5: Preserve the legacy ARI A-law transport boundary

**Files:**
- Modify: `agent/ari.py`
- Modify: `tests/test_ari.py`

**Interfaces:**
- Consumes: complete or streamed 16 kHz PCM from `VoicePipeline`.
- Produces: complete or stateful streamed 8 kHz A-law only for `RtpServer`.

- [ ] **Step 1: Write failing complete and split-invariant adapter tests**

Add literal PCM fixtures and capture real RTP consumer input:

```python
async def test_ari_complete_pcm_is_adapted_to_alaw(ari):
    pcm = np.arange(1_600, dtype="<i2").tobytes()
    await ari._play_audio("ch-1", pcm, session, gen=1)
    sent = ari._rtp_servers["ch-1"].stream_audio.await_args.args[0]
    assert sent == alaw_encode(
        np.frombuffer(resample_pcm16(np.frombuffer(pcm, dtype="<i2"), 16_000, 8_000), dtype="<i2")
    )


async def test_ari_stream_adapter_retains_resampler_state(ari):
    source = np.arange(-2_000, 2_000, dtype=np.int16)
    async def stream(*_args):
        yield source[:101].tobytes()
        yield source[101:].tobytes()
    ari._pipeline.process_turn_stream = stream
    await ari._play_stream("ch-1", session, 1, pcm_input)
    emitted = b"".join(ari._rtp_servers["ch-1"].captured_chunks)
    expected_pcm = resample_pcm16(source, 16_000, 8_000)
    assert emitted == alaw_encode(np.frombuffer(expected_pcm, dtype="<i2"))
```

The production mutation caught is reintroducing stateless per-chunk conversion at the rollback boundary.

- [ ] **Step 2: Run ARI tests and verify red**

```bash
venv/bin/pytest tests/test_ari.py -q
```

Expected: raw PCM is currently treated as A-law.

- [ ] **Step 3: Implement the ARI-only adapter**

- Add private complete PCM16-to-A-law conversion using `resample_pcm16`.
- In `_play_stream.produce`, instantiate one
  `StreamingPcm16Resampler(16_000, 8_000)`, convert each even PCM chunk,
  A-law-encode non-empty output, and close it in `finally`.
- Reject odd input bytes.
- Keep `RtpServer` and its prebuffer/pacing code unchanged.

- [ ] **Step 4: Verify green**

```bash
venv/bin/pytest tests/test_ari.py tests/test_rtp.py -q
ruff check agent/ari.py tests/test_ari.py
ruff format --check agent/ari.py tests/test_ari.py
```

- [ ] **Step 5: Commit**

```bash
git add agent/ari.py tests/test_ari.py
git commit -m "fix(ari): adapt continuous PCM at RTP boundary"
```

---

### Task 6: Documentation, complete verification, review, and rollout

**Files:**
- Modify: `README.md`
- Modify: `docs/pjsip-migration.md`
- Modify: `agent/config.py` only if an outdated outbound A-law comment remains
- Verify: all repository tests and live services

**Interfaces:**
- Consumes: completed Tasks 1-5.
- Produces: documented direct-PCM production contract, exact built image, reviewed code, and live acceptance evidence.

- [ ] **Step 1: Update operational documentation**

Document:

```text
Qwen TTS 24 kHz PCM
  -> one stateful 24-to-16 kHz conversion per utterance
  -> 300 ms prebuffer
  -> PJSUA2 16 kHz PCM media port
```

State that A-law now exists only in the ARI rollback adapter and that sentence
prefetch is bounded to two.

- [ ] **Step 2: Run the complete local verification**

```bash
ruff check agent tests dgx/tts
ruff format --check agent tests dgx/tts
venv/bin/pytest -q
docker compose -f dgx/docker-compose.yml config --quiet
docker compose config --quiet
git diff --check
```

Expected: all checks pass with pristine output.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md docs/pjsip-migration.md agent/config.py
git commit -m "docs(audio): describe direct PCM streaming"
```

- [ ] **Step 4: Request independent review and fix every finding**

Review the full range from `69bf5d3` through `HEAD`, focusing on sample
continuity, queue bounds, cancellation, state transitions, legacy ARI
conversion, and test validity. Apply review feedback through fresh TDD cycles,
then rerun Step 2.

- [ ] **Step 5: Build the exact commit without replacing the live container**

```bash
commit=$(git rev-parse --short HEAD)
docker build -t "voip-agent-voip-agent:continuous-pcm-${commit}" .
```

Run a container-level import/config smoke test and record its immutable image
ID. Do not print environment values.

- [ ] **Step 6: Deploy through the feature Compose contract**

Tag the verified image as `voip-agent-voip-agent:latest`, then recreate only:

```bash
docker compose -p voip-agent \
  --project-directory /home/volsch/projekte/voip-agent \
  -f /home/volsch/projekte/.worktrees/voip-agent-shared-ai-traefik/compose.yml \
  up -d --force-recreate --no-deps --no-build voip-agent
```

Require running state, restart count zero, secret mounts present, SIP
registration `200 OK`, and the exact image ID.

- [ ] **Step 7: Run automated live acceptance**

Through the authenticated production client:

- stream a long sentence and assert chunked stateful conversion equals a
  one-shot `ratecv` result byte-for-byte;
- simulate the 300 ms playback buffer across five short sentences and require
  zero post-start underflow;
- require first playable PCM below 1.3 seconds;
- exercise cancellation and require the next request to recover below 3.0
  seconds;
- confirm container restart count remains zero and logs contain no queue,
  media, or streaming errors.

- [ ] **Step 8: Run the real-call gate**

Ask the user to call once and verify:

1. greeting is smooth;
2. a multi-sentence reply has no periodic chopping or sentence gaps;
3. barge-in stops playback promptly;
4. hangup is clean.

Correlate that call with sanitized logs. Only after this auditory gate and all
preceding evidence pass may the objective be marked complete.
