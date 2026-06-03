# Streaming Voice Pipeline (#3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the serialized turn loop with a streaming pipeline (LLM tokens → sentence segments → faster-qwen3-tts audio chunks → RTP) so the caller hears first audio in ~300–800ms instead of after the whole turn computes.

**Architecture:** STT stays whole-turn; the LLM streams tokens, a German-aware segmenter batches them into clauses, each clause is synthesized by faster-qwen3-tts's streaming generator into 24kHz PCM chunks, which are resampled to 8kHz aLaw and drained to RTP as 20ms frames. Compute and playback overlap; the FSM enters SPEAKING on the first emitted chunk. Barge-in cancels the whole producer chain via a supervising task.

**Tech Stack:** Python 3.12 asyncio, httpx (SSE + chunked streaming), faster-qwen3-tts (DGX), FastAPI StreamingResponse, scipy resample, audioop-lts aLaw, pytest (asyncio_mode=auto), respx.

---

## ⚠️ Wire-contract caveat (read before writing any boundary test)

Two wire formats are **unverified** because the DGX is unreachable from the dev box:

1. **LLM SSE** — `dgx-spark:8000` `/v1/chat/completions` streamed `delta` shape, especially incremental `tool_calls` fragmentation.
2. **TTS streaming chunks** — the endpoint does not exist yet (Task 1 builds it); sample rate, dtype, and framing of each yielded chunk are an assumption until verified on the box.

Per the project's own lesson (`tests-mock-client-server-boundary`): a fixture that encodes an *assumed* shape passes green whether or not the real service matches. **Every test below that mocks one of these two boundaries carries an `ASSUMPTION` note. At execution time, on the DGX box, capture one real response and replace the fixture.** Task 1 and Task 4 each include an explicit on-box verification step. Do not treat the invented fixtures as final.

---

## File structure

| File | Responsibility | New/changed |
|---|---|---|
| `dgx/tts/server.py` | Add `/v1/audio/speech/stream` → `generate_voice_design_streaming` → raw PCM `StreamingResponse`. Keep `/v1/audio/speech`. | changed |
| `dgx/tts/requirements.txt` (or Dockerfile) | Add `faster-qwen3-tts`. | changed |
| `agent/segmenter.py` | `SentenceSegmenter` — buffer tokens, emit on clause boundary, German abbreviations. Pure, no I/O. | new |
| `agent/tts.py` | `TtsClient.synthesize_stream(text)` async-gen yielding 24kHz int16 chunks. Keep `synthesize`. | changed |
| `agent/llm.py` | `LlmClient.complete_stream(messages, caller_id, on_tool_round)` async-gen of `str` text deltas; reuses the existing auth/cap/`_dispatch_safe`. | changed |
| `agent/rtp.py` | `RtpServer.stream_audio_chunks(queue, ...)` — prebuffer + underrun-safe drain. Keep `stream_audio`. | changed |
| `agent/pipeline.py` | `VoicePipeline.process_turn_stream(session, pcm_16k)` async-gen of aLaw chunk bytes; filler on tool round; mid-stream recovery. | changed |
| `agent/ari.py` | Streaming playback, first-chunk SPEAKING, PROCESSING-window barge-in, producer-chain teardown. | changed |
| `agent/main.py` | Pass `stt`/`llm.complete_stream`/`tts.synthesize_stream` callables; keep non-stream for greeting/filler. | changed |
| `tests/test_segmenter.py` | Segmenter unit tests. | new |
| `tests/test_tts.py` | Streaming client tests (ASSUMPTION fixtures). | changed |
| `tests/test_llm.py` | Streaming completion + unauthorized-streaming tests (ASSUMPTION fixtures). | changed |
| `tests/test_rtp.py` | Chunk-drain prebuffer/underrun tests. | changed |
| `tests/test_pipeline.py` | `process_turn_stream` orchestration + filler + recovery. | changed |
| `tests/test_ari.py` | Overlap FSM + PROCESSING-window barge-in. | changed |

---

## Task 1: TTS server streaming endpoint (DGX) + on-box wire verification

**Files:**
- Modify: `dgx/tts/server.py`
- Modify: `dgx/tts/requirements.txt` (add `faster-qwen3-tts`)

This task runs/verifies on the DGX box, not the dev machine. No pytest — the agent-side tests mock this boundary.

- [ ] **Step 1: Add faster-qwen3-tts dependency**

In `dgx/tts/requirements.txt` add:
```
faster-qwen3-tts
```

- [ ] **Step 2: Load the VoiceDesign model via faster-qwen3-tts**

Replace the `_load()` body in `dgx/tts/server.py` (the `from qwen_tts import ...` block) with the faster runtime, keeping the same `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` weights:
```python
def _load() -> None:
    global _model, _tokenizer
    if _model is not None:
        return
    from faster_qwen3_tts import FasterQwen3TTS
    print(f"[load] faster-qwen3-tts model {MODEL_ID} device={DEVICE}", flush=True)
    _model = FasterQwen3TTS.from_pretrained(MODEL_ID, device=DEVICE, dtype=DTYPE)
    _tokenizer = None  # faster runtime manages tokenization internally
    print("[load] OK", flush=True)
```
Note: confirm the exact constructor/import name against the installed `faster-qwen3-tts` version — the README shows `model.generate_voice_design_streaming(...)`; mirror its loading example.

- [ ] **Step 3: Add the streaming endpoint**

Add to `dgx/tts/server.py` (keep the existing `/v1/audio/speech`):
```python
from fastapi.responses import StreamingResponse

@app.post("/v1/audio/speech/stream")
def synthesize_stream(req: SpeechRequest):
    if _model is None:
        _load()
    instruct = req.voice or DEFAULT_INSTRUCT

    def gen():
        for audio_chunk, sample_rate, _timing in _model.generate_voice_design_streaming(
            text=req.input,
            instruct=instruct,
            language=req.language,
            chunk_size=8,  # ~667ms audio per chunk; tune later
        ):
            # Yield raw little-endian int16 PCM. Sample rate is fixed by the
            # model (24kHz). The agent client resamples 24k->8k.
            import numpy as np
            yield np.asarray(audio_chunk).astype("<i2").tobytes()

    return StreamingResponse(gen(), media_type="application/octet-stream")
```

- [ ] **Step 4: ON-BOX WIRE VERIFICATION (do not skip — resolves the TTS assumption)**

On the DGX, start the server and capture one real streamed response:
```bash
curl -N -s -X POST http://localhost:8002/v1/audio/speech/stream \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hallo, wie kann ich Ihnen helfen?","language":"de"}' \
  --output /tmp/tts_stream.raw
python3 -c "import numpy as np; a=np.fromfile('/tmp/tts_stream.raw',dtype='<i2'); print('samples',a.size,'≈sec@24k',a.size/24000)"
```
Expected: non-empty, plausible duration. **Record the confirmed sample rate and dtype** — if they differ from 24kHz/`<i2`, update `agent/tts.py` (Task 3) and its fixture before merging the agent side.

- [ ] **Step 5: Commit**
```bash
git add dgx/tts/server.py dgx/tts/requirements.txt
git commit -m "feat(tts-server): streaming PCM endpoint via faster-qwen3-tts (#3)"
```

---

## Task 2: SentenceSegmenter (pure, German-aware)

**Files:**
- Create: `agent/segmenter.py`
- Test: `tests/test_segmenter.py`

No wire risk — fully deterministic, test exhaustively.

- [ ] **Step 1: Write the failing tests**

`tests/test_segmenter.py`:
```python
from agent.segmenter import SentenceSegmenter


def _feed_all(seg, tokens):
    out = []
    for t in tokens:
        out.extend(seg.feed(t))
    tail = seg.flush()
    if tail:
        out.append(tail)
    return out


def test_emits_on_sentence_boundary():
    seg = SentenceSegmenter()
    out = _feed_all(seg, ["Hallo", " Welt", ".", " Wie", " geht", "'s", "?"])
    assert out == ["Hallo Welt.", "Wie geht's?"]


def test_flush_returns_trailing_partial():
    seg = SentenceSegmenter()
    out = _feed_all(seg, ["Kein", " Punkt", " hier"])
    assert out == ["Kein Punkt hier"]


def test_german_abbreviation_does_not_split():
    seg = SentenceSegmenter()
    # "z.B." must not split into three fragments.
    out = _feed_all(seg, ["Das", " ist", " z.B.", " ein", " Test", "."])
    assert out == ["Das ist z.B. ein Test."]


def test_empty_stream_flushes_none():
    seg = SentenceSegmenter()
    assert seg.flush() is None
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `venv/bin/pytest tests/test_segmenter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.segmenter'`

- [ ] **Step 3: Implement the segmenter**

`agent/segmenter.py`:
```python
"""Buffer an LLM token stream into clause/sentence units for TTS.

Feeding raw tokens to TTS produces robotic prosody — the model needs
coherent units. Emit on sentence-final punctuation, but never split on a
period that belongs to a known German abbreviation.
"""

_SENTENCE_END = {".", "!", "?"}
# Abbreviations whose trailing period must not end a sentence.
_ABBREVIATIONS = {
    "z.b.", "u.a.", "d.h.", "u.s.w.", "usw.", "etc.", "bzw.", "ca.",
    "nr.", "abs.", "vgl.", "z.t.", "evtl.", "ggf.", "inkl.", "max.", "min.",
}


class SentenceSegmenter:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """Append a token; return any completed sentences (possibly empty)."""
        self._buf += token
        out: list[str] = []
        while True:
            idx = self._next_boundary(self._buf)
            if idx is None:
                break
            sentence = self._buf[: idx + 1].strip()
            self._buf = self._buf[idx + 1 :]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        """Return any buffered trailing text (no terminal punctuation)."""
        tail = self._buf.strip()
        self._buf = ""
        return tail or None

    def _next_boundary(self, text: str) -> int | None:
        for i, ch in enumerate(text):
            if ch in _SENTENCE_END:
                # The char after the boundary must exist and not be a digit
                # (avoid splitting "3.14"); require it to be whitespace/end.
                if i + 1 < len(text) and not text[i + 1].isspace():
                    continue
                if self._ends_with_abbreviation(text[: i + 1]):
                    continue
                return i
        return None

    @staticmethod
    def _ends_with_abbreviation(text: str) -> bool:
        last = text.strip().split()[-1].lower() if text.strip() else ""
        return last in _ABBREVIATIONS
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `venv/bin/pytest tests/test_segmenter.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**
```bash
git add agent/segmenter.py tests/test_segmenter.py
git commit -m "feat(segmenter): German-aware sentence segmenter for token streams (#3)"
```

---

## Task 3: TtsClient.synthesize_stream

**Files:**
- Modify: `agent/tts.py`
- Test: `tests/test_tts.py`

- [ ] **Step 1: Write the failing test (ASSUMPTION fixture)**

Append to `tests/test_tts.py`:
```python
import numpy as np
import httpx
import pytest

from agent.tts import TtsClient


async def test_synthesize_stream_yields_pcm_chunks():
    # ASSUMPTION: server streams raw little-endian int16 PCM at 24kHz.
    # Replace this fixture with a captured /v1/audio/speech/stream response
    # once verified on the DGX box (see Task 1 Step 4).
    chunk_a = (np.arange(240, dtype="<i2")).tobytes()
    chunk_b = (np.arange(240, 480, dtype="<i2")).tobytes()

    async def handler(request):
        return httpx.Response(200, stream=httpx.ByteStream(chunk_a + chunk_b))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    tts = TtsClient(base_url="http://tts:8002", client=client)

    chunks = [c async for c in tts.synthesize_stream("Hallo")]
    joined = np.concatenate(chunks)

    assert joined.dtype == np.int16
    assert joined.size == 480
    await client.aclose()
```
Note: `MockTransport` delivers the body in one read, so this asserts decode/dtype, not multi-read chunking. The real multi-chunk behavior is covered by Task 1's on-box verification.

- [ ] **Step 2: Run test, verify it fails**

Run: `venv/bin/pytest tests/test_tts.py::test_synthesize_stream_yields_pcm_chunks -v`
Expected: FAIL — `AttributeError: 'TtsClient' object has no attribute 'synthesize_stream'`

- [ ] **Step 3: Implement synthesize_stream**

Add to `agent/tts.py` (keep imports; add `numpy`):
```python
import numpy as np
from collections.abc import AsyncIterator
```
Add the method to `TtsClient`:
```python
    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield 24kHz int16 PCM chunks as the server produces them.

        Buffers across reads so each yielded array is whole int16 samples
        (a chunk boundary can fall mid-sample on the wire).
        """
        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/audio/speech/stream",
            json={"input": text, "voice": VOICE_INSTRUCT},
            timeout=30.0,
        ) as resp:
            resp.raise_for_status()
            carry = b""
            async for raw in resp.aiter_bytes():
                buf = carry + raw
                n = len(buf) - (len(buf) % 2)  # whole int16 samples only
                if n:
                    yield np.frombuffer(buf[:n], dtype="<i2")
                carry = buf[n:]
            if carry:
                # Trailing odd byte should not happen; drop with no crash.
                pass
```

- [ ] **Step 4: Run test, verify it passes**

Run: `venv/bin/pytest tests/test_tts.py::test_synthesize_stream_yields_pcm_chunks -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add agent/tts.py tests/test_tts.py
git commit -m "feat(tts): synthesize_stream yields 24k PCM chunks (#3, ASSUMPTION fixture)"
```

---

## Task 4: LlmClient.complete_stream (reusing auth/cap, with unauthorized tests)

**Files:**
- Modify: `agent/llm.py`
- Test: `tests/test_llm.py`

**Guard preservation (advisor #2):** `complete_stream` MUST reuse `_is_authorized`, the `max_tool_rounds` cap, and `_dispatch_safe` exactly as `complete()` does. A streaming loop that sets `payload["tools"]` without the `authorized` gate reopens the exfiltration hole. The streaming-path unauthorized test below is mandatory.

- [ ] **Step 1: Write the failing tests (ASSUMPTION fixtures for SSE shape)**

Append to `tests/test_llm.py`:
```python
import httpx

from agent.llm import LlmClient


def _sse(*events: str) -> bytes:
    # ASSUMPTION: OpenAI/vLLM SSE framing `data: {json}\n\n`, terminated by
    # `data: [DONE]`. Replace with a captured dgx-spark:8000 stream once
    # reachable (see plan wire-contract caveat).
    body = "".join(f"data: {e}\n\n" for e in events) + "data: [DONE]\n\n"
    return body.encode()


def _text_delta(content: str) -> str:
    return (
        '{"choices":[{"delta":{"content":' + f'"{content}"' + '},"finish_reason":null}]}'
    )


def _make_client(handler, **over):
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return LlmClient(
        base_url="http://llm:8000",
        model="hermes",
        system_prompt="prompt",
        rag=None,
        calendar=None,
        trusted_callers={"+49123"},
        client=client,
        **over,
    )


async def test_complete_stream_yields_text_deltas():
    def handler(request):
        return httpx.Response(200, stream=httpx.ByteStream(
            _sse(_text_delta("Hallo"), _text_delta(" Welt"))
        ))

    llm = _make_client(handler)
    out = [t async for t in llm.complete_stream(
        [{"role": "user", "content": "hi"}], caller_id="+49123")]
    assert "".join(out) == "Hallo Welt"
    await llm._client.aclose()


async def test_complete_stream_unauthorized_caller_gets_no_tools():
    # The request payload must NOT include "tools" for an untrusted caller.
    seen = {}

    def handler(request):
        import json as _j
        seen["payload"] = _j.loads(request.content)
        return httpx.Response(200, stream=httpx.ByteStream(_sse(_text_delta("ok"))))

    llm = _make_client(handler)
    _ = [t async for t in llm.complete_stream(
        [{"role": "user", "content": "hi"}], caller_id="+49999")]  # not trusted
    assert "tools" not in seen["payload"]
    await llm._client.aclose()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `venv/bin/pytest tests/test_llm.py -k complete_stream -v`
Expected: FAIL — `AttributeError: ... 'complete_stream'`

- [ ] **Step 3: Implement complete_stream**

Add to `agent/llm.py`:
```python
from collections.abc import AsyncIterator, Callable
```
Add the method to `LlmClient` (reuses `_is_authorized`, `_max_tool_rounds`, `_dispatch_safe`):
```python
    async def complete_stream(
        self,
        messages: list[dict],
        caller_id: str | None = None,
        on_tool_round: Callable[[], None] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the final assistant answer as text deltas.

        Tool rounds are resolved internally (not yielded). Same auth/cap as
        complete(): tools are offered only to authorized callers and only
        below the round cap. on_tool_round fires once when a tool round is
        entered, so the caller can play a filler utterance during dispatch.
        """
        authorized = self._is_authorized(caller_id)
        full_messages = [
            {"role": "system", "content": self._system_prompt},
            *messages,
        ]
        for round_idx in range(self._max_tool_rounds + 1):
            tools_allowed = authorized and round_idx < self._max_tool_rounds
            payload = {"model": self._model, "messages": full_messages, "stream": True}
            if tools_allowed:
                payload["tools"] = TOOLS

            content_parts: list[str] = []
            tool_calls = _ToolCallAccumulator()
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=60.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"]
                    if delta.get("content"):
                        # Only stream text out when no tool round is pending —
                        # otherwise this is an intermediate tool turn.
                        if tools_allowed and tool_calls.started:
                            pass
                        else:
                            content_parts.append(delta["content"])
                            yield delta["content"]
                    if tools_allowed and delta.get("tool_calls"):
                        tool_calls.add(delta["tool_calls"])

            if not tools_allowed or not tool_calls.started:
                return  # final answer already yielded

            # Tool round: notify (for filler) and dispatch, then loop.
            if on_tool_round is not None:
                on_tool_round()
            assistant_msg = tool_calls.to_message()
            full_messages.append(assistant_msg)
            for tc in assistant_msg["tool_calls"]:
                result = await self._dispatch_safe(tc)
                full_messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result}
                )
        # Cap reached without a text answer.
        yield _FALLBACK_MSG
```
Add the accumulator helper at module scope in `agent/llm.py`:
```python
class _ToolCallAccumulator:
    """Reassemble streamed tool_calls deltas (fragmented by index)."""

    def __init__(self) -> None:
        self._calls: dict[int, dict] = {}

    @property
    def started(self) -> bool:
        return bool(self._calls)

    def add(self, deltas: list[dict]) -> None:
        for d in deltas:
            i = d.get("index", 0)
            call = self._calls.setdefault(
                i, {"id": "", "type": "function",
                     "function": {"name": "", "arguments": ""}}
            )
            if d.get("id"):
                call["id"] = d["id"]
            fn = d.get("function", {})
            if fn.get("name"):
                call["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                call["function"]["arguments"] += fn["arguments"]

    def to_message(self) -> dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [self._calls[i] for i in sorted(self._calls)],
        }
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `venv/bin/pytest tests/test_llm.py -k complete_stream -v`
Expected: PASS (2 passed)

- [ ] **Step 5: ON-BOX VERIFICATION (resolves the SSE assumption — do when DGX reachable)**

Capture a real streamed tool call and a real text stream:
```bash
curl -N -s http://dgx-spark:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nous-hermes","stream":true,"messages":[{"role":"user","content":"hallo"}]}' | head -40
```
Confirm `delta.content` framing and (with a tool-eliciting prompt) the `delta.tool_calls` index/fragment shape. If they differ from the fixtures, fix `_ToolCallAccumulator`/parsing and the fixtures, then re-run Step 4.

- [ ] **Step 6: Commit**
```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat(llm): complete_stream with reused auth/cap and tool reassembly (#3, ASSUMPTION SSE)"
```

---

## Task 5: RtpServer.stream_audio_chunks (prebuffer + underrun-safe)

**Files:**
- Modify: `agent/rtp.py`
- Test: `tests/test_rtp.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rtp.py`:
```python
import asyncio

from agent.rtp import RtpServer


class _FakeTransport:
    def __init__(self):
        self.sent = []

    def sendto(self, packet, addr):
        self.sent.append(packet)

    def close(self):
        pass


def _ready_server():
    srv = RtpServer(host="127.0.0.1", port=0, on_audio=lambda p: None)
    srv._transport = _FakeTransport()
    srv._remote_addr = ("127.0.0.1", 5000)
    return srv


async def test_stream_chunks_drains_all_frames():
    srv = _ready_server()
    queue: asyncio.Queue = asyncio.Queue()
    # 2 frames of aLaw per chunk.
    frame = b"\xd5" * RtpServer.SAMPLES_PER_FRAME
    await queue.put(frame * 2)
    await queue.put(frame)
    await queue.put(None)  # sentinel: producer done

    await srv.stream_audio_chunks(queue, prebuffer_frames=0)

    assert len(srv._transport.sent) == 3  # 2 + 1 frames


async def test_stream_chunks_underrun_does_not_stop():
    srv = _ready_server()
    queue: asyncio.Queue = asyncio.Queue()
    frame = b"\xd5" * RtpServer.SAMPLES_PER_FRAME

    async def slow_producer():
        await queue.put(frame)
        await asyncio.sleep(0.05)  # gap > one frame: forces underrun
        await queue.put(frame)
        await queue.put(None)

    asyncio.create_task(slow_producer())
    await srv.stream_audio_chunks(queue, prebuffer_frames=0)

    # Both real frames sent despite the gap; underrun filled with silence,
    # so total frames > 2 and the stream did not abort early.
    assert len(srv._transport.sent) >= 2
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `venv/bin/pytest tests/test_rtp.py -k stream_chunks -v`
Expected: FAIL — `AttributeError: ... 'stream_audio_chunks'`

- [ ] **Step 3: Implement stream_audio_chunks**

Add to `RtpServer` in `agent/rtp.py`:
```python
    async def stream_audio_chunks(
        self, queue: "asyncio.Queue", prebuffer_frames: int = 5
    ) -> None:
        """Drain aLaw bytes from a queue as 20ms frames on the monotonic clock.

        The queue carries aLaw byte blobs of arbitrary length; a None sentinel
        marks producer completion. Frames are paced against an absolute
        schedule (start + n*20ms) like stream_audio. On underrun (no frame
        ready at its slot) a silence frame is sent so the RTP clock never
        stalls and the caller hears comfort silence, not a glitch.

        prebuffer_frames delays the start until that many frames are buffered,
        trading a little first-audio latency for underrun resilience.
        """
        silence = b"\xd5" * self.SAMPLES_PER_FRAME
        pending = bytearray()
        done = False
        loop = asyncio.get_running_loop()

        async def refill(block: bool) -> None:
            nonlocal done
            try:
                item = await queue.get() if block else queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is None:
                done = True
            else:
                pending.extend(item)

        # Prebuffer.
        while not done and len(pending) < prebuffer_frames * self.SAMPLES_PER_FRAME:
            await refill(block=True)

        start = loop.time()
        frame_idx = 0
        while True:
            # Pull any ready items without blocking.
            while not done:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    done = True
                else:
                    pending.extend(item)

            if len(pending) >= self.SAMPLES_PER_FRAME:
                frame = bytes(pending[: self.SAMPLES_PER_FRAME])
                del pending[: self.SAMPLES_PER_FRAME]
            elif done:
                if pending:  # final short frame, pad to a full frame
                    frame = bytes(pending) + silence[len(pending):]
                    pending.clear()
                else:
                    break
            else:
                frame = silence  # underrun: comfort silence, keep the clock

            self.send_frame(frame)
            frame_idx += 1
            target = start + frame_idx * FRAME_DURATION_S
            delay = target - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif not done and len(pending) < self.SAMPLES_PER_FRAME:
                # Behind schedule and starved: yield so the producer can run.
                await refill(block=True)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `venv/bin/pytest tests/test_rtp.py -k stream_chunks -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add agent/rtp.py tests/test_rtp.py
git commit -m "feat(rtp): stream_audio_chunks with prebuffer and underrun comfort silence (#3)"
```

---

## Task 6: VoicePipeline.process_turn_stream

**Files:**
- Modify: `agent/pipeline.py`
- Test: `tests/test_pipeline.py`

The pipeline takes new streaming callables. Update the constructor to accept them alongside the existing ones (keep `synthesize_alaw`/`process_turn` for greeting/filler/fallback).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:
```python
from datetime import datetime, timezone

import numpy as np

from agent.pipeline import VoicePipeline
from agent.session import CallSession, SessionState


def _session():
    return CallSession(
        call_id="c", caller_id="+49123", history=[], created_at=datetime.now(timezone.utc)
    )


def _pcm():
    return np.zeros(320, dtype=np.int16)


async def test_process_turn_stream_yields_alaw_incrementally():
    async def stt(_b):
        return "hallo"

    async def llm_stream(_msgs, _caller, on_tool_round=None):
        for tok in ["Hallo", " Welt", "."]:
            yield tok

    async def tts_stream(_text):
        yield np.zeros(2400, dtype=np.int16)  # 100ms @ 24k -> ~33ms @ 8k

    pipe = VoicePipeline(
        stt=stt, llm=None, tts=None,
        llm_stream=llm_stream, tts_stream=tts_stream,
    )
    s = _session()
    s.transition(SessionState.LISTENING)
    chunks = [c async for c in pipe.process_turn_stream(s, _pcm())]

    assert chunks and all(isinstance(c, bytes) for c in chunks)
    assert s.history[-1]["role"] == "assistant"
    assert s.history[-1]["content"] == "Hallo Welt."


async def test_process_turn_stream_plays_filler_on_tool_round():
    async def stt(_b):
        return "frage"

    async def llm_stream(_msgs, _caller, on_tool_round=None):
        if on_tool_round:
            on_tool_round()  # simulate a tool round
        for tok in ["Antwort", "."]:
            yield tok

    seen = {"filler": 0}

    async def tts_stream(text):
        if "Moment" in text:
            seen["filler"] += 1
        yield np.zeros(2400, dtype=np.int16)

    pipe = VoicePipeline(
        stt=stt, llm=None, tts=None,
        llm_stream=llm_stream, tts_stream=tts_stream,
    )
    s = _session()
    s.transition(SessionState.LISTENING)
    _ = [c async for c in pipe.process_turn_stream(s, _pcm())]
    assert seen["filler"] == 1


async def test_process_turn_stream_recovers_on_midstream_error():
    async def stt(_b):
        return "hallo"

    async def llm_stream(_msgs, _caller, on_tool_round=None):
        yield "Teil"
        raise RuntimeError("llm died mid-stream")

    async def tts_stream(_text):
        yield np.zeros(2400, dtype=np.int16)

    pipe = VoicePipeline(
        stt=stt, llm=None, tts=None,
        llm_stream=llm_stream, tts_stream=tts_stream,
    )
    s = _session()
    s.transition(SessionState.LISTENING)
    # Must not raise; should still produce audio (the recovery prompt).
    chunks = [c async for c in pipe.process_turn_stream(s, _pcm())]
    assert chunks
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `venv/bin/pytest tests/test_pipeline.py -k process_turn_stream -v`
Expected: FAIL — `TypeError` (constructor) / `AttributeError: process_turn_stream`

- [ ] **Step 3: Implement the streaming pipeline**

Edit `agent/pipeline.py`. Update the constructor and add the method + constants:
```python
FILLER_TEXT = "Einen Moment, ich schaue nach."
FALLBACK_RECOVERY = "Entschuldigung, da ist etwas schiefgelaufen."


class VoicePipeline:
    def __init__(self, stt, llm, tts, llm_stream=None, tts_stream=None) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._llm_stream = llm_stream
        self._tts_stream = tts_stream

    async def _tts_alaw_chunks(self, text):
        """Synthesize text and yield aLaw byte blobs (resampled 24k->8k)."""
        async for pcm_24k in self._tts_stream(text):
            yield alaw_encode(resample_24k_to_8k(pcm_24k))

    async def process_turn_stream(self, session: CallSession, pcm_16k: np.ndarray):
        """Stream a turn: STT (full) -> LLM tokens -> segments -> TTS -> aLaw.

        Yields aLaw byte blobs. Owns the PROCESSING entry; the caller drives
        SPEAKING (on first chunk) and LISTENING. On any mid-stream failure,
        yields the recovery prompt audio instead of raising.
        """
        from agent.segmenter import SentenceSegmenter

        session.transition(SessionState.PROCESSING)
        try:
            transcript = await self._stt(pcm_16k.tobytes())
        except Exception:
            log.exception("STT failed")
            async for c in self._tts_alaw_chunks(FALLBACK_ASR):
                yield c
            return

        if not transcript.strip():
            async for c in self._tts_alaw_chunks(FALLBACK_ASR):
                yield c
            return

        session.history.append({"role": "user", "content": transcript})

        seg = SentenceSegmenter()
        parts: list[str] = []
        tool_round = {"hit": False}

        def _on_tool_round() -> None:
            tool_round["hit"] = True

        try:
            async for token in self._llm_stream(
                session.history, session.caller_id, on_tool_round=_on_tool_round
            ):
                # Play filler once, the first time a tool round is signalled.
                if tool_round["hit"] and tool_round.get("filler_played") is not True:
                    tool_round["filler_played"] = True
                    async for c in self._tts_alaw_chunks(FILLER_TEXT):
                        yield c
                parts.append(token)
                for sentence in seg.feed(token):
                    async for c in self._tts_alaw_chunks(sentence):
                        yield c
            tail = seg.flush()
            if tail:
                async for c in self._tts_alaw_chunks(tail):
                    yield c
        except Exception:
            log.exception("LLM/TTS failed mid-stream")
            # Leave the user turn in history (the model saw it); do not append
            # a partial assistant turn. Emit the recovery prompt and stop.
            async for c in self._tts_alaw_chunks(FALLBACK_RECOVERY):
                yield c
            return

        session.history.append({"role": "assistant", "content": "".join(parts)})
```
Note: the filler check uses a dict flag so `_on_tool_round` (a plain callback) and the generator coordinate without a closure-rebinding bug.

- [ ] **Step 4: Run tests, verify they pass**

Run: `venv/bin/pytest tests/test_pipeline.py -k process_turn_stream -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full pipeline suite (no regressions on process_turn)**

Run: `venv/bin/pytest tests/test_pipeline.py -v`
Expected: PASS (existing + new)

- [ ] **Step 6: Commit**
```bash
git add agent/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): process_turn_stream with filler + mid-stream recovery (#3)"
```

---

## Task 7: AriClient streaming playback + first-chunk SPEAKING overlap

**Files:**
- Modify: `agent/ari.py`
- Test: `tests/test_ari.py`

Replace the buffer-playback path with a chunk-streaming path. A supervising task owns the producer chain (so barge-in cancels it with one `.cancel()`); SPEAKING is entered on the first chunk.

- [ ] **Step 1: Write the failing test (first-chunk SPEAKING)**

Append to `tests/test_ari.py`:
```python
async def test_streaming_play_enters_speaking_on_first_chunk(ari):
    from datetime import datetime, timezone

    from agent.session import CallSession, SessionState

    session = CallSession(
        call_id="ch-1", caller_id="+49123", history=[],
        created_at=datetime.now(timezone.utc),
    )
    session.transition(SessionState.LISTENING)
    ari._sessions["ch-1"] = session
    ari._generation["ch-1"] = 1

    states = []

    async def fake_stream(_sess, _pcm):
        states.append(session.state)  # PROCESSING here (before first chunk)
        yield b"\xd5" * 160
        states.append(session.state)  # SPEAKING here (after first chunk)
        yield b"\xd5" * 160

    rtp = MagicMock()
    rtp.stream_audio_chunks = AsyncMock()
    ari._rtp_servers["ch-1"] = rtp
    ari._pipeline.process_turn_stream = fake_stream

    await ari._play_stream("ch-1", session, gen=1, pcm=None)

    assert SessionState.PROCESSING in states
    assert SessionState.SPEAKING in states
    assert session.state == SessionState.LISTENING  # back to listening at end
```
Note: this asserts the FSM crosses PROCESSING then SPEAKING during the stream and lands at LISTENING. Adjust to the final `_play_stream` signature if refined during implementation.

- [ ] **Step 2: Run test, verify it fails**

Run: `venv/bin/pytest tests/test_ari.py -k speaking_on_first_chunk -v`
Expected: FAIL — `AttributeError: ... '_play_stream'`

- [ ] **Step 3: Implement streaming playback**

Add to `AriClient` in `agent/ari.py` an output queue per call and the streaming play method. Add `self._out_queues: dict[str, asyncio.Queue] = {}` to `__init__` (alongside the existing per-call dicts).
```python
    async def _play_stream(self, channel_id, session, gen, pcm) -> None:
        """Drive a streaming turn: feed pipeline aLaw chunks into the RTP
        chunk drain, entering SPEAKING on the first chunk. A supervising
        TaskGroup owns producer (pipeline) + consumer (RTP drain); a single
        cancel on barge-in tears both down."""
        rtp = self._rtp_servers.get(channel_id)
        if not rtp:
            return
        if self._generation.get(channel_id) != gen:
            return

        out: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._out_queues[channel_id] = out
        first = {"seen": False}

        async def produce():
            try:
                async for alaw in self._pipeline.process_turn_stream(session, pcm):
                    if self._generation.get(channel_id) != gen:
                        break
                    if not first["seen"]:
                        first["seen"] = True
                        if session.state == SessionState.PROCESSING:
                            session.transition(SessionState.SPEAKING)
                    await out.put(alaw)
            finally:
                await out.put(None)  # sentinel

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(produce())
                tg.create_task(rtp.stream_audio_chunks(out))
        except* Exception:
            log.exception("streaming turn failed for %s", channel_id)
        finally:
            self._out_queues.pop(channel_id, None)

        if self._generation.get(channel_id) == gen and session.state == SessionState.SPEAKING:
            session.transition(SessionState.LISTENING)
            vad = self._vad_buffers.get(channel_id)
            if vad:
                vad.reset()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `venv/bin/pytest tests/test_ari.py -k speaking_on_first_chunk -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add agent/ari.py tests/test_ari.py
git commit -m "feat(ari): streaming playback, SPEAKING on first chunk via TaskGroup (#3)"
```

---

## Task 8: Barge-in during PROCESSING + producer-chain teardown

**Files:**
- Modify: `agent/ari.py`
- Test: `tests/test_ari.py`

**Advisor #3:** `_on_audio` currently early-returns unless state is `LISTENING`/`SPEAKING` (ari.py:225). In streaming, the PROCESSING→first-chunk window lasts seconds; without extending interruptibility to PROCESSING, the caller cannot barge in during generation. Add PROCESSING to the interruptible states and ensure the bumped generation tears down the in-flight `_play_stream` task.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ari.py`:
```python
async def test_bargein_during_processing_cancels_inflight_turn(ari):
    from datetime import datetime, timezone

    from agent.session import CallSession, SessionState

    session = CallSession(
        call_id="ch-1", caller_id="+49123", history=[],
        created_at=datetime.now(timezone.utc),
    )
    session.transition(SessionState.PROCESSING)  # mid-generation
    ari._sessions["ch-1"] = session
    ari._generation["ch-1"] = 1

    started = asyncio.Event()
    cancelled = {"hit": False}

    async def slow_turn(_cid, _sess, _gen, _pcm):
        started.set()
        try:
            await asyncio.sleep(10)  # simulate long generation
        except asyncio.CancelledError:
            cancelled["hit"] = True
            raise

    task = asyncio.create_task(slow_turn("ch-1", session, 1, None))
    ari._playback_tasks["ch-1"] = task
    await started.wait()

    # _on_audio must treat PROCESSING as interruptible: bump generation and
    # cancel the in-flight task. We call the post-VAD branch directly by
    # asserting the interruptible-state set includes PROCESSING.
    assert SessionState.PROCESSING in ari._INTERRUPTIBLE_STATES

    # Simulate the teardown _on_audio performs on barge-in:
    ari._generation["ch-1"] = 2
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert cancelled["hit"] is True
```

- [ ] **Step 2: Run test, verify it fails**

Run: `venv/bin/pytest tests/test_ari.py -k bargein_during_processing -v`
Expected: FAIL — `AttributeError: ... '_INTERRUPTIBLE_STATES'`

- [ ] **Step 3: Extend interruptibility to PROCESSING**

In `agent/ari.py`, add a class constant and use it in `_on_audio`. Add near the other class attributes:
```python
    _INTERRUPTIBLE_STATES = (
        SessionState.LISTENING,
        SessionState.SPEAKING,
        SessionState.PROCESSING,
    )
```
Replace the two state checks in `_on_audio` (the early `if session.state not in (LISTENING, SPEAKING)` at ari.py:225 and the post-lock recheck at ari.py:244) with:
```python
        if session.state not in self._INTERRUPTIBLE_STATES:
            return
```
and inside the lock:
```python
            if session.state not in self._INTERRUPTIBLE_STATES:
                return
```
In the same locked block, the existing generation bump + task cancel already tears down the in-flight task — but it must move the FSM from PROCESSING/SPEAKING back to LISTENING before the new turn. Replace the existing `if session.state == SessionState.SPEAKING: session.transition(LISTENING)` with:
```python
            if session.state in (SessionState.SPEAKING, SessionState.PROCESSING):
                session.transition(SessionState.LISTENING)
```
(`PROCESSING → LISTENING` is already a valid transition.) Finally, switch the turn dispatch from `process_turn` + `_play_audio` to the streaming path:
```python
            task = asyncio.create_task(self._play_stream(channel_id, session, gen, speech))
            self._playback_tasks[channel_id] = task
```
Remove the now-unused `response_alaw = await self._pipeline.process_turn(...)` block and the stale-generation recheck that followed it (the recheck now lives inside `_play_stream`).

- [ ] **Step 4: Run test, verify it passes**

Run: `venv/bin/pytest tests/test_ari.py -k bargein_during_processing -v`
Expected: PASS

- [ ] **Step 5: Run the full ari suite (no regressions)**

Run: `venv/bin/pytest tests/test_ari.py -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add agent/ari.py tests/test_ari.py
git commit -m "feat(ari): interruptible PROCESSING window + producer-chain teardown (#3)"
```

---

## Task 9: Wire main.py + update docs

**Files:**
- Modify: `agent/main.py`
- Modify: `CLAUDE.md`
- Modify: `TODO.md`

- [ ] **Step 1: Pass streaming callables into the pipeline**

In `agent/main.py`, find the `VoicePipeline(...)` construction and add the streaming callables (keep the existing bound methods for greeting/filler/fallback):
```python
    pipeline = VoicePipeline(
        stt=stt_client.transcribe,
        llm=llm_client.complete,
        tts=tts_client.synthesize,
        llm_stream=llm_client.complete_stream,
        tts_stream=tts_client.synthesize_stream,
    )
```

- [ ] **Step 2: Import sanity check**

Run: `venv/bin/python -c "from agent.main import main; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Update CLAUDE.md architecture invariants**

In `CLAUDE.md`, update the **State ownership** and **Audio path out** sections to describe the overlap: PROCESSING and SPEAKING now overlap; SPEAKING is entered on the first emitted aLaw chunk while compute continues; the producer chain (LLM stream → segmenter → TTS stream → RTP drain) is owned by a per-turn `TaskGroup` that barge-in cancels; PROCESSING is now interruptible. Add a line to **VoicePipeline callables** noting `llm_stream`/`tts_stream`.

- [ ] **Step 4: Mark #3 done in TODO.md**

In `TODO.md`, change the `#3 Streaming pipeline` checkbox from `[ ]` to `[x]` and the faster-qwen3-tts model item to `[x]`, with a one-line note pointing at the spec/plan.

- [ ] **Step 5: Full suite + lint**

Run:
```bash
venv/bin/pytest -q && venv/bin/ruff check agent/ tests/ && venv/bin/ruff format --check agent/ tests/
```
Expected: all pass, all formatted.

- [ ] **Step 6: Commit**
```bash
git add agent/main.py CLAUDE.md TODO.md
git commit -m "feat(#3): wire streaming pipeline into main; document overlap FSM"
```

---

## Self-review against the spec

- **Adopt faster-qwen3-tts, keep VoiceDesign** → Task 1 (same `Qwen3-TTS-12Hz-1.7B-VoiceDesign` weights via faster runtime).
- **Audio-output streaming requires the swap** → Task 1 uses `generate_voice_design_streaming`.
- **STT stays whole-turn** → Task 6 (`await self._stt(...)` then stream LLM→TTS).
- **Sentence-aware feeding** → Task 2 segmenter.
- **LLM token streaming** → Task 4.
- **Barge-in tears down producer chain (TaskGroup)** → Task 7 (TaskGroup) + Task 8 (cancel on generation bump).
- **FSM overlap, SPEAKING on first chunk** → Task 7.
- **Tool turns can't stream → filler** → Task 4 (`on_tool_round`) + Task 6 (filler playback).
- **Output prebuffer + underrun** → Task 5.
- **Mid-stream failure recovery** → Task 6.
- **Tuning knobs (chunk_size, prebuffer, VAD)** → exposed as params in Tasks 1/5; tuning is a runtime step, not code.
- **Guard preservation in streaming tool loop (advisor #2)** → Task 4 reuses `_is_authorized`/cap/`_dispatch_safe` + unauthorized-streaming test.
- **PROCESSING-window interruptibility (advisor #3)** → Task 8.
- **Wire-contract assumptions flagged (advisor #1)** → caveat section + ASSUMPTION notes on every boundary fixture + on-box verification steps in Tasks 1 and 4.

**Out of scope (per spec):** streaming STT, input-side endpointing, full jitter buffer + PLC, Parakeet ASR — no tasks, intentionally.

**Open implementation questions (carry to execution):** exact `faster-qwen3-tts` import/constructor name; confirmed TTS chunk sample-rate/dtype on the box; real vLLM SSE `delta.tool_calls` fragment shape. All three have on-box verification steps; none block writing the agent-side code against the flagged assumptions.
