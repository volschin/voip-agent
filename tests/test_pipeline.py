import asyncio
import io
import wave
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from agent.audio import resample_pcm16
from agent.pipeline import FALLBACK_RECOVERY, FILLER_TEXT, VoicePipeline, _decode_wav
from agent.session import CallSession, SessionState


def _wav(n_samples: int = 24000, rate: int = 24000) -> bytes:
    """Build a 24 kHz mono PCM_16 WAV, matching dgx/tts/server.py output."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.zeros(n_samples, dtype=np.int16).tobytes())
    return buf.getvalue()


def _wav_samples(
    samples: np.ndarray,
    rate: int = 24_000,
    channels: int = 1,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.astype("<i2", copy=False).tobytes())
    return buf.getvalue()


def _make_session() -> CallSession:
    s = CallSession(
        call_id="ch-1",
        caller_id="+49",
        history=[],
        created_at=datetime.now(timezone.utc),
    )
    s.state = SessionState.LISTENING
    return s


def _pcm_16k(duration_ms: int = 200) -> np.ndarray:
    return np.zeros(16000 * duration_ms // 1000, dtype=np.int16)


@pytest.fixture
def pipeline():
    stt = AsyncMock(return_value="Wie ist das Wetter?")
    tts = AsyncMock(return_value=_wav())
    llm = AsyncMock(return_value="Es ist sonnig.")
    return VoicePipeline(stt=stt, llm=llm, tts=tts)


def test_decode_wav_strips_header():
    # The old code did np.frombuffer(wav, int16), feeding the 44-byte RIFF
    # header into the resampler. _decode_wav must return exactly the samples.
    pcm = _decode_wav(_wav(24000))
    assert pcm.dtype == np.int16
    assert len(pcm) == 24000


def test_decode_wav_rejects_non_24khz_audio():
    with pytest.raises(ValueError, match="24 kHz"):
        _decode_wav(_wav_samples(np.zeros(160, dtype=np.int16), rate=16_000))


def test_decode_wav_rejects_non_mono_audio():
    stereo = np.zeros((160, 2), dtype=np.int16)

    with pytest.raises(ValueError, match="mono"):
        _decode_wav(_wav_samples(stereo, channels=2))


async def test_synthesize_pcm16_falls_back_on_non_wav():
    tts = AsyncMock(return_value=b"not a wav at all")
    p = VoicePipeline(stt=AsyncMock(), llm=AsyncMock(), tts=tts)
    assert await p.synthesize_pcm16("x") == b""


async def test_synthesize_pcm16_returns_direct_16k_pcm():
    source = np.arange(24_000, dtype=np.int16)
    p = VoicePipeline(
        stt=AsyncMock(),
        llm=AsyncMock(),
        tts=AsyncMock(return_value=_wav_samples(source)),
    )

    result = await p.synthesize_pcm16("Hallo")

    assert result == resample_pcm16(source, 24_000, 16_000)


async def test_process_full_turn_returns_pcm16(pipeline):
    session = _make_session()
    pcm_chunk = _pcm_16k(500)
    result = await pipeline.process_turn(session, pcm_chunk)
    assert isinstance(result, bytes)
    assert len(result) > 0
    # process_turn hands off in PROCESSING; AriClient._play_audio drives SPEAKING.
    assert session.state == SessionState.PROCESSING


async def test_process_turn_appends_to_history(pipeline):
    session = _make_session()
    await pipeline.process_turn(session, _pcm_16k(200))
    assert len(session.history) == 2  # user + assistant
    assert session.history[0]["role"] == "user"
    assert session.history[1]["role"] == "assistant"


async def test_process_turn_state_machine(pipeline):
    session = _make_session()
    assert session.state == SessionState.LISTENING
    await pipeline.process_turn(session, _pcm_16k(200))
    assert session.state == SessionState.PROCESSING  # hands off in PROCESSING


async def test_stt_failure_returns_fallback(pipeline):
    pipeline._stt.side_effect = RuntimeError("STT down")
    session = _make_session()
    result = await pipeline.process_turn(session, _pcm_16k(200))
    assert isinstance(result, bytes)
    assert session.state == SessionState.PROCESSING
    assert len(session.history) == 0


async def test_llm_failure_returns_fallback(pipeline):
    pipeline._llm.side_effect = RuntimeError("LLM down")
    session = _make_session()
    result = await pipeline.process_turn(session, _pcm_16k(200))
    assert isinstance(result, bytes)
    assert session.state == SessionState.PROCESSING
    assert len(session.history) == 0


def _strm_session() -> CallSession:
    return CallSession(
        call_id="c", caller_id="+49123", history=[], created_at=datetime.now(timezone.utc)
    )


def _pcm_zero() -> np.ndarray:
    return np.zeros(320, dtype=np.int16)


async def test_tts_pcm16_chunks_rechunk_stable_wav_in_exact_source_order():
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

    expected = resample_pcm16(source, 24_000, 16_000)
    assert chunks == [expected[offset : offset + 640] for offset in range(0, len(expected), 640)]
    assert len(chunks) > 2
    assert chunks[0] != chunks[1]
    assert all(len(chunk) <= 640 and len(chunk) % 2 == 0 for chunk in chunks)
    tts.assert_awaited_once_with("Hallo")
    tts_stream.assert_not_called()


async def test_tts_pcm16_chunks_emit_nothing_when_stable_synthesis_fails():
    pipe = VoicePipeline(
        stt=AsyncMock(),
        llm=AsyncMock(),
        tts=AsyncMock(return_value=b"not a wav"),
        tts_stream=AsyncMock(),
    )

    chunks = [chunk async for chunk in pipe._tts_pcm16_chunks("Kaputt")]

    assert chunks == []


async def test_process_turn_stream_yields_pcm16_incrementally():
    async def stt(_b):
        return "hallo"

    async def llm_stream(_msgs, _caller, on_tool_round=None):
        for tok in ["Hallo", " Welt", "."]:
            yield tok

    async def tts(_text):
        return _wav(2400)

    pipe = VoicePipeline(
        stt=stt,
        llm=None,
        tts=tts,
        llm_stream=llm_stream,
        tts_stream=AsyncMock(),
    )
    s = _strm_session()
    s.transition(SessionState.LISTENING)
    chunks = [c async for c in pipe.process_turn_stream(s, _pcm_zero())]

    assert chunks and all(isinstance(c, bytes) for c in chunks)
    assert s.history[-1]["role"] == "assistant"
    assert s.history[-1]["content"] == "Hallo Welt."


async def test_sentence_prefetch_is_bounded_and_preserves_order():
    first_tts_started = asyncio.Event()
    release_first = asyncio.Event()
    consumed = []
    tts_calls = []
    tokens = ["Eins. ", "Zwei. ", "Drei. ", "Vier. ", "Fünf. ", "Sechs."]

    async def stt(_pcm):
        return "frage"

    async def llm_stream(*_args, **_kwargs):
        for token in tokens:
            consumed.append(token)
            yield token

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
    session = _strm_session()
    session.transition(SessionState.LISTENING)
    stream = pipe.process_turn_stream(session, _pcm_zero())
    first = asyncio.create_task(anext(stream))

    await first_tts_started.wait()
    await asyncio.sleep(0)

    assert tts_calls == ["Eins."]
    assert len(consumed) <= 4
    assert len(consumed) < len(tokens)

    release_first.set()
    output = [await first] + [chunk async for chunk in stream]

    assert tts_calls == ["Eins.", "Zwei.", "Drei.", "Vier.", "Fünf.", "Sechs."]
    assert len(tts_calls) == len(set(tts_calls))
    assert b"".join(output)
    assert len(output) == 6
    assert forbidden_stream.call_count == 0


async def test_cancelled_turn_closes_prefetch_and_tts_tasks():
    tts_started = asyncio.Event()
    closed = asyncio.Event()

    async def stt(_pcm):
        return "frage"

    async def llm_stream(*_args, **_kwargs):
        yield "Eins."

    async def tts(_text):
        try:
            tts_started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    pipe = VoicePipeline(
        stt=stt,
        llm=None,
        tts=tts,
        llm_stream=llm_stream,
        tts_stream=AsyncMock(),
    )
    session = _strm_session()
    session.transition(SessionState.LISTENING)
    stream = pipe.process_turn_stream(session, _pcm_zero())
    pending = asyncio.create_task(anext(stream))

    await tts_started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await stream.aclose()

    assert closed.is_set()


async def test_process_turn_stream_plays_filler_on_tool_round():
    async def stt(_b):
        return "frage"

    async def llm_stream(_msgs, _caller, on_tool_round=None):
        if on_tool_round:
            on_tool_round()  # simulate a tool round
        for tok in ["Antwort", "."]:
            yield tok

    tts_calls = []

    async def tts(text):
        tts_calls.append(text)
        return _wav(2400)

    pipe = VoicePipeline(
        stt=stt,
        llm=None,
        tts=tts,
        llm_stream=llm_stream,
        tts_stream=AsyncMock(),
    )
    s = _strm_session()
    s.transition(SessionState.LISTENING)
    _ = [c async for c in pipe.process_turn_stream(s, _pcm_zero())]
    assert tts_calls == [FILLER_TEXT, "Antwort."]


async def test_process_turn_stream_recovers_on_midstream_error():
    async def stt(_b):
        return "hallo"

    async def llm_stream(_msgs, _caller, on_tool_round=None):
        yield "Teil"
        raise RuntimeError("llm died mid-stream")

    tts_calls = []

    async def tts(text):
        tts_calls.append(text)
        return _wav(2400)

    pipe = VoicePipeline(
        stt=stt,
        llm=None,
        tts=tts,
        llm_stream=llm_stream,
        tts_stream=AsyncMock(),
    )
    s = _strm_session()
    s.transition(SessionState.LISTENING)
    # Must not raise; should still produce audio (the recovery prompt).
    chunks = [c async for c in pipe.process_turn_stream(s, _pcm_zero())]
    assert chunks
    assert tts_calls[-1] == FALLBACK_RECOVERY


@pytest.mark.parametrize("tokens", [[], [" ", "\t", "\n"]], ids=["zero-token", "whitespace"])
async def test_process_turn_stream_recovers_from_empty_llm_output_without_blank_history(tokens):
    async def stt(_pcm):
        return "hallo"

    async def llm_stream(_msgs, _caller, on_tool_round=None):
        for token in tokens:
            yield token

    tts_calls = []

    async def tts(text):
        tts_calls.append(text)
        return _wav(2400)

    pipe = VoicePipeline(
        stt=stt,
        llm=None,
        tts=tts,
        llm_stream=llm_stream,
        tts_stream=AsyncMock(),
    )
    session = _strm_session()
    session.transition(SessionState.LISTENING)

    chunks = [chunk async for chunk in pipe.process_turn_stream(session, _pcm_zero())]

    assert chunks
    assert tts_calls == [FALLBACK_RECOVERY]
    assert session.history == [{"role": "user", "content": "hallo"}]


@pytest.mark.parametrize("failed_audio", [b"", b"not a wav"])
async def test_process_turn_stream_recovers_when_sentence_synthesis_is_empty(failed_audio):
    async def stt(_pcm):
        return "hallo"

    async def llm_stream(_msgs, _caller, on_tool_round=None):
        yield "Antwort."

    tts_calls = []

    async def tts(text):
        tts_calls.append(text)
        if text == FALLBACK_RECOVERY:
            return _wav(2400)
        return failed_audio

    pipe = VoicePipeline(
        stt=stt,
        llm=None,
        tts=tts,
        llm_stream=llm_stream,
        tts_stream=AsyncMock(),
    )
    session = _strm_session()
    session.transition(SessionState.LISTENING)

    chunks = [chunk async for chunk in pipe.process_turn_stream(session, _pcm_zero())]

    assert chunks
    assert tts_calls == ["Antwort.", FALLBACK_RECOVERY]
    assert session.history == [{"role": "user", "content": "hallo"}]
