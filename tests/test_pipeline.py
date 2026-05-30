import io
import wave
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from agent.pipeline import _SILENCE_FRAME, VoicePipeline, _decode_wav
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


async def test_synthesize_alaw_falls_back_on_non_wav():
    tts = AsyncMock(return_value=b"not a wav at all")
    p = VoicePipeline(stt=AsyncMock(), llm=AsyncMock(), tts=tts)
    assert await p.synthesize_alaw("x") == _SILENCE_FRAME


async def test_process_full_turn_returns_alaw(pipeline):
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
