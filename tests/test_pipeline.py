from datetime import datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from agent.pipeline import VoicePipeline
from agent.session import CallSession, SessionState


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
    tts = AsyncMock(return_value=np.zeros(8000, dtype=np.int16).tobytes())
    llm = AsyncMock(return_value="Es ist sonnig.")
    return VoicePipeline(stt=stt, llm=llm, tts=tts)


async def test_process_full_turn_returns_alaw(pipeline):
    session = _make_session()
    pcm_chunk = _pcm_16k(500)
    result = await pipeline.process_turn(session, pcm_chunk)
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert session.state == SessionState.LISTENING


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
    assert session.state == SessionState.LISTENING  # back to LISTENING after full turn


async def test_stt_failure_returns_fallback(pipeline):
    pipeline._stt.side_effect = RuntimeError("STT down")
    session = _make_session()
    result = await pipeline.process_turn(session, _pcm_16k(200))
    assert isinstance(result, bytes)
    assert session.state == SessionState.LISTENING
    assert len(session.history) == 0


async def test_llm_failure_returns_fallback(pipeline):
    pipeline._llm.side_effect = RuntimeError("LLM down")
    session = _make_session()
    result = await pipeline.process_turn(session, _pcm_16k(200))
    assert isinstance(result, bytes)
    assert session.state == SessionState.LISTENING
    assert len(session.history) == 0
