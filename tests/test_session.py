import pytest
from datetime import datetime, timezone
from agent.session import CallSession, SessionState


def _make() -> CallSession:
    return CallSession(
        call_id="ch-123",
        caller_id="+49123456789",
        history=[],
        created_at=datetime.now(timezone.utc),
    )


def test_initial_state_is_answer():
    assert _make().state == SessionState.ANSWER


def test_full_happy_path():
    s = _make()
    s.transition(SessionState.LISTENING)    # greeting finished
    s.transition(SessionState.PROCESSING)  # speech detected
    s.transition(SessionState.SPEAKING)    # TTS ready
    s.transition(SessionState.LISTENING)   # turn complete
    assert s.state == SessionState.LISTENING


def test_interruption():
    s = _make()
    s.transition(SessionState.LISTENING)
    s.transition(SessionState.PROCESSING)
    s.transition(SessionState.SPEAKING)
    s.transition(SessionState.PROCESSING)  # caller speaks mid-playback
    assert s.state == SessionState.PROCESSING


def test_any_state_to_ended():
    for initial in (SessionState.LISTENING, SessionState.PROCESSING, SessionState.SPEAKING):
        s = _make()
        s.state = initial
        s.transition(SessionState.ENDED)
        assert s.state == SessionState.ENDED


def test_invalid_transition_raises():
    s = _make()
    s.transition(SessionState.LISTENING)
    with pytest.raises(ValueError, match="Invalid transition"):
        s.transition(SessionState.SPEAKING)  # LISTENING → SPEAKING not allowed
