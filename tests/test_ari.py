import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent.ari import AriClient
from agent.config import Settings


@pytest.fixture
def ari(settings):
    pipeline = AsyncMock(return_value=b"\xd5" * 160)
    return AriClient(settings=settings, pipeline=pipeline)


def _stasis_start_event(channel_id: str = "ch-1", caller: str = "+49123") -> str:
    return json.dumps({
        "type": "StasisStart",
        "channel": {
            "id": channel_id,
            "caller": {"number": caller},
        },
        "application": "voip-agent",
    })


def _stasis_end_event(channel_id: str = "ch-1") -> str:
    return json.dumps({
        "type": "StasisEnd",
        "channel": {"id": channel_id},
        "application": "voip-agent",
    })


async def test_stasis_start_creates_session(ari):
    with patch.object(ari, "_setup_call", new_callable=AsyncMock) as mock_setup:
        await ari._handle_event(json.loads(_stasis_start_event("ch-1", "+49")))
        mock_setup.assert_awaited_once_with("ch-1", "+49")


async def test_stasis_end_removes_session(ari):
    from agent.session import CallSession, SessionState
    from datetime import datetime, timezone

    session = CallSession(
        call_id="ch-1", caller_id="+49",
        history=[], created_at=datetime.now(timezone.utc)
    )
    session.state = SessionState.LISTENING
    ari._sessions["ch-1"] = session

    await ari._handle_event(json.loads(_stasis_end_event("ch-1")))
    assert "ch-1" not in ari._sessions


async def test_unknown_event_ignored(ari):
    await ari._handle_event({"type": "ChannelDtmfReceived", "channel": {"id": "ch-1"}})
    # no exception
