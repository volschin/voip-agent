import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.ari import AriClient


@pytest.fixture
def ari(settings):
    pipeline = AsyncMock(return_value=b"\xd5" * 160)
    return AriClient(settings=settings, pipeline=pipeline)


def _stasis_start_event(channel_id: str = "ch-1", caller: str = "+49123") -> str:
    return json.dumps(
        {
            "type": "StasisStart",
            "channel": {
                "id": channel_id,
                "caller": {"number": caller},
            },
            "application": "voip-agent",
        }
    )


def _stasis_end_event(channel_id: str = "ch-1") -> str:
    return json.dumps(
        {
            "type": "StasisEnd",
            "channel": {"id": channel_id},
            "application": "voip-agent",
        }
    )


async def test_stasis_start_creates_session(ari):
    called_with = []

    async def fake_setup(ch_id, caller_id):
        called_with.append((ch_id, caller_id))

    ari._setup_call = fake_setup
    await ari._handle_event(json.loads(_stasis_start_event("ch-1", "+49")))
    await asyncio.sleep(0)  # let the created task run
    assert called_with == [("ch-1", "+49")]


async def test_stasis_end_removes_session(ari):
    from datetime import datetime, timezone

    from agent.session import CallSession, SessionState

    session = CallSession(
        call_id="ch-1",
        caller_id="+49",
        history=[],
        created_at=datetime.now(timezone.utc),
    )
    session.state = SessionState.LISTENING
    ari._sessions["ch-1"] = session

    mock_rtp = MagicMock()
    ari._rtp_servers["ch-1"] = mock_rtp

    await ari._handle_event(json.loads(_stasis_end_event("ch-1")))
    assert "ch-1" not in ari._sessions
    mock_rtp.close.assert_called_once()


async def test_unknown_event_ignored(ari):
    await ari._handle_event({"type": "ChannelDtmfReceived", "channel": {"id": "ch-1"}})
    # no exception
