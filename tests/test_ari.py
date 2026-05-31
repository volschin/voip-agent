import asyncio
import json
import socket
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.ari import AriClient
from agent.session import CallSession, SessionState


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


# --- #9 per-call queue + consumer ---------------------------------------


async def test_enqueue_audio_drops_when_full(ari):
    ari._audio_queues["ch-1"] = asyncio.Queue(maxsize=1)
    ari._enqueue_audio("ch-1", b"a")
    ari._enqueue_audio("ch-1", b"b")  # over capacity -> dropped, no raise
    assert ari._audio_queues["ch-1"].qsize() == 1


async def test_enqueue_audio_unknown_channel_is_noop(ari):
    ari._enqueue_audio("missing", b"a")  # no queue yet -> no raise


async def test_audio_consumer_drains_queue(ari):
    seen = []

    async def fake_on_audio(cid, payload):
        seen.append((cid, payload))

    ari._on_audio = fake_on_audio
    ari._audio_queues["ch-1"] = asyncio.Queue()
    task = asyncio.create_task(ari._audio_consumer("ch-1"))
    ari._audio_queues["ch-1"].put_nowait(b"x")
    await asyncio.sleep(0.01)
    task.cancel()
    assert seen == [("ch-1", b"x")]


async def test_audio_consumer_survives_handler_error(ari):
    calls = []

    async def boom(cid, payload):
        calls.append(payload)
        raise RuntimeError("boom")

    ari._on_audio = boom
    ari._audio_queues["ch-1"] = asyncio.Queue()
    task = asyncio.create_task(ari._audio_consumer("ch-1"))
    ari._audio_queues["ch-1"].put_nowait(b"1")
    ari._audio_queues["ch-1"].put_nowait(b"2")
    await asyncio.sleep(0.01)
    task.cancel()
    # consumer kept draining after the first frame raised
    assert calls == [b"1", b"2"]


# --- #8 barge-in generation token ---------------------------------------


async def test_play_audio_skips_stale_generation(ari):
    session = CallSession(
        call_id="ch-1", caller_id="+49", history=[], created_at=datetime.now(timezone.utc)
    )
    session.state = SessionState.PROCESSING
    rtp = MagicMock()
    rtp.stream_audio = AsyncMock()
    ari._rtp_servers["ch-1"] = rtp
    ari._generation["ch-1"] = 5  # current turn is gen 5

    # A playback scheduled for an older generation must not play or move state.
    await ari._play_audio("ch-1", b"\xd5" * 160, session, gen=1)

    rtp.stream_audio.assert_not_awaited()
    assert session.state == SessionState.PROCESSING


async def test_play_audio_current_generation_plays(ari):
    session = CallSession(
        call_id="ch-1", caller_id="+49", history=[], created_at=datetime.now(timezone.utc)
    )
    session.state = SessionState.PROCESSING
    ari._vad_buffers["ch-1"] = MagicMock()
    rtp = MagicMock()
    rtp.stream_audio = AsyncMock()
    ari._rtp_servers["ch-1"] = rtp
    ari._generation["ch-1"] = 3

    await ari._play_audio("ch-1", b"\xd5" * 160, session, gen=3)

    rtp.stream_audio.assert_awaited_once()
    assert session.state == SessionState.LISTENING


# --- #12 RTP bind collision handling ------------------------------------


async def test_bind_rtp_server_skips_busy_port(ari):
    ari._rtp_port_counter = ari._s.rtp_port
    busy_port = ari._s.rtp_port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ari._s.rtp_bind_host, busy_port))
    server = None
    try:
        server, port = await ari._bind_rtp_server("ch-1")
        assert port != busy_port
    finally:
        sock.close()
        if server:
            server.close()
