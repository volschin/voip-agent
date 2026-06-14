import asyncio
import json
import socket
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
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


async def test_stasis_start_ignores_external_media_channel(ari):
    # Our own ExternalMedia leg re-enters Stasis with an ext- id. It must NOT
    # spawn a call, or each call recursively creates another session/RTP/bridge.
    called = []

    async def fake_setup(ch_id, caller_id):
        called.append(ch_id)

    ari._setup_call = fake_setup
    await ari._handle_event(
        {
            "type": "StasisStart",
            "channel": {"id": f"{ari._EXT_PREFIX}ch-1", "caller": {"number": ""}},
            "application": "voip-agent",
        }
    )
    await asyncio.sleep(0)
    assert called == []


async def test_stasis_start_tolerates_missing_caller_number(ari):
    # ExternalMedia / originated channels may lack caller.number; must not raise.
    called = []

    async def fake_setup(ch_id, caller_id):
        called.append((ch_id, caller_id))

    ari._setup_call = fake_setup
    await ari._handle_event({"type": "StasisStart", "channel": {"id": "ch-9"}})
    await asyncio.sleep(0)
    assert called == [("ch-9", "")]


async def test_setup_call_rolls_back_on_failure(ari):
    # A failure after per-call state is registered must tear everything back
    # down — no leaked session, RTP socket, queue, or consumer task.
    rtp = MagicMock()

    async def fake_bind(channel_id):
        return rtp, 5002

    ari._bind_rtp_server = fake_bind
    ari._create_external_media = AsyncMock(side_effect=RuntimeError("boom"))

    await ari._setup_call("ch-1", "+49")

    assert "ch-1" not in ari._sessions
    assert "ch-1" not in ari._rtp_servers
    assert "ch-1" not in ari._audio_queues
    assert "ch-1" not in ari._consumer_tasks
    assert "ch-1" not in ari._vad_buffers
    rtp.close.assert_called_once()


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


# --- #7 ExternalMedia advertises a routable host, not the bind host -------


async def test_external_media_advertises_routable_host(ari):
    # Asterisk sends RTP *to* external_host. It must be the advertise host
    # (reachable from Asterisk), never the bind host — which may be 0.0.0.0.
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"id": "ext-ch-1"})
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    ari._http = client

    ext_id = await ari._create_external_media("ch-1", 5002)

    assert ext_id == "ext-ch-1"
    params = client.post.await_args.kwargs["params"]
    assert params["external_host"] == "192.168.178.2:5002"
    # bind host (127.0.0.1 in the fixture) must NOT be what we advertise
    assert ari._s.rtp_bind_host not in params["external_host"]


# --- #3 streaming playback + overlap FSM --------------------------------


async def test_streaming_play_enters_speaking_on_first_chunk(ari):
    session = CallSession(
        call_id="ch-1",
        caller_id="+49123",
        history=[],
        created_at=datetime.now(timezone.utc),
    )
    session.transition(SessionState.LISTENING)
    ari._sessions["ch-1"] = session
    ari._generation["ch-1"] = 1

    states = []

    async def fake_stream(_sess, _pcm):
        # Mirror the real process_turn_stream: it owns the PROCESSING entry.
        session.transition(SessionState.PROCESSING)
        states.append(session.state)  # PROCESSING (before first chunk)
        yield b"\xd5" * 160
        states.append(session.state)  # SPEAKING (after first chunk)
        yield b"\xd5" * 160

    rtp = MagicMock()
    rtp.stream_audio_chunks = AsyncMock()
    ari._rtp_servers["ch-1"] = rtp
    ari._pipeline.process_turn_stream = fake_stream

    await ari._play_stream("ch-1", session, gen=1, pcm=None)

    assert SessionState.PROCESSING in states
    assert SessionState.SPEAKING in states
    assert session.state == SessionState.LISTENING  # back to listening at end


async def test_bargein_during_processing_cancels_and_starts_stream(ari):
    # Drives _on_audio for real: VAD fires while the session is PROCESSING.
    # Verifies the WIRING (generation bump + _play_stream dispatch), not just
    # that PROCESSING is in the interruptible-state constant.
    session = CallSession(
        call_id="ch-1",
        caller_id="+49123",
        history=[],
        created_at=datetime.now(timezone.utc),
    )
    session.transition(SessionState.LISTENING)
    session.transition(SessionState.PROCESSING)  # mid-generation
    ari._sessions["ch-1"] = session
    ari._generation["ch-1"] = 1
    ari._turn_locks["ch-1"] = asyncio.Lock()

    vad = MagicMock()
    vad.add_frame = MagicMock(return_value=b"speech-pcm")  # VAD fires
    ari._vad_buffers["ch-1"] = vad

    ari._play_stream = AsyncMock()

    assert SessionState.PROCESSING in ari._INTERRUPTIBLE_STATES

    await ari._on_audio("ch-1", b"\xd5" * 160)  # valid aLaw -> real decode path
    await asyncio.sleep(0)  # let the dispatched task run

    assert ari._generation["ch-1"] == 2  # generation bumped
    ari._play_stream.assert_awaited_once()
    args = ari._play_stream.await_args.args
    assert args[0] == "ch-1" and args[2] == 2  # channel_id, gen
    vad.reset.assert_called_once()


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


def test_reset_vad_clears_both_buffers(ari_td):
    # Codex P2: stale barge-in buffer must be cleared on return to LISTENING,
    # not just the turn-end buffer, or a partial noise survives into the next
    # response and can fire a false barge-in.
    ari, _td = ari_td
    turn_vad = MagicMock()
    bargein_vad = MagicMock()
    ari._vad_buffers["ch-1"] = turn_vad
    ari._bargein_buffers["ch-1"] = bargein_vad
    ari._reset_vad("ch-1")
    turn_vad.reset.assert_called_once()
    bargein_vad.reset.assert_called_once()


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
