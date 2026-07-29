import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from agent.conversation import ConversationManager
from agent.pjsip import PcmPlaybackBuffer, PjsipAudioSink
from agent.priority import PriorityUnavailable
from agent.session import CallSession, SessionState


class FakeSink:
    def __init__(self):
        self.played = []
        self.cleared = 0
        self.closed = 0

    async def play_pcm16(self, pcm):
        self.played.append(pcm)

    async def play_pcm16_chunks(self, queue):
        while (chunk := await queue.get()) is not None:
            self.played.append(chunk)

    def clear(self):
        self.cleared += 1

    def close(self):
        self.closed += 1


class BlockingStreamSink(FakeSink):
    async def play_pcm16_chunks(self, _queue):
        await asyncio.Event().wait()


class ObservedPlaybackBuffer(PcmPlaybackBuffer):
    def __init__(self):
        super().__init__()
        self.maximum_buffered_bytes = 0

    def write(self, pcm):
        written = super().write(pcm)
        self.maximum_buffered_bytes = max(self.maximum_buffered_bytes, self.buffered_bytes)
        return written


def _manager(settings):
    pipeline = MagicMock()
    pipeline.synthesize_pcm16 = AsyncMock(return_value=b"greeting")
    pipeline.process_turn_stream = MagicMock()
    lease = MagicMock()
    lease.renew = AsyncMock()
    lease.release = AsyncMock()
    priority = MagicMock()
    priority.acquire = AsyncMock(return_value=lease)
    return ConversationManager(settings, pipeline, priority_client=priority), pipeline


async def test_start_call_plays_greeting_and_listens(settings):
    manager, pipeline = _manager(settings)
    sink = FakeSink()

    assert await manager.start_call("1", "+49123", sink) is True
    await asyncio.sleep(0)

    pipeline.synthesize_pcm16.assert_awaited_once_with(settings.greeting_text)
    assert sink.played == [b"greeting"]
    assert manager._sessions["1"].state is SessionState.LISTENING

    await manager.stop_call("1")
    assert sink.closed == 1
    assert manager.call_count == 0


async def test_start_call_acquires_priority_before_session_or_greeting(settings):
    order = []
    pipeline = MagicMock()

    async def greeting(_text):
        order.append("greeting")
        return b"greeting"

    lease = MagicMock()
    lease.renew = AsyncMock()
    lease.release = AsyncMock()
    priority = MagicMock()

    async def acquire():
        assert order == []
        order.append("acquire")
        return lease

    priority.acquire = AsyncMock(side_effect=acquire)
    pipeline.synthesize_pcm16 = AsyncMock(side_effect=greeting)
    manager = ConversationManager(settings, pipeline, priority_client=priority)

    assert await manager.start_call("1", "+49123", FakeSink()) is True
    assert order == ["acquire", "greeting"]
    await manager.stop_call("1")
    lease.release.assert_awaited_once()


async def test_priority_failure_starts_no_session_or_ai_operation(settings):
    pipeline = MagicMock()
    pipeline.synthesize_pcm16 = AsyncMock()
    priority = MagicMock()
    priority.acquire = AsyncMock(side_effect=PriorityUnavailable())
    manager = ConversationManager(settings, pipeline, priority_client=priority)
    sink = FakeSink()
    terminate_transport = MagicMock()

    started = await manager.start_call(
        "1",
        "+49123",
        sink,
        terminate_transport=terminate_transport,
    )

    assert started is False
    assert manager.call_count == 0
    pipeline.synthesize_pcm16.assert_not_awaited()
    assert sink.closed == 1
    terminate_transport.assert_called_once_with()


async def test_disconnect_while_priority_is_pending_releases_late_lease_and_allows_reuse(
    settings,
):
    first_acquire_started = asyncio.Event()
    finish_first_acquire = asyncio.Event()
    first_lease = MagicMock()
    first_lease.renew = AsyncMock()
    first_lease.release = AsyncMock()
    second_lease = MagicMock()
    second_lease.renew = AsyncMock()
    second_lease.release = AsyncMock()
    acquire_count = 0

    async def acquire():
        nonlocal acquire_count
        acquire_count += 1
        if acquire_count == 1:
            first_acquire_started.set()
            await finish_first_acquire.wait()
            return first_lease
        return second_lease

    pipeline = MagicMock()
    pipeline.synthesize_pcm16 = AsyncMock(return_value=b"greeting")
    priority = MagicMock()
    priority.acquire = AsyncMock(side_effect=acquire)
    manager = ConversationManager(settings, pipeline, priority_client=priority)
    first_sink = FakeSink()
    first_terminate = MagicMock()
    first_start = asyncio.create_task(
        manager.start_call(
            "1",
            "+49123",
            first_sink,
            terminate_transport=first_terminate,
        )
    )
    await first_acquire_started.wait()

    await manager.stop_call("1")

    second_sink = FakeSink()
    second_terminate = MagicMock()
    assert (
        await manager.start_call(
            "1",
            "+49456",
            second_sink,
            terminate_transport=second_terminate,
        )
        is True
    )
    finish_first_acquire.set()
    assert await first_start is False
    await asyncio.sleep(0)

    assert manager.call_count == 1
    assert manager._sinks["1"] is second_sink
    assert first_sink.played == []
    assert first_sink.closed == 1
    first_terminate.assert_not_called()
    first_lease.release.assert_awaited_once()
    first_lease.renew.assert_not_awaited()
    assert second_sink.played == [b"greeting"]

    await manager.stop_call("1")
    second_lease.release.assert_awaited_once()
    second_terminate.assert_not_called()


async def test_priority_renewal_failure_stops_call_and_clears_media(settings):
    pipeline = MagicMock()
    pipeline.synthesize_pcm16 = AsyncMock(return_value=b"greeting")
    lease = MagicMock()
    lease.renew = AsyncMock(side_effect=PriorityUnavailable())
    lease.release = AsyncMock()
    priority = MagicMock()
    priority.acquire = AsyncMock(return_value=lease)
    manager = ConversationManager(settings, pipeline, priority_client=priority)
    manager.PRIORITY_HEARTBEAT_SECONDS = 0.01
    sink = FakeSink()

    assert await manager.start_call("1", "+49123", sink) is True
    for _ in range(20):
        if manager.call_count == 0:
            break
        await asyncio.sleep(0.01)

    assert manager.call_count == 0
    assert sink.cleared >= 1
    assert sink.closed == 1
    lease.release.assert_awaited_once()


async def test_remote_disconnect_cleanup_does_not_terminate_transport(settings):
    manager, _pipeline = _manager(settings)
    terminate_transport = MagicMock()

    assert (
        await manager.start_call(
            "1",
            "+49123",
            FakeSink(),
            terminate_transport=terminate_transport,
        )
        is True
    )

    await manager.stop_call("1")
    await manager.stop_call("1")

    terminate_transport.assert_not_called()


async def test_pcm_barge_in_cancels_playback_and_starts_turn(settings):
    manager, pipeline = _manager(settings)
    sink = FakeSink()
    await manager.start_call("1", "+49123", sink)
    await asyncio.sleep(0)

    vad = MagicMock()
    vad.add_frame.return_value = np.ones(320, dtype=np.int16)
    manager._vad_buffers["1"] = vad

    async def stream(_session, _pcm):
        _session.transition(SessionState.PROCESSING)
        yield b"reply"

    pipeline.process_turn_stream = stream
    await manager._on_pcm("1", np.zeros(320, dtype=np.int16).tobytes())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert b"reply" in sink.played
    assert manager._generation["1"] == 1
    assert sink.cleared >= 1
    await manager.stop_call("1")


async def test_full_output_queue_does_not_deadlock_stream_cancellation(settings):
    manager, pipeline = _manager(settings)
    sink = BlockingStreamSink()
    await manager.start_call("1", "+49123", sink)
    await asyncio.sleep(0)
    session = manager._sessions["1"]
    closed = asyncio.Event()

    async def stream(_session, _pcm):
        _session.transition(SessionState.PROCESSING)
        try:
            for _ in range(100):
                yield b"\x00\x00" * 320
        finally:
            closed.set()

    pipeline.process_turn_stream = stream
    task = asyncio.create_task(
        manager._play_stream("1", session, manager._generation["1"], np.zeros(1, dtype=np.int16))
    )

    for _ in range(100):
        if manager._out_queues.get("1") and manager._out_queues["1"].full():
            break
        await asyncio.sleep(0)
    assert manager._out_queues["1"].full()

    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=0.05)
    completed_without_unblock = task in done
    if not completed_without_unblock:
        manager._out_queues["1"].get_nowait()
        await asyncio.wait({task}, timeout=0.1)

    assert completed_without_unblock
    assert closed.is_set()
    await manager.stop_call("1")


async def test_blocked_pjsip_playback_never_exceeds_two_second_aggregate_backlog(settings):
    manager, pipeline = _manager(settings)
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    session = CallSession(
        call_id="1",
        caller_id="+49123",
        history=[],
        created_at=datetime.now(timezone.utc),
    )
    session.transition(SessionState.LISTENING)
    manager._sinks["1"] = sink
    manager._generation["1"] = 0

    async def stream(_session, _pcm):
        _session.transition(SessionState.PROCESSING)
        for value in range(200):
            yield np.full(320, value, dtype="<i2").tobytes()

    pipeline.process_turn_stream = stream
    playback = asyncio.create_task(
        manager._play_stream("1", session, generation=0, pcm=np.zeros(1, dtype=np.int16))
    )
    try:
        for _ in range(1_000):
            queue = manager._out_queues.get("1")
            if queue is not None and queue.full():
                break
            await asyncio.sleep(0)
        queue = manager._out_queues["1"]

        assert (
            buffer.buffered_bytes + queue.queued_bytes + PjsipAudioSink.PCM_BLOCK_BYTES
            <= PjsipAudioSink.MAX_AHEAD_BYTES
        )
    finally:
        playback.cancel()
        await asyncio.gather(playback, return_exceptions=True)


async def test_streaming_playback_error_clears_prefix_before_following_turn(settings):
    manager, pipeline = _manager(settings)
    buffer = ObservedPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    session = CallSession(
        call_id="1",
        caller_id="+49123",
        history=[],
        created_at=datetime.now(timezone.utc),
    )
    session.transition(SessionState.LISTENING)
    manager._sinks["1"] = sink
    manager._generation["1"] = 0
    first = True

    async def stream(_session, _pcm):
        nonlocal first
        _session.transition(SessionState.PROCESSING)
        if first:
            first = False
            for _ in range(15):
                yield b"\x11\x00" * 320
            yield b"\xff"
            return
        yield b"\x22\x00" * 320

    pipeline.process_turn_stream = stream

    await manager._play_stream("1", session, generation=0, pcm=np.zeros(1, dtype=np.int16))

    assert buffer.maximum_buffered_bytes >= sink.PREBUFFER_BYTES
    assert buffer.buffered_bytes == 0
    assert session.state is SessionState.LISTENING

    following = asyncio.create_task(
        manager._play_stream("1", session, generation=0, pcm=np.zeros(1, dtype=np.int16))
    )
    for _ in range(100):
        if buffer.buffered_bytes:
            break
        await asyncio.sleep(0)
    assert buffer.read(sink.PCM_BLOCK_BYTES) == b"\x22\x00" * 320
    await asyncio.wait_for(following, timeout=0.1)


async def test_real_pjsip_barge_in_finalizes_old_producer_and_starts_clean(settings):
    pipeline = MagicMock()
    pipeline.synthesize_pcm16 = AsyncMock(return_value=b"\x00\x00" * 320)
    lease = MagicMock()
    lease.renew = AsyncMock()
    lease.release = AsyncMock()
    priority = MagicMock()
    priority.acquire = AsyncMock(return_value=lease)
    manager = ConversationManager(settings, pipeline, priority_client=priority)
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)

    assert await manager.start_call("1", "+49123", sink) is True
    for _ in range(100):
        if buffer.buffered_bytes:
            break
        await asyncio.sleep(0)
    buffer.read(buffer.buffered_bytes)
    await asyncio.wait_for(manager._playback_tasks["1"], timeout=0.1)

    first_finalized = asyncio.Event()
    first_finalizing = asyncio.Event()
    release_first_finalizer = asyncio.Event()
    second_started = asyncio.Event()
    generation_count = 0

    async def stream(session, _pcm):
        nonlocal generation_count
        generation_count += 1
        current = generation_count
        session.transition(SessionState.PROCESSING)
        session.history.append({"role": "user", "content": f"user-{current}"})
        if current == 1:
            try:
                for _ in range(15):
                    yield b"\x11\x00" * 320
                await asyncio.Event().wait()
            finally:
                first_finalizing.set()
                await release_first_finalizer.wait()
                first_finalized.set()
            session.history.append({"role": "assistant", "content": "late-old"})
            return
        second_started.set()
        for _ in range(15):
            yield b"\x22\x00" * 320
        session.history.append({"role": "assistant", "content": "new"})

    pipeline.process_turn_stream = stream
    vad = MagicMock()
    vad.add_frame.return_value = np.ones(320, dtype=np.int16)
    manager._vad_buffers["1"] = vad

    await manager._on_pcm("1", np.zeros(320, dtype=np.int16).tobytes())
    for _ in range(100):
        if buffer.buffered_bytes >= sink.PREBUFFER_BYTES:
            break
        await asyncio.sleep(0)
    assert buffer.buffered_bytes >= sink.PREBUFFER_BYTES

    barge_in = asyncio.create_task(manager._on_pcm("1", np.zeros(320, dtype=np.int16).tobytes()))
    await asyncio.wait_for(first_finalizing.wait(), timeout=0.1)
    await asyncio.sleep(0)
    assert second_started.is_set() is False
    release_first_finalizer.set()
    await asyncio.wait_for(barge_in, timeout=0.1)
    assert first_finalized.is_set()
    for _ in range(100):
        if buffer.buffered_bytes >= sink.PREBUFFER_BYTES:
            break
        await asyncio.sleep(0)

    assert buffer.read(sink.PREBUFFER_BYTES) == b"\x22\x00" * 4_800
    current = manager._playback_tasks["1"]
    while not current.done():
        if buffer.buffered_bytes:
            buffer.read(buffer.buffered_bytes)
        await asyncio.sleep(0)
    await current

    assert manager._sessions["1"].history == [
        {"role": "user", "content": "user-1"},
        {"role": "user", "content": "user-2"},
        {"role": "assistant", "content": "new"},
    ]
    await manager.stop_call("1")


async def test_complete_playback_error_restores_listening_state(settings):
    manager, _pipeline = _manager(settings)
    sink = FakeSink()

    async def fail(_pcm):
        raise RuntimeError("sink failed")

    sink.play_pcm16 = fail
    session = MagicMock()
    session.state = SessionState.PROCESSING
    session.transition = MagicMock(side_effect=lambda state: setattr(session, "state", state))
    manager._sinks["1"] = sink
    manager._generation["1"] = 3

    await manager._play_pcm16("1", b"\x00\x00", session, generation=3)

    assert session.state is SessionState.LISTENING
    assert sink.cleared == 1


async def test_stop_all_closes_every_sink(settings):
    manager, _pipeline = _manager(settings)
    first = FakeSink()
    second = FakeSink()
    await manager.start_call("1", "+491", first)
    await manager.start_call("2", "+492", second)

    await manager.stop_all()

    assert manager.call_count == 0
    assert first.closed == 1
    assert second.closed == 1
