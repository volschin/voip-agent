import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from agent.conversation import ConversationManager
from agent.priority import PriorityUnavailable
from agent.session import SessionState


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

    started = await manager.start_call("1", "+49123", FakeSink())

    assert started is False
    assert manager.call_count == 0
    pipeline.synthesize_pcm16.assert_not_awaited()


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
                yield b"\x00\x00"
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
