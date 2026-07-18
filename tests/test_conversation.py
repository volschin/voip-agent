import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from agent.conversation import ConversationManager
from agent.session import SessionState


class FakeSink:
    def __init__(self):
        self.played = []
        self.cleared = 0
        self.closed = 0

    async def play_audio(self, alaw):
        self.played.append(alaw)

    async def play_audio_chunks(self, queue):
        while (chunk := await queue.get()) is not None:
            self.played.append(chunk)

    def clear(self):
        self.cleared += 1

    def close(self):
        self.closed += 1


def _manager(settings):
    pipeline = MagicMock()
    pipeline.synthesize_alaw = AsyncMock(return_value=b"greeting")
    pipeline.process_turn_stream = MagicMock()
    return ConversationManager(settings, pipeline), pipeline


async def test_start_call_plays_greeting_and_listens(settings):
    manager, pipeline = _manager(settings)
    sink = FakeSink()

    await manager.start_call("1", "+49123", sink)
    await asyncio.sleep(0)

    pipeline.synthesize_alaw.assert_awaited_once_with(settings.greeting_text)
    assert sink.played == [b"greeting"]
    assert manager._sessions["1"].state is SessionState.LISTENING

    await manager.stop_call("1")
    assert sink.closed == 1
    assert manager.call_count == 0


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
    await manager.stop_call("1")


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
