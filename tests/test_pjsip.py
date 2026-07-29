import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from agent.conversation import ConversationManager
from agent.pjsip import (
    PcmPlaybackBuffer,
    PjsipAudioSink,
    PjsipClient,
    caller_id_from_uri,
)
from agent.priority import PriorityUnavailable


class RecordingPlaybackBuffer(PcmPlaybackBuffer):
    def __init__(self):
        super().__init__()
        self.writes = []

    def write(self, pcm):
        self.writes.append(pcm)
        return super().write(pcm)


def test_caller_id_is_extracted_without_display_name():
    uri = '"Caller" <sip:+4912345@fritz.box>;tag=abc'

    assert caller_id_from_uri(uri) == "+4912345"
    assert caller_id_from_uri("anonymous") == ""


def test_playback_buffer_reads_across_chunks_and_pads_silence():
    buffer = PcmPlaybackBuffer(max_bytes=20)
    assert buffer.write(b"ab") is True
    assert buffer.write(b"cdef") is True

    assert buffer.read(4) == b"abcd"
    assert buffer.read(4) == b"ef\x00\x00"
    assert buffer.buffered_bytes == 0


def test_playback_buffer_rejects_overflow_and_closed_writes():
    buffer = PcmPlaybackBuffer(max_bytes=4)

    assert buffer.write(b"12345") is False
    buffer.close()
    assert buffer.write(b"1") is False


async def test_pjsip_sink_writes_pcm_without_alaw_roundtrip():
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    pcm = np.arange(320, dtype="<i2").tobytes()

    playback = asyncio.create_task(sink.play_pcm16(pcm))
    await asyncio.sleep(0)

    assert buffer.read(len(pcm)) == pcm
    assert not playback.done()
    await asyncio.wait_for(playback, timeout=0.1)


async def test_stream_playback_waits_for_300ms_prebuffer():
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    queue = asyncio.Queue()
    playback = asyncio.create_task(sink.play_pcm16_chunks(queue))

    await queue.put(b"\x01\x00" * 4_799)
    await asyncio.sleep(0)
    assert buffer.buffered_bytes == 0

    await queue.put(b"\x02\x00")
    await asyncio.sleep(0)
    assert buffer.buffered_bytes == 9_600

    buffer.read(9_600)
    await queue.put(None)
    await asyncio.wait_for(playback, timeout=0.1)


async def test_stream_playback_releases_short_response_at_end_of_stream():
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    queue = asyncio.Queue()
    await queue.put(b"\x01\x00" * 1_000)
    await queue.put(None)

    playback = asyncio.create_task(sink.play_pcm16_chunks(queue))
    await asyncio.sleep(0)

    assert buffer.buffered_bytes == 2_000
    buffer.read(2_000)
    await asyncio.wait_for(playback, timeout=0.1)


async def test_stream_prebuffer_splits_oversized_threshold_chunk():
    buffer = RecordingPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    queue = asyncio.Queue()
    await queue.put(b"\x01\x00" * 6_000)
    await queue.put(None)
    playback = asyncio.create_task(sink.play_pcm16_chunks(queue))

    try:
        await asyncio.sleep(0)
        assert len(buffer.writes[0]) == sink.PREBUFFER_BYTES
        assert b"".join(buffer.writes) == b"\x01\x00" * 6_000
    finally:
        playback.cancel()
        with suppress(asyncio.CancelledError):
            await playback


async def test_audio_sink_never_buffers_beyond_maximum_ahead():
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    sink.MAX_AHEAD_BYTES = 1_000
    playback = asyncio.create_task(sink.play_pcm16(b"\x01\x00" * 1_500))

    try:
        await asyncio.sleep(0)
        assert buffer.buffered_bytes <= sink.MAX_AHEAD_BYTES
    finally:
        playback.cancel()
        with suppress(asyncio.CancelledError):
            await playback


async def test_audio_sink_rejects_odd_pcm_byte_count():
    sink = PjsipAudioSink(PcmPlaybackBuffer())

    with pytest.raises(ValueError, match="even byte count"):
        await sink.play_pcm16(b"\x00")


async def test_cancelled_stream_discards_prebuffered_pcm():
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)
    queue = asyncio.Queue()
    playback = asyncio.create_task(sink.play_pcm16_chunks(queue))
    await queue.put(b"\x01\x00" * 1_000)
    await asyncio.sleep(0)

    playback.cancel()
    with pytest.raises(asyncio.CancelledError):
        await playback

    assert buffer.buffered_bytes == 0


async def test_failed_priority_acquisition_terminates_unusable_call(settings):
    conversations = MagicMock()
    conversations.start_call = AsyncMock(return_value=False)
    call = MagicMock()
    call.call_id = "7"
    call.caller_id = "+49123"
    call.sink = MagicMock()
    client = PjsipClient(settings=settings, conversations=conversations)

    await client._start_conversation(call)

    conversations.start_call.assert_awaited_once_with(
        "7",
        "+49123",
        call.sink,
        terminate_transport=call.terminate,
    )


async def test_priority_renewal_failure_terminates_native_call_once(settings):
    pipeline = MagicMock()
    pipeline.synthesize_pcm16 = AsyncMock(return_value=b"greeting")
    lease = MagicMock()
    lease.renew = AsyncMock(side_effect=PriorityUnavailable())
    lease.release = AsyncMock()
    priority = MagicMock()
    priority.acquire = AsyncMock(return_value=lease)
    conversations = ConversationManager(settings, pipeline, priority_client=priority)
    conversations.PRIORITY_HEARTBEAT_SECONDS = 0.01
    client = PjsipClient(settings=settings, conversations=conversations)
    call = MagicMock()
    call.call_id = "7"
    call.caller_id = "+49123"
    playback = PcmPlaybackBuffer()
    call.sink = PjsipAudioSink(playback)

    await client._start_conversation(call)
    for _ in range(20):
        if conversations.call_count == 0:
            break
        await asyncio.sleep(0.01)
    await conversations.stop_call("7")

    assert conversations.call_count == 0
    assert playback.buffered_bytes == 0
    call.terminate.assert_called_once_with()
    lease.release.assert_awaited_once()
