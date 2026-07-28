import asyncio
from unittest.mock import AsyncMock, MagicMock

from agent.pjsip import (
    PcmPlaybackBuffer,
    PjsipAudioSink,
    PjsipClient,
    caller_id_from_uri,
)


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


async def test_audio_sink_waits_until_pjsip_consumes_audio():
    buffer = PcmPlaybackBuffer()
    sink = PjsipAudioSink(buffer)

    playback = asyncio.create_task(sink.play_audio(b"\xd5" * 160))
    await asyncio.sleep(0)

    assert buffer.buffered_bytes == 640
    assert not playback.done()
    assert len(buffer.read(640)) == 640
    await asyncio.wait_for(playback, timeout=0.1)


async def test_failed_priority_acquisition_terminates_unusable_call(settings):
    conversations = MagicMock()
    conversations.start_call = AsyncMock(return_value=False)
    call = MagicMock()
    call.call_id = "7"
    call.caller_id = "+49123"
    call.sink = MagicMock()
    client = PjsipClient(settings=settings, conversations=conversations)

    await client._start_conversation(call)

    call.sink.clear.assert_called_once()
    call.terminate.assert_called_once()
