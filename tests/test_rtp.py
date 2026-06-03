import asyncio
import struct
import time
from unittest.mock import MagicMock

from agent.rtp import RtpServer, build_rtp_packet, parse_rtp_payload


def _rtp_packet(payload: bytes, seq: int = 1, ts: int = 160) -> bytes:
    header = struct.pack("!BBHII", 0x80, 0x08, seq, ts, 0xDEADBEEF)
    return header + payload


def test_parse_rtp_payload_strips_header():
    payload = b"\xd5" * 160  # 160 bytes aLaw
    packet = _rtp_packet(payload)
    result = parse_rtp_payload(packet)
    assert result == payload


def test_parse_rtp_too_short_returns_empty():
    assert parse_rtp_payload(b"\x80\x08") == b""


def test_parse_rtp_wrong_version_returns_empty():
    # version bits must be 2 (0b10). 0x00 = version 0.
    bad = struct.pack("!BBHII", 0x00, 0x08, 1, 160, 0xDEADBEEF) + b"\xd5" * 160
    assert parse_rtp_payload(bad) == b""


def test_parse_rtp_with_csrc_skips_csrc_list():
    payload = b"\xd5" * 160
    cc = 2  # 2 CSRC identifiers => 8 extra header bytes
    b0 = 0x80 | cc
    header = struct.pack("!BBHII", b0, 0x08, 1, 160, 0xDEADBEEF) + b"\x00" * (4 * cc)
    assert parse_rtp_payload(header + payload) == payload


def test_parse_rtp_with_extension_header_skipped():
    payload = b"\xd5" * 160
    b0 = 0x80 | 0x10  # extension bit set
    base = struct.pack("!BBHII", b0, 0x08, 1, 160, 0xDEADBEEF)
    ext_words = 3
    ext = struct.pack("!HH", 0xBEDE, ext_words) + b"\x00" * (4 * ext_words)
    assert parse_rtp_payload(base + ext + payload) == payload


def test_parse_rtp_with_padding_stripped():
    b0 = 0x80 | 0x20  # padding bit set
    base = struct.pack("!BBHII", b0, 0x08, 1, 160, 0xDEADBEEF)
    pad_len = 4
    payload = b"\xd5" * 160 + b"\x00" * (pad_len - 1) + bytes([pad_len])
    assert parse_rtp_payload(base + payload) == b"\xd5" * 160


def test_parse_rtp_bad_padding_length_returns_empty():
    b0 = 0x80 | 0x20
    base = struct.pack("!BBHII", b0, 0x08, 1, 160, 0xDEADBEEF)
    # claims 200 bytes of padding in a 4-byte payload
    payload = b"\x01\x02\x03" + bytes([200])
    assert parse_rtp_payload(base + payload) == b""


def test_parse_rtp_truncated_extension_returns_empty():
    b0 = 0x80 | 0x10
    base = struct.pack("!BBHII", b0, 0x08, 1, 160, 0xDEADBEEF)
    # extension declares 5 words but none follow
    ext = struct.pack("!HH", 0xBEDE, 5)
    assert parse_rtp_payload(base + ext) == b""


def test_build_rtp_packet_has_correct_header():
    payload = b"\x00" * 160
    packet = build_rtp_packet(payload, seq=5, timestamp=800, ssrc=0xCAFE)
    assert len(packet) == 172  # 12 header + 160 payload
    v, pt, seq_out, ts_out, ssrc_out = struct.unpack("!BBHII", packet[:12])
    assert v == 0x80
    assert pt == 0x08  # G.711 aLaw
    assert seq_out == 5
    assert ts_out == 800
    assert ssrc_out == 0xCAFE


async def test_rtp_server_calls_callback_with_payload():
    received = []

    def on_audio(payload: bytes) -> None:
        received.append(payload)

    server = RtpServer(host="127.0.0.1", port=0, on_audio=on_audio)
    transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
        lambda: server, local_addr=("127.0.0.1", 0)
    )
    bound_port = transport.get_extra_info("sockname")[1]

    payload = b"\xd5" * 160
    packet = _rtp_packet(payload)
    send_transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
        asyncio.DatagramProtocol,
        remote_addr=("127.0.0.1", bound_port),
    )
    send_transport.sendto(packet)
    await asyncio.sleep(0.05)

    transport.close()
    send_transport.close()

    assert received == [payload]


async def test_stream_audio_sends_paced_frames():
    """stream_audio must split audio into 160-byte frames with 20 ms gaps."""
    frames_sent = []

    server = RtpServer(host="127.0.0.1", port=0, on_audio=lambda _: None)
    # Inject a fake transport that records sendto calls
    fake_transport = MagicMock()
    fake_transport.sendto = lambda pkt, _addr: frames_sent.append(pkt)
    server._transport = fake_transport
    server._remote_addr = ("127.0.0.1", 9999)

    # 3 frames of audio
    alaw = b"\xd5" * 480
    await server.stream_audio(alaw)

    assert len(frames_sent) == 3
    # Each RTP packet = 12-byte header + 160-byte payload
    for pkt in frames_sent:
        assert len(pkt) == 172


async def test_stream_audio_paces_against_monotonic_clock():
    """Total wall time tracks the absolute 20 ms/frame schedule and does not
    drift longer when per-frame work consumes time."""
    server = RtpServer(host="127.0.0.1", port=0, on_audio=lambda _: None)
    fake_transport = MagicMock()

    # Burn real time inside each send so naive per-frame sleeps would
    # accumulate drift; absolute scheduling must absorb it.
    def slow_sendto(_pkt, _addr):
        time.sleep(0.005)

    fake_transport.sendto = slow_sendto
    server._transport = fake_transport
    server._remote_addr = ("127.0.0.1", 9999)

    n_frames = 10
    alaw = b"\xd5" * (160 * n_frames)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await server.stream_audio(alaw)
    elapsed = loop.time() - start

    # Schedule is n_frames * 20 ms = 200 ms. The naive sleep(0.02)-after-work
    # loop would need ~10 * (5 + 20) ms = 250 ms. Absolute pacing stays near
    # 200 ms because each per-frame delay shrinks to swallow the work time.
    assert 0.19 <= elapsed <= 0.225


class _FakeTransport:
    def __init__(self):
        self.sent = []

    def sendto(self, packet, addr):
        self.sent.append(packet)

    def close(self):
        pass


def _ready_server():
    srv = RtpServer(host="127.0.0.1", port=0, on_audio=lambda p: None)
    srv._transport = _FakeTransport()
    srv._remote_addr = ("127.0.0.1", 5000)
    return srv


async def test_stream_chunks_drains_all_frames():
    srv = _ready_server()
    queue: asyncio.Queue = asyncio.Queue()
    # 2 frames of aLaw per chunk.
    frame = b"\xd5" * RtpServer.SAMPLES_PER_FRAME
    await queue.put(frame * 2)
    await queue.put(frame)
    await queue.put(None)  # sentinel: producer done

    await srv.stream_audio_chunks(queue, prebuffer_frames=0)

    assert len(srv._transport.sent) == 3  # 2 + 1 frames


async def test_stream_chunks_underrun_does_not_stop():
    srv = _ready_server()
    queue: asyncio.Queue = asyncio.Queue()
    frame = b"\xd5" * RtpServer.SAMPLES_PER_FRAME

    async def slow_producer():
        await queue.put(frame)
        await asyncio.sleep(0.05)  # gap > one frame: forces underrun
        await queue.put(frame)
        await queue.put(None)

    asyncio.create_task(slow_producer())
    await srv.stream_audio_chunks(queue, prebuffer_frames=0)

    # Both real frames sent despite the gap; underrun filled with silence,
    # so total frames > 2 and the stream did not abort early.
    assert len(srv._transport.sent) >= 2
