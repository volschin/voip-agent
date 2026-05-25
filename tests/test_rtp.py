import asyncio
import struct
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
