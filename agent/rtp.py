import asyncio
import struct
from collections.abc import Callable

RTP_HEADER_SIZE = 12
PAYLOAD_TYPE_ALAW = 0x08
RTP_VERSION = 2
FRAME_DURATION_S = 0.02  # 20 ms


def parse_rtp_payload(packet: bytes) -> bytes:
    """Extract the audio payload from an RTP packet (RFC 3550).

    The old code blindly stripped 12 bytes. A malformed datagram — wrong
    version, a CSRC list, a header extension, or padding — would then feed
    garbage (or header bytes) into the VAD buffer. Validate the header and
    honor the CC / X / P fields instead; return b"" on anything malformed so
    a bad packet is dropped, not mis-decoded.
    """
    if len(packet) < RTP_HEADER_SIZE:
        return b""

    b0 = packet[0]
    if (b0 >> 6) != RTP_VERSION:
        return b""
    padding = (b0 >> 5) & 0x1
    extension = (b0 >> 4) & 0x1
    csrc_count = b0 & 0x0F

    header_len = RTP_HEADER_SIZE + 4 * csrc_count
    if len(packet) < header_len:
        return b""

    if extension:
        # Extension header: 2 bytes profile + 2 bytes length (in 32-bit words),
        # followed by that many words of extension data.
        if len(packet) < header_len + 4:
            return b""
        ext_words = struct.unpack("!H", packet[header_len + 2 : header_len + 4])[0]
        header_len += 4 + 4 * ext_words
        if len(packet) < header_len:
            return b""

    payload = packet[header_len:]

    if padding:
        # Last octet is the padding length, including itself.
        if not payload:
            return b""
        pad_len = payload[-1]
        if pad_len == 0 or pad_len > len(payload):
            return b""
        payload = payload[:-pad_len]

    return payload


def build_rtp_packet(payload: bytes, seq: int, timestamp: int, ssrc: int) -> bytes:
    header = struct.pack(
        "!BBHII",
        0x80,
        PAYLOAD_TYPE_ALAW,
        seq & 0xFFFF,
        timestamp & 0xFFFFFFFF,
        ssrc & 0xFFFFFFFF,
    )
    return header + payload


class RtpServer(asyncio.DatagramProtocol):
    """Async UDP server that receives RTP from Asterisk ExternalMedia."""

    SAMPLES_PER_FRAME = 160  # 20ms at 8kHz

    def __init__(
        self,
        host: str,
        port: int,
        on_audio: Callable[[bytes], None],
    ) -> None:
        self._host = host
        self._port = port
        self._on_audio = on_audio
        self._transport: asyncio.DatagramTransport | None = None
        self._remote_addr: tuple[str, int] | None = None
        self._ssrc = 0x1234ABCD
        self._seq = 0
        self._timestamp = 0

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._remote_addr is None:
            self._remote_addr = addr
        payload = parse_rtp_payload(data)
        if payload:
            self._on_audio(payload)

    def send_frame(self, alaw_frame: bytes) -> None:
        if not self._transport or not self._remote_addr:
            return
        packet = build_rtp_packet(
            alaw_frame,
            seq=self._seq,
            timestamp=self._timestamp,
            ssrc=self._ssrc,
        )
        self._transport.sendto(packet, self._remote_addr)
        self._seq = (self._seq + 1) & 0xFFFF
        self._timestamp = (self._timestamp + self.SAMPLES_PER_FRAME) & 0xFFFFFFFF

    async def stream_audio(self, alaw: bytes) -> None:
        _SILENCE = b"\xd5" * self.SAMPLES_PER_FRAME
        loop = asyncio.get_running_loop()
        start = loop.time()
        frame_idx = 0
        for i in range(0, len(alaw), self.SAMPLES_PER_FRAME):
            chunk = alaw[i : i + self.SAMPLES_PER_FRAME]
            if len(chunk) < self.SAMPLES_PER_FRAME:
                chunk = chunk + _SILENCE[len(chunk) :]
            self.send_frame(chunk)
            frame_idx += 1
            # Pace against an absolute monotonic schedule, not sleep(0.02)
            # after each send. The old approach added the per-frame work time
            # and event-loop jitter on top of every gap, so playback drifted
            # slower than the RTP timestamps it was emitting. Anchoring each
            # frame to start + n*20ms keeps the wall clock and the RTP clock
            # aligned; a slow frame is absorbed by a shorter next sleep.
            target = start + frame_idx * FRAME_DURATION_S
            delay = target - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

    def close(self) -> None:
        if self._transport:
            self._transport.close()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: self, local_addr=(self._host, self._port))
