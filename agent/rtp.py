import asyncio
import struct
from collections.abc import Callable

RTP_HEADER_SIZE = 12
PAYLOAD_TYPE_ALAW = 0x08


def parse_rtp_payload(packet: bytes) -> bytes:
    if len(packet) < RTP_HEADER_SIZE:
        return b""
    return packet[RTP_HEADER_SIZE:]


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
        for i in range(0, len(alaw), self.SAMPLES_PER_FRAME):
            chunk = alaw[i : i + self.SAMPLES_PER_FRAME]
            if len(chunk) < self.SAMPLES_PER_FRAME:
                chunk = chunk + _SILENCE[len(chunk):]
            self.send_frame(chunk)
            await asyncio.sleep(0.02)

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: self, local_addr=(self._host, self._port)
        )
