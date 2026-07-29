"""Direct FRITZ!Box PJSUA2 transport for the production voice pipeline."""

from __future__ import annotations

import asyncio
import importlib
import logging
import re
import threading
from collections import deque
from contextlib import suppress

from agent.config import Settings
from agent.conversation import ConversationManager
from agent.pjsip_poc import DelayedAnswerService

log = logging.getLogger(__name__)

_CALLER_URI = re.compile(r"sip:([^@;>]+)", re.IGNORECASE)


def caller_id_from_uri(remote_uri: str) -> str:
    """Extract the SIP user while keeping the raw URI out of application logs."""

    match = _CALLER_URI.search(remote_uri)
    return match.group(1) if match else ""


class PcmPlaybackBuffer:
    """Thread-safe PCM buffer shared by asyncio and PJSIP media callbacks."""

    def __init__(self, max_bytes: int = 16_000 * 2 * 10) -> None:
        self._chunks: deque[bytes] = deque()
        self._head_offset = 0
        self._size = 0
        self._max_bytes = max_bytes
        self._closed = False
        self._lock = threading.Lock()

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return self._size

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def write(self, pcm: bytes) -> bool:
        if not pcm:
            return True
        with self._lock:
            if self._closed or self._size + len(pcm) > self._max_bytes:
                return False
            self._chunks.append(pcm)
            self._size += len(pcm)
            return True

    def read(self, size: int) -> bytes:
        if size <= 0:
            return b""
        output = bytearray()
        with self._lock:
            while self._chunks and len(output) < size:
                chunk = self._chunks[0]
                available = len(chunk) - self._head_offset
                take = min(size - len(output), available)
                output.extend(chunk[self._head_offset : self._head_offset + take])
                self._head_offset += take
                self._size -= take
                if self._head_offset == len(chunk):
                    self._chunks.popleft()
                    self._head_offset = 0
        if len(output) < size:
            output.extend(b"\x00" * (size - len(output)))
        return bytes(output)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._head_offset = 0
            self._size = 0

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._chunks.clear()
            self._head_offset = 0
            self._size = 0


class PjsipAudioSink:
    """Buffer pipeline PCM for the 16 kHz PJSIP audio port."""

    SAMPLE_RATE = 16_000
    BYTES_PER_SAMPLE = 2
    PREBUFFER_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 300 // 1000
    MAX_AHEAD_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 2

    def __init__(self, buffer: PcmPlaybackBuffer) -> None:
        self._buffer = buffer
        self._closed = False

    def clear(self) -> None:
        self._buffer.clear()

    def close(self) -> None:
        self._closed = True
        self._buffer.close()

    async def _write_with_backpressure(self, pcm: bytes) -> None:
        if len(pcm) % self.BYTES_PER_SAMPLE:
            raise ValueError("PCM16 requires an even byte count")
        while not self._closed and self._buffer.buffered_bytes > self.MAX_AHEAD_BYTES:
            await asyncio.sleep(0.02)
        if self._closed:
            return
        if not self._buffer.write(pcm):
            raise RuntimeError("PJSIP playback buffer capacity exceeded")

    async def _wait_drained(self) -> None:
        while not self._closed and self._buffer.buffered_bytes:
            await asyncio.sleep(0.02)

    async def play_pcm16(self, pcm: bytes) -> None:
        try:
            await self._write_with_backpressure(pcm)
            await self._wait_drained()
        except asyncio.CancelledError:
            self.clear()
            raise

    async def play_pcm16_chunks(self, queue: asyncio.Queue) -> None:
        pending = bytearray()
        playback_started = False
        try:
            while not self._closed:
                pcm = await queue.get()
                if pcm is None:
                    if pending:
                        await self._write_with_backpressure(bytes(pending))
                    break
                if len(pcm) % self.BYTES_PER_SAMPLE:
                    raise ValueError("PCM16 requires an even byte count")
                if not playback_started:
                    pending.extend(pcm)
                    if len(pending) < self.PREBUFFER_BYTES:
                        continue
                    await self._write_with_backpressure(bytes(pending))
                    pending.clear()
                    playback_started = True
                    continue
                await self._write_with_backpressure(pcm)
            await self._wait_drained()
        except asyncio.CancelledError:
            self.clear()
            raise


class PjsipClient:
    """Register with the FRITZ!Box and connect calls to ConversationManager."""

    def __init__(self, settings: Settings, conversations: ConversationManager) -> None:
        self._s = settings
        self._conversations = conversations
        self._stop_requested = False
        self._endpoint = None
        self._account = None
        self._answer_policy = DelayedAnswerService(
            answer_delay_seconds=settings.answer_delay_seconds,
            max_call_seconds=settings.max_call_seconds,
            max_concurrent_calls=settings.max_concurrent_calls,
        )

    def request_stop(self) -> None:
        self._stop_requested = True

    async def _start_conversation(self, call: object) -> None:
        """Terminate an answered media call if priority cannot be acquired."""
        try:
            started = await self._conversations.start_call(
                call.call_id,
                call.caller_id,
                call.sink,
            )
        except Exception:
            log.exception("Could not start conversation for call %s", call.call_id)
            started = False
        if not started:
            call.sink.clear()
            call.terminate()

    @property
    def registrar_uri(self) -> str:
        uri = f"sip:{self._s.fritzbox_host}"
        if self._s.pjsip_transport == "tcp":
            return f"{uri};transport=tcp"
        return uri

    @property
    def identity_uri(self) -> str:
        return f"sip:{self._s.fritzbox_sip_username}@{self._s.fritzbox_host}"

    async def run(self) -> None:
        try:
            pj = importlib.import_module("pjsua2")
        except ImportError as exc:
            raise RuntimeError("pjsua2 is unavailable; run the agent Docker image") from exc

        loop = asyncio.get_running_loop()
        client = self

        def schedule(coro_func, *args) -> None:
            loop.call_soon_threadsafe(lambda: asyncio.create_task(coro_func(*args)))

        class AgentAudioPort(pj.AudioMediaPort):
            def __init__(self, call_id: str, playback: PcmPlaybackBuffer) -> None:
                super().__init__()
                self.call_id = call_id
                self.playback = playback

                media_format = pj.MediaFormatAudio()
                media_format.type = pj.PJMEDIA_TYPE_AUDIO
                media_format.id = pj.PJMEDIA_FORMAT_L16
                media_format.clockRate = PjsipAudioSink.SAMPLE_RATE
                media_format.channelCount = 1
                media_format.bitsPerSample = 16
                media_format.frameTimeUsec = 20_000
                self.createPort(f"agent-{call_id}", media_format)

            def onFrameReceived(self, frame: object) -> None:  # noqa: N802
                pcm = bytes(frame.buf)[: int(frame.size)]
                if pcm:
                    loop.call_soon_threadsafe(
                        client._conversations.enqueue_pcm,
                        self.call_id,
                        pcm,
                    )

            def onFrameRequested(self, frame: object) -> None:  # noqa: N802
                pcm = self.playback.read(int(frame.size))
                frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
                frame.buf.assign_from_bytes(pcm)
                frame.size = len(pcm)

        class AgentCall(pj.Call):
            def __init__(self, account: object, native_call_id: int) -> None:
                super().__init__(account, native_call_id)
                self.native_call_id = native_call_id
                self.call_id = str(native_call_id)
                self.caller_id = ""
                self.playback = PcmPlaybackBuffer()
                self.sink = PjsipAudioSink(self.playback)
                self.audio_port: AgentAudioPort | None = None
                self.call_media = None
                self.media_started = False

            def _reply(self, status_code: int) -> None:
                parameter = pj.CallOpParam()
                parameter.statusCode = status_code
                super().answer(parameter)

            def signal_ringing(self) -> None:
                self._reply(pj.PJSIP_SC_RINGING)

            def accept(self) -> None:
                self._reply(pj.PJSIP_SC_OK)

            def reject_busy(self) -> None:
                self._reply(pj.PJSIP_SC_BUSY_HERE)

            def terminate(self) -> None:
                parameter = pj.CallOpParam()
                parameter.statusCode = pj.PJSIP_SC_DECLINE
                super().hangup(parameter)

            def onCallState(self, _parameter: object) -> None:  # noqa: N802
                try:
                    info = self.getInfo()
                except Exception:
                    log.exception("Could not query state for call %s", self.call_id)
                    return
                log.info(
                    "Call %s state=%s status=%s %s",
                    self.call_id,
                    info.stateText,
                    info.lastStatusCode,
                    info.lastReason,
                )
                if info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                    self.sink.close()
                    client._answer_policy.disconnected(
                        self.native_call_id,
                        int(info.lastStatusCode),
                        info.lastReason,
                    )
                    schedule(client._conversations.stop_call, self.call_id)
                    if client._account is not None:
                        client._account.defer_cleanup(self.native_call_id)

            def onCallMediaState(self, _parameter: object) -> None:  # noqa: N802
                if self.media_started:
                    return
                try:
                    info = self.getInfo()
                    active_audio = next(
                        media
                        for media in info.media
                        if media.type == pj.PJMEDIA_TYPE_AUDIO
                        and media.status == pj.PJSUA_CALL_MEDIA_ACTIVE
                    )
                    call_media = self.getAudioMedia(active_audio.index)
                    audio_port = AgentAudioPort(self.call_id, self.playback)
                    call_media.startTransmit(audio_port)
                    audio_port.startTransmit(call_media)
                except StopIteration:
                    return
                except Exception:
                    log.exception("Failed to connect media for call %s", self.call_id)
                    self.terminate()
                    return

                self.call_media = call_media
                self.audio_port = audio_port
                self.media_started = True
                schedule(
                    client._start_conversation,
                    self,
                )
                log.info("Call %s PJSIP audio bridge active", self.call_id)

        class AgentAccount(pj.Account):
            def __init__(self) -> None:
                super().__init__()
                self.calls: dict[int, AgentCall] = {}
                self.cleanup_ids: set[int] = set()

            def onRegState(self, parameter: object) -> None:  # noqa: N802
                info = self.getInfo()
                log.info(
                    "SIP registration active=%s status=%s %s",
                    info.regIsActive,
                    parameter.code,
                    parameter.reason,
                )

            def onIncomingCall(self, parameter: object) -> None:  # noqa: N802
                call = AgentCall(self, parameter.callId)
                self.calls[parameter.callId] = call
                try:
                    call.caller_id = caller_id_from_uri(call.getInfo().remoteUri)
                    client._answer_policy.offer(
                        parameter.callId,
                        call.caller_id or "unknown",
                        call,
                    )
                except Exception:
                    log.exception("Failed to process incoming call %s", parameter.callId)
                    with suppress(Exception):
                        call.reject_busy()

            def defer_cleanup(self, call_id: int) -> None:
                self.cleanup_ids.add(call_id)

            def cleanup(self) -> None:
                for call_id in self.cleanup_ids:
                    self.calls.pop(call_id, None)
                self.cleanup_ids.clear()

        endpoint = pj.Endpoint()
        self._endpoint = endpoint
        endpoint.libCreate()
        account = None
        try:
            endpoint_config = pj.EpConfig()
            endpoint_config.uaConfig.threadCnt = 0
            endpoint_config.uaConfig.mainThreadOnly = True
            endpoint_config.uaConfig.userAgent = "voip-agent-pjsip/0.2"
            endpoint_config.logConfig.level = self._s.pjsip_log_level
            endpoint_config.logConfig.consoleLevel = self._s.pjsip_log_level
            endpoint_config.medConfig.clockRate = PjsipAudioSink.SAMPLE_RATE
            endpoint_config.medConfig.sndClockRate = PjsipAudioSink.SAMPLE_RATE
            endpoint_config.medConfig.channelCount = 1
            endpoint_config.medConfig.audioFramePtime = 20
            endpoint.libInit(endpoint_config)

            transport_config = pj.TransportConfig()
            transport_config.port = self._s.pjsip_local_port
            transport_type = {
                "udp": pj.PJSIP_TRANSPORT_UDP,
                "tcp": pj.PJSIP_TRANSPORT_TCP,
            }[self._s.pjsip_transport]
            transport_id = endpoint.transportCreate(transport_type, transport_config)
            endpoint.libStart()
            endpoint.audDevManager().setNullDev()

            account_config = pj.AccountConfig()
            account_config.idUri = self.identity_uri
            account_config.regConfig.registrarUri = self.registrar_uri
            account_config.sipConfig.transportId = transport_id
            account_config.sipConfig.authCreds.append(
                pj.AuthCredInfo(
                    "digest",
                    "*",
                    self._s.fritzbox_sip_username,
                    0,
                    self._s.fritzbox_sip_password,
                )
            )

            account = AgentAccount()
            self._account = account
            account.create(account_config)
            log.info(
                "PJSIP agent started: registrar=%s transport=%s local_port=%s",
                self.registrar_uri,
                self._s.pjsip_transport,
                self._s.pjsip_local_port,
            )

            while not self._stop_requested:
                endpoint.libHandleEvents(0)
                self._answer_policy.tick()
                account.cleanup()
                await asyncio.sleep(self._s.pjsip_event_poll_ms / 1000)
        finally:
            log.info("Shutting down PJSIP agent")
            self._answer_policy.terminate_all()
            await self._conversations.stop_all()
            if account is not None:
                account.cleanup()
                account.shutdown()
            self._account = None
            endpoint.libDestroy()
            self._endpoint = None
