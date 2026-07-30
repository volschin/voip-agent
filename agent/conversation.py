"""Transport-neutral call conversation orchestration.

The media transport provides 16 kHz mono PCM from the caller and accepts the
pipeline's 16 kHz mono PCM output. SIP/RTP lifecycle belongs to the transport
adapter; VAD, turn detection, barge-in, and pipeline state live here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from agent.audio import PCM16_PLAYBACK_BLOCK_BYTES, VadBuffer
from agent.config import Settings
from agent.pipeline import VoicePipeline
from agent.priority import LeaseHandle, PriorityLeaseClient, PriorityUnavailable
from agent.session import CallSession, SessionState
from agent.turn_detector import TurnDetector

log = logging.getLogger(__name__)


class AudioSink(Protocol):
    """Playback boundary implemented by the active telephony transport."""

    async def play_pcm16(self, pcm: bytes) -> None: ...

    async def play_pcm16_chunks(self, queue: asyncio.Queue) -> None: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...


@dataclass
class _TransportTermination:
    callback: Callable[[], None] | None
    invoked: bool = False

    def invoke(self) -> None:
        if self.callback is None or self.invoked:
            return
        self.invoked = True
        self.callback()


@dataclass
class _PendingStart:
    sink: AudioSink
    termination: _TransportTermination


class _PcmByteQueue:
    """Bound playback handoff by PCM bytes, independent of item count."""

    def __init__(self, max_bytes: int) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._max_bytes = max_bytes
        self._queued_bytes = 0
        self._space_available = asyncio.Event()
        self._space_available.set()

    @property
    def queued_bytes(self) -> int:
        return self._queued_bytes

    def full(self) -> bool:
        return self._queued_bytes >= self._max_bytes

    async def put(self, item: bytes | None) -> None:
        size = len(item) if item is not None else 0
        if size > self._max_bytes:
            raise ValueError("PCM playback block exceeds queue byte limit")
        while size and self._queued_bytes + size > self._max_bytes:
            self._space_available.clear()
            await self._space_available.wait()
        self._queued_bytes += size
        self._queue.put_nowait(item)

    async def get(self) -> bytes | None:
        item = await self._queue.get()
        self._release(item)
        return item

    def get_nowait(self) -> bytes | None:
        item = self._queue.get_nowait()
        self._release(item)
        return item

    def _release(self, item: bytes | None) -> None:
        if item is not None:
            self._queued_bytes -= len(item)
            self._space_available.set()


class ConversationManager:
    """Drive one or more calls independently from SIP/ARI transport details."""

    AUDIO_QUEUE_MAXSIZE = 100
    PRIORITY_HEARTBEAT_SECONDS = 10
    _MIN_CLASSIFY_SAMPLES = 1600
    _INTERRUPTIBLE_STATES = (
        SessionState.LISTENING,
        SessionState.SPEAKING,
        SessionState.PROCESSING,
    )

    def __init__(
        self,
        settings: Settings,
        pipeline: VoicePipeline,
        *,
        priority_client: PriorityLeaseClient,
        turn_detector: TurnDetector | None = None,
    ) -> None:
        self._s = settings
        self._pipeline = pipeline
        self._priority_client = priority_client
        self._turn_detector = turn_detector
        self._sessions: dict[str, CallSession] = {}
        self._sinks: dict[str, AudioSink] = {}
        self._vad_buffers: dict[str, VadBuffer] = {}
        self._bargein_buffers: dict[str, VadBuffer] = {}
        self._playback_tasks: dict[str, asyncio.Task] = {}
        self._audio_queues: dict[str, asyncio.Queue[bytes]] = {}
        self._out_queues: dict[str, _PcmByteQueue] = {}
        self._consumer_tasks: dict[str, asyncio.Task] = {}
        self._turn_locks: dict[str, asyncio.Lock] = {}
        self._generation: dict[str, int] = {}
        self._priority_leases: dict[str, LeaseHandle] = {}
        self._priority_tasks: dict[str, asyncio.Task] = {}
        self._pending_starts: dict[str, _PendingStart] = {}
        self._transport_terminations: dict[str, _TransportTermination] = {}

    @property
    def call_count(self) -> int:
        return len(self._sessions)

    def _turn_active(self) -> bool:
        return self._turn_detector is not None and self._s.turn_detection_enabled

    def _reset_vad(self, call_id: str) -> None:
        for buffer in (self._vad_buffers.get(call_id), self._bargein_buffers.get(call_id)):
            if buffer:
                buffer.reset()

    async def start_call(
        self,
        call_id: str,
        caller_id: str,
        sink: AudioSink,
        *,
        terminate_transport: Callable[[], None] | None = None,
    ) -> bool:
        if call_id in self._sessions:
            return True
        if call_id in self._pending_starts:
            return False
        pending = _PendingStart(sink, _TransportTermination(terminate_transport))
        self._pending_starts[call_id] = pending
        try:
            lease = await self._priority_client.acquire()
        except PriorityUnavailable:
            log.warning("Voice priority unavailable; call %s not started", call_id)
            if self._pending_starts.get(call_id) is pending:
                self._pending_starts.pop(call_id, None)
                sink.clear()
                sink.close()
                pending.termination.invoke()
            return False
        except asyncio.CancelledError:
            if self._pending_starts.get(call_id) is pending:
                self._pending_starts.pop(call_id, None)
                sink.clear()
                sink.close()
            raise

        if self._pending_starts.get(call_id) is not pending:
            try:
                await lease.release()
            except PriorityUnavailable:
                log.warning("Voice priority release failed for stale call %s", call_id)
            return False
        self._pending_starts.pop(call_id, None)
        self._priority_leases[call_id] = lease
        self._transport_terminations[call_id] = pending.termination

        session = CallSession(
            call_id=call_id,
            caller_id=caller_id,
            history=[],
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[call_id] = session
        self._sinks[call_id] = sink
        if self._turn_active():
            self._vad_buffers[call_id] = VadBuffer(silence_threshold_ms=self._s.turn_vad_silence_ms)
            self._bargein_buffers[call_id] = VadBuffer()
        else:
            self._vad_buffers[call_id] = VadBuffer()
        self._audio_queues[call_id] = asyncio.Queue(maxsize=self.AUDIO_QUEUE_MAXSIZE)
        self._turn_locks[call_id] = asyncio.Lock()
        self._generation[call_id] = 0
        self._consumer_tasks[call_id] = asyncio.create_task(self._audio_consumer(call_id))
        self._priority_tasks[call_id] = asyncio.create_task(
            self._priority_heartbeat(call_id, lease)
        )

        try:
            pcm = await self._pipeline.synthesize_pcm16(self._s.greeting_text)
        except Exception:
            log.exception("Greeting synthesis failed for call %s", call_id)
            await self.stop_call(call_id, terminate_transport=True)
            return False

        if self._sessions.get(call_id) is not session:
            return False
        generation = self._generation[call_id]
        task = asyncio.create_task(self._play_pcm16(call_id, pcm, session, generation))
        self._playback_tasks[call_id] = task
        log.info("Call %s conversation ready", call_id)
        return True

    async def _priority_heartbeat(
        self,
        call_id: str,
        lease: LeaseHandle,
    ) -> None:
        try:
            while self._priority_leases.get(call_id) is lease:
                await asyncio.sleep(self.PRIORITY_HEARTBEAT_SECONDS)
                await lease.renew()
        except asyncio.CancelledError:
            raise
        except PriorityUnavailable:
            log.warning("Voice priority renewal failed; ending call %s", call_id)
            await self.stop_call(call_id, terminate_transport=True)

    def enqueue_pcm(self, call_id: str, pcm_16k: bytes) -> None:
        """Enqueue one PJSIP media frame; safe to schedule from another thread."""

        if not pcm_16k or len(pcm_16k) % 2:
            return
        queue = self._audio_queues.get(call_id)
        if queue is None:
            return
        try:
            queue.put_nowait(pcm_16k)
        except asyncio.QueueFull:
            log.warning("Audio queue full for %s; dropping frame", call_id)

    async def _audio_consumer(self, call_id: str) -> None:
        queue = self._audio_queues.get(call_id)
        if queue is None:
            return
        try:
            while True:
                frame = await queue.get()
                try:
                    await self._on_pcm(call_id, frame)
                except Exception:
                    log.exception("Audio processing failed for %s", call_id)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            pass

    async def stop_call(self, call_id: str, *, terminate_transport: bool = False) -> None:
        pending = self._pending_starts.pop(call_id, None)
        if pending is not None:
            pending.sink.clear()
            pending.sink.close()
        heartbeat = self._priority_tasks.pop(call_id, None)
        current = asyncio.current_task()
        if heartbeat and heartbeat is not current and not heartbeat.done():
            heartbeat.cancel()
        cancelled_tasks = []
        lease = self._priority_leases.pop(call_id, None)
        for tasks in (self._playback_tasks, self._consumer_tasks):
            task = tasks.pop(call_id, None)
            if task and task is not current and not task.done():
                task.cancel()
                cancelled_tasks.append(task)
        if cancelled_tasks:
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)
        self._audio_queues.pop(call_id, None)
        self._out_queues.pop(call_id, None)
        self._turn_locks.pop(call_id, None)
        self._generation.pop(call_id, None)
        self._vad_buffers.pop(call_id, None)
        self._bargein_buffers.pop(call_id, None)
        session = self._sessions.pop(call_id, None)
        if session and session.state is not SessionState.ENDED:
            session.transition(SessionState.ENDED)
        sink = self._sinks.pop(call_id, None)
        if sink:
            sink.clear()
            sink.close()
        if lease is not None:
            try:
                await lease.release()
            except PriorityUnavailable:
                log.warning("Voice priority release failed for call %s", call_id)
        termination = self._transport_terminations.pop(call_id, None)
        if terminate_transport and termination is not None:
            try:
                termination.invoke()
            except Exception:
                log.exception("Transport termination failed for call %s", call_id)
        log.info("Call %s conversation ended", call_id)

    async def stop_all(self) -> None:
        for call_id in set(self._sessions) | set(self._pending_starts):
            await self.stop_call(call_id)

    async def _play_pcm16(
        self,
        call_id: str,
        pcm: bytes,
        session: CallSession,
        generation: int,
    ) -> None:
        sink = self._sinks.get(call_id)
        if sink is None or self._generation.get(call_id) != generation:
            return
        session.transition(SessionState.SPEAKING)
        try:
            await sink.play_pcm16(pcm)
            if (
                self._generation.get(call_id) == generation
                and session.state is SessionState.SPEAKING
            ):
                session.transition(SessionState.LISTENING)
                self._reset_vad(call_id)
        except asyncio.CancelledError:
            sink.clear()
            raise
        except Exception:
            log.exception("Complete playback failed for %s", call_id)
            sink.clear()
            if (
                self._generation.get(call_id) == generation
                and session.state is SessionState.SPEAKING
            ):
                session.transition(SessionState.LISTENING)
                self._reset_vad(call_id)

    async def _play_stream(
        self,
        call_id: str,
        session: CallSession,
        generation: int,
        pcm: np.ndarray,
    ) -> None:
        sink = self._sinks.get(call_id)
        if sink is None or self._generation.get(call_id) != generation:
            return

        out = _PcmByteQueue(max_bytes=PCM16_PLAYBACK_BLOCK_BYTES)
        self._out_queues[call_id] = out
        first_chunk_seen = False

        async def produce() -> None:
            nonlocal first_chunk_seen
            async for pcm_chunk in self._pipeline.process_turn_stream(session, pcm):
                if self._generation.get(call_id) != generation:
                    break
                if not first_chunk_seen:
                    first_chunk_seen = True
                    if session.state is SessionState.PROCESSING:
                        session.transition(SessionState.SPEAKING)
                await out.put(pcm_chunk)
            await out.put(None)

        try:
            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(produce())
                    group.create_task(sink.play_pcm16_chunks(out))
            except* Exception:
                sink.clear()
                log.exception("Streaming turn failed for %s", call_id)
        except asyncio.CancelledError:
            sink.clear()
            raise
        finally:
            self._out_queues.pop(call_id, None)

        if self._generation.get(call_id) == generation and session.state in (
            SessionState.SPEAKING,
            SessionState.PROCESSING,
        ):
            session.transition(SessionState.LISTENING)
            self._reset_vad(call_id)

    async def _on_pcm(self, call_id: str, pcm_bytes: bytes) -> None:
        session = self._sessions.get(call_id)
        if session is None or session.state not in self._INTERRUPTIBLE_STATES:
            return
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).copy()

        if self._turn_active() and session.state is SessionState.LISTENING:
            vad = self._vad_buffers.get(call_id)
            if vad is None:
                return
            candidate = vad.add_frame_candidate(pcm)
            if candidate is None:
                return
            speech = await self._gate_turn_end(call_id, vad, candidate)
            if speech is None:
                return
        else:
            vad = (
                self._bargein_buffers.get(call_id)
                if self._turn_active()
                else self._vad_buffers.get(call_id)
            )
            if vad is None:
                return
            speech = vad.add_frame(pcm)
            if speech is None:
                return

        lock = self._turn_locks.get(call_id)
        if lock is None:
            return
        async with lock:
            if session.state not in self._INTERRUPTIBLE_STATES:
                return
            interrupted_response = session.state in (
                SessionState.SPEAKING,
                SessionState.PROCESSING,
            )
            generation = self._generation.get(call_id, 0) + 1
            self._generation[call_id] = generation

            task = self._playback_tasks.pop(call_id, None)
            if task and not task.done():
                if interrupted_response:
                    session.previous_response_interrupted = True
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            sink = self._sinks.get(call_id)
            if sink:
                sink.clear()

            if session.state in (SessionState.SPEAKING, SessionState.PROCESSING):
                session.transition(SessionState.LISTENING)
            self._reset_vad(call_id)

            task = asyncio.create_task(self._play_stream(call_id, session, generation, speech))
            self._playback_tasks[call_id] = task

    async def _gate_turn_end(
        self,
        call_id: str,
        vad: VadBuffer,
        candidate: np.ndarray,
    ) -> np.ndarray | None:
        if vad.at_cap or len(candidate) < self._MIN_CLASSIFY_SAMPLES:
            return candidate
        try:
            complete = await self._turn_detector.classify(candidate)
        except Exception:
            log.warning("Turn classify failed for %s; flushing (degrade)", call_id)
            return candidate
        session = self._sessions.get(call_id)
        if session is None or session.state is not SessionState.LISTENING:
            return None
        if complete:
            return candidate
        vad.continue_speech()
        return None
