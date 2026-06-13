"""ARI client — connects to Asterisk WebSocket event stream and drives the voice pipeline."""

import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx
import websockets

from agent.audio import VadBuffer, alaw_decode, resample_8k_to_16k
from agent.config import Settings
from agent.pipeline import VoicePipeline
from agent.rtp import RtpServer
from agent.session import CallSession, SessionState
from agent.turn_detector import TurnDetectorClient

log = logging.getLogger(__name__)


class AriClient:
    # Bound the inbound audio backlog so a stalled consumer cannot grow memory
    # without limit; ~100 frames = 2 s of 20 ms audio. Oldest-relevant frames
    # are dropped on overflow rather than buffering a growing delay.
    AUDIO_QUEUE_MAXSIZE = 100
    RTP_BIND_ATTEMPTS = 50

    # Prefix for the ExternalMedia channels we create. Asterisk routes such a
    # channel into the same Stasis app, firing a second StasisStart — without a
    # filter that would recursively spawn a call (session/RTP/bridge) for our
    # own media leg. Shared by the StasisStart filter and _create_external_media
    # so the two can never drift.
    _EXT_PREFIX = "ext-"

    # States from which a caller may barge in. PROCESSING is included because
    # in the streaming pipeline the PROCESSING->first-chunk window lasts
    # seconds; the caller must be able to interrupt mid-generation.
    _INTERRUPTIBLE_STATES = (
        SessionState.LISTENING,
        SessionState.SPEAKING,
        SessionState.PROCESSING,
    )

    def __init__(
        self,
        settings: Settings,
        pipeline: VoicePipeline,
        turn_detector: TurnDetectorClient | None = None,
    ) -> None:
        self._s = settings
        self._pipeline = pipeline
        self._turn_detector = turn_detector
        self._sessions: dict[str, CallSession] = {}
        self._rtp_servers: dict[str, RtpServer] = {}
        self._vad_buffers: dict[str, VadBuffer] = {}
        # Second buffer used only when turn detection is active: barge-in
        # (SPEAKING/PROCESSING) keeps the legacy 800ms floor here while the
        # primary buffer runs the lowered turn-end floor.
        self._bargein_buffers: dict[str, VadBuffer] = {}
        self._playback_tasks: dict[str, asyncio.Task] = {}
        self._audio_queues: dict[str, asyncio.Queue[bytes]] = {}
        self._out_queues: dict[str, asyncio.Queue] = {}
        self._consumer_tasks: dict[str, asyncio.Task] = {}
        # Per-call turn lock + monotonically increasing generation id. The lock
        # serializes the cancel-playback / process-turn / start-playback
        # critical section; the generation id lets a playback or turn that was
        # superseded by a barge-in detect that it is stale and emit nothing.
        self._turn_locks: dict[str, asyncio.Lock] = {}
        self._generation: dict[str, int] = {}
        self._rtp_port_counter = settings.rtp_port
        # One shared HTTP client for all ARI REST calls instead of a fresh
        # connection pool per request.
        self._http: httpx.AsyncClient | None = None
        self._running = True

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient()
        return self._http

    def _turn_active(self) -> bool:
        return self._turn_detector is not None and self._s.turn_detection_enabled

    async def aclose(self) -> None:
        self._running = False
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _ws_url(self) -> str:
        return (
            f"ws://{self._s.ari_base_url.split('://', 1)[-1]}"
            f"/ari/events?api_key={self._s.ari_username}:{self._s.ari_password}"
            f"&app={self._s.ari_app_name}&subscribeAll=true"
        )

    async def run(self) -> None:
        # Reconnect with capped exponential backoff so a transient ARI restart
        # or network blip does not permanently drop the agent's event stream.
        backoff = 1.0
        while self._running:
            try:
                log.info("Connecting to ARI at %s", self._s.ari_base_url)
                async with websockets.connect(self._ws_url()) as ws:
                    backoff = 1.0  # reset on a successful connect
                    async for raw in ws:
                        try:
                            event = json.loads(raw)
                            await self._handle_event(event)
                        except Exception:
                            log.exception("Error handling ARI event")
            except asyncio.CancelledError:
                raise
            except Exception:
                if not self._running:
                    break
                log.warning("ARI websocket lost; reconnecting in %.1fs", backoff, exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _handle_event(self, event: dict) -> None:
        t = event.get("type")
        if t == "StasisStart":
            ch = event["channel"]
            ch_id = ch["id"]
            # Our own ExternalMedia leg re-enters Stasis; ignore it or we
            # recurse (new session/RTP/bridge for the media channel).
            if ch_id.startswith(self._EXT_PREFIX):
                log.debug("Ignoring StasisStart for ExternalMedia channel %s", ch_id)
                return
            caller_id = ch.get("caller", {}).get("number", "")
            asyncio.create_task(self._setup_call(ch_id, caller_id))
        elif t == "StasisEnd":
            ch_id = event["channel"]["id"]
            await self._teardown_call(ch_id)

    async def _setup_call(self, channel_id: str, caller_id: str) -> None:
        # Transactional: any failure after we start registering per-call state
        # must roll the whole thing back, otherwise a half-built call leaks a
        # session, RTP socket, and consumer task. _setup_call runs in a detached
        # task, so we also can't rely on a caller to observe the exception.
        try:
            session = CallSession(
                call_id=channel_id,
                caller_id=caller_id,
                history=[],
                created_at=datetime.now(timezone.utc),
            )
            self._sessions[channel_id] = session

            # Bind the RTP server, skipping ports that are already in use instead
            # of assuming the next counter value is free (the old code wrapped the
            # counter without checking live binds, so a long-lived call could
            # collide with a reused port).
            rtp_server, rtp_port = await self._bind_rtp_server(channel_id)
            self._rtp_servers[channel_id] = rtp_server
            if self._turn_active():
                self._vad_buffers[channel_id] = VadBuffer(
                    silence_threshold_ms=self._s.turn_vad_silence_ms
                )
                self._bargein_buffers[channel_id] = VadBuffer()
            else:
                self._vad_buffers[channel_id] = VadBuffer()

            # Single consumer drains a bounded queue, so per-packet work no longer
            # spawns a task per datagram (~50/s/call) and exceptions surface in one
            # place. The datagram callback only enqueues.
            self._audio_queues[channel_id] = asyncio.Queue(maxsize=self.AUDIO_QUEUE_MAXSIZE)
            self._turn_locks[channel_id] = asyncio.Lock()
            self._generation[channel_id] = 0
            self._consumer_tasks[channel_id] = asyncio.create_task(self._audio_consumer(channel_id))

            ext_channel_id = await self._create_external_media(channel_id, rtp_port)
            await self._bridge_channels(channel_id, ext_channel_id)

            alaw = await self._pipeline.synthesize_alaw(self._s.greeting_text)
            gen = self._generation[channel_id]
            task = asyncio.create_task(self._play_audio(channel_id, alaw, session, gen))
            self._playback_tasks[channel_id] = task
            log.info("Call %s from %s ready", channel_id, caller_id)
        except Exception:
            # Local teardown frees session/RTP/consumer/queue (the leaked
            # resources from the review). It does NOT hang up the Asterisk-side
            # channels / ext leg — StasisEnd will fire its own _teardown_call
            # when Asterisk tears the channel down.
            log.exception("Call setup failed for %s; tearing down", channel_id)
            await self._teardown_call(channel_id)

    def _next_rtp_port(self) -> int:
        port = self._rtp_port_counter
        self._rtp_port_counter += 2
        if self._rtp_port_counter > 65534:
            self._rtp_port_counter = self._s.rtp_port  # wrap around
        return port

    async def _bind_rtp_server(self, channel_id: str) -> tuple[RtpServer, int]:
        for _ in range(self.RTP_BIND_ATTEMPTS):
            port = self._next_rtp_port()
            server = RtpServer(
                host=self._s.rtp_bind_host,
                port=port,
                on_audio=lambda payload, cid=channel_id: self._enqueue_audio(cid, payload),
            )
            try:
                await server.start()
            except OSError:
                log.warning("RTP port %d unavailable, trying next", port)
                continue
            return server, port
        raise RuntimeError("no free RTP port available")

    def _enqueue_audio(self, channel_id: str, payload: bytes) -> None:
        queue = self._audio_queues.get(channel_id)
        if queue is None:
            return
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            log.warning("Audio queue full for %s; dropping frame", channel_id)

    async def _audio_consumer(self, channel_id: str) -> None:
        queue = self._audio_queues.get(channel_id)
        if queue is None:
            return
        try:
            while True:
                payload = await queue.get()
                try:
                    await self._on_audio(channel_id, payload)
                except Exception:
                    log.exception("Audio processing failed for %s", channel_id)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            pass

    async def _teardown_call(self, channel_id: str) -> None:
        for tasks in (self._playback_tasks, self._consumer_tasks):
            task = tasks.pop(channel_id, None)
            if task and not task.done():
                task.cancel()
        self._audio_queues.pop(channel_id, None)
        self._out_queues.pop(channel_id, None)
        self._turn_locks.pop(channel_id, None)
        self._generation.pop(channel_id, None)
        self._vad_buffers.pop(channel_id, None)
        self._bargein_buffers.pop(channel_id, None)
        session = self._sessions.pop(channel_id, None)
        if session:
            session.transition(SessionState.ENDED)
        rtp = self._rtp_servers.pop(channel_id, None)
        if rtp:
            rtp.close()
        log.info("Call %s ended", channel_id)

    async def _play_audio(
        self, channel_id: str, alaw: bytes, session: CallSession, gen: int
    ) -> None:
        rtp = self._rtp_servers.get(channel_id)
        if not rtp:
            return
        # A barge-in may have superseded this turn between scheduling and start.
        if self._generation.get(channel_id) != gen:
            return
        session.transition(SessionState.SPEAKING)
        try:
            await rtp.stream_audio(alaw)
            # Only return to LISTENING if we are still the current generation;
            # otherwise a newer turn owns the state and we must not touch it.
            if self._generation.get(channel_id) == gen and session.state == SessionState.SPEAKING:
                session.transition(SessionState.LISTENING)
                vad = self._vad_buffers.get(channel_id)
                if vad:
                    vad.reset()
        except asyncio.CancelledError:
            pass

    async def _play_stream(self, channel_id, session, gen, pcm) -> None:
        """Drive a streaming turn: feed pipeline aLaw chunks into the RTP
        chunk drain, entering SPEAKING on the first chunk. A supervising
        TaskGroup owns producer (pipeline) + consumer (RTP drain); a single
        cancel on barge-in tears both down."""
        rtp = self._rtp_servers.get(channel_id)
        if not rtp:
            return
        if self._generation.get(channel_id) != gen:
            return

        out: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._out_queues[channel_id] = out
        first = {"seen": False}

        async def produce():
            try:
                async for alaw in self._pipeline.process_turn_stream(session, pcm):
                    if self._generation.get(channel_id) != gen:
                        break
                    if not first["seen"]:
                        first["seen"] = True
                        if session.state == SessionState.PROCESSING:
                            session.transition(SessionState.SPEAKING)
                    await out.put(alaw)
            finally:
                await out.put(None)  # sentinel

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(produce())
                tg.create_task(rtp.stream_audio_chunks(out))
        except* Exception:
            log.exception("streaming turn failed for %s", channel_id)
        finally:
            self._out_queues.pop(channel_id, None)

        # Return to LISTENING if still current. Cover both SPEAKING (audio
        # emitted) and PROCESSING (turn yielded nothing) so the FSM never
        # strands the call mid-turn.
        if self._generation.get(channel_id) == gen and session.state in (
            SessionState.SPEAKING,
            SessionState.PROCESSING,
        ):
            session.transition(SessionState.LISTENING)
            vad = self._vad_buffers.get(channel_id)
            if vad:
                vad.reset()

    async def _on_audio(self, channel_id: str, alaw_payload: bytes) -> None:
        session = self._sessions.get(channel_id)
        if not session:
            return

        if session.state not in self._INTERRUPTIBLE_STATES:
            return

        vad = self._vad_buffers.get(channel_id)
        if vad is None:
            return

        pcm_8k = alaw_decode(alaw_payload)
        pcm_16k = resample_8k_to_16k(pcm_8k)
        speech = vad.add_frame(pcm_16k)

        if speech is None:
            return

        lock = self._turn_locks.get(channel_id)
        if lock is None:
            return
        async with lock:
            # State may have advanced while we waited for the lock.
            if session.state not in self._INTERRUPTIBLE_STATES:
                return

            # Bump the generation: any in-flight playback for the prior
            # generation will now no-op on completion, and the turn we start
            # below tags its own audio with this id.
            gen = self._generation.get(channel_id, 0) + 1
            self._generation[channel_id] = gen

            # Tear down the in-flight turn (streaming producer chain or a
            # greeting playback). The bumped generation already neutralizes any
            # late chunk; the cancel stops the work immediately.
            task = self._playback_tasks.pop(channel_id, None)
            if task and not task.done():
                task.cancel()

            # Move back to LISTENING before the new turn enters PROCESSING.
            # Both SPEAKING and the now-interruptible PROCESSING are valid
            # sources for this transition.
            if session.state in (SessionState.SPEAKING, SessionState.PROCESSING):
                session.transition(SessionState.LISTENING)
            vad.reset()

            # Streaming turn: process_turn_stream owns PROCESSING entry,
            # _play_stream drives SPEAKING (first chunk) -> LISTENING. The
            # stale-generation guard now lives inside _play_stream.
            task = asyncio.create_task(self._play_stream(channel_id, session, gen, speech))
            self._playback_tasks[channel_id] = task

    async def _create_external_media(self, channel_id: str, rtp_port: int) -> str:
        client = self._client()
        resp = await client.post(
            f"{self._s.ari_base_url}/ari/channels/externalMedia",
            auth=(self._s.ari_username, self._s.ari_password),
            params={
                "app": self._s.ari_app_name,
                "external_host": f"{self._s.rtp_advertise_host}:{rtp_port}",
                "format": "alaw",
                "channelId": f"{self._EXT_PREFIX}{channel_id}",
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()["id"]

    async def originate(self, to_number: str) -> None:
        client = self._client()
        resp = await client.post(
            f"{self._s.ari_base_url}/ari/channels",
            auth=(self._s.ari_username, self._s.ari_password),
            params={
                "endpoint": f"PJSIP/{to_number}@fritzbox-endpoint",
                "callerId": self._s.caller_id,
                "app": self._s.ari_app_name,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        log.info("Outbound call to %s initiated", to_number)

    async def _bridge_channels(self, channel_id: str, ext_channel_id: str) -> None:
        client = self._client()
        resp = await client.post(
            f"{self._s.ari_base_url}/ari/bridges",
            auth=(self._s.ari_username, self._s.ari_password),
            params={"type": "mixing", "bridgeId": f"bridge-{channel_id}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        bridge_id = resp.json()["id"]
        await client.post(
            f"{self._s.ari_base_url}/ari/bridges/{bridge_id}/addChannel",
            auth=(self._s.ari_username, self._s.ari_password),
            params={"channel": f"{channel_id},{ext_channel_id}"},
            timeout=5.0,
        )
