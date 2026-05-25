"""ARI client — connects to Asterisk WebSocket event stream and drives the voice pipeline."""
import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx
import websockets

from agent.audio import alaw_decode, resample_8k_to_16k, VadBuffer
from agent.config import Settings
from agent.pipeline import VoicePipeline
from agent.rtp import RtpServer
from agent.session import CallSession, SessionState

log = logging.getLogger(__name__)


class AriClient:
    def __init__(self, settings: Settings, pipeline: VoicePipeline) -> None:
        self._s = settings
        self._pipeline = pipeline
        self._sessions: dict[str, CallSession] = {}
        self._rtp_servers: dict[str, RtpServer] = {}
        self._vad_buffers: dict[str, VadBuffer] = {}
        self._playback_tasks: dict[str, asyncio.Task] = {}
        self._rtp_port_counter = settings.rtp_port

    async def run(self) -> None:
        url = (
            f"ws://{self._s.ari_base_url.split('://', 1)[-1]}"
            f"/ari/events?api_key={self._s.ari_username}:{self._s.ari_password}"
            f"&app={self._s.ari_app_name}&subscribeAll=true"
        )
        log.info("Connecting to ARI at %s", url)
        async with websockets.connect(url) as ws:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                    await self._handle_event(event)
                except Exception:
                    log.exception("Error handling ARI event")

    async def _handle_event(self, event: dict) -> None:
        t = event.get("type")
        if t == "StasisStart":
            ch = event["channel"]
            asyncio.create_task(self._setup_call(ch["id"], ch["caller"]["number"]))
        elif t == "StasisEnd":
            ch_id = event["channel"]["id"]
            await self._teardown_call(ch_id)

    async def _setup_call(self, channel_id: str, caller_id: str) -> None:
        session = CallSession(
            call_id=channel_id,
            caller_id=caller_id,
            history=[],
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[channel_id] = session

        rtp_port = self._rtp_port_counter
        self._rtp_port_counter += 2
        if self._rtp_port_counter > 65534:
            self._rtp_port_counter = self._s.rtp_port  # wrap around

        loop = asyncio.get_running_loop()
        rtp_server = RtpServer(
            host=self._s.rtp_bind_host,
            port=rtp_port,
            on_audio=lambda payload: loop.create_task(
                self._on_audio(channel_id, payload)
            ),
        )
        await rtp_server.start()
        self._rtp_servers[channel_id] = rtp_server
        self._vad_buffers[channel_id] = VadBuffer()

        ext_channel_id = await self._create_external_media(channel_id, rtp_port)
        await self._bridge_channels(channel_id, ext_channel_id)

        alaw = await self._pipeline.synthesize_alaw(self._s.greeting_text)
        task = asyncio.create_task(self._play_audio(channel_id, alaw, session))
        self._playback_tasks[channel_id] = task
        log.info("Call %s from %s ready", channel_id, caller_id)

    async def _teardown_call(self, channel_id: str) -> None:
        task = self._playback_tasks.pop(channel_id, None)
        if task and not task.done():
            task.cancel()
        self._vad_buffers.pop(channel_id, None)
        session = self._sessions.pop(channel_id, None)
        if session:
            session.transition(SessionState.ENDED)
        rtp = self._rtp_servers.pop(channel_id, None)
        if rtp:
            rtp.close()
        log.info("Call %s ended", channel_id)

    async def _play_audio(self, channel_id: str, alaw: bytes, session: CallSession) -> None:
        rtp = self._rtp_servers.get(channel_id)
        if not rtp:
            return
        session.transition(SessionState.SPEAKING)
        try:
            await rtp.stream_audio(alaw)
            if session.state == SessionState.SPEAKING:
                session.transition(SessionState.LISTENING)
            vad = self._vad_buffers.get(channel_id)
            if vad:
                vad.reset()
        except asyncio.CancelledError:
            pass

    async def _on_audio(self, channel_id: str, alaw_payload: bytes) -> None:
        session = self._sessions.get(channel_id)
        if not session:
            return

        state = session.state
        if state not in (SessionState.LISTENING, SessionState.SPEAKING):
            return

        vad = self._vad_buffers.get(channel_id)
        if vad is None:
            return

        pcm_8k = alaw_decode(alaw_payload)
        pcm_16k = resample_8k_to_16k(pcm_8k)
        speech = vad.add_frame(pcm_16k)

        if speech is None:
            return

        task = self._playback_tasks.pop(channel_id, None)
        if task and not task.done():
            task.cancel()

        if session.state == SessionState.SPEAKING:
            session.transition(SessionState.LISTENING)
        vad.reset()

        response_alaw = await self._pipeline.process_turn(session, speech)
        task = asyncio.create_task(self._play_audio(channel_id, response_alaw, session))
        self._playback_tasks[channel_id] = task

    async def _create_external_media(self, channel_id: str, rtp_port: int) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._s.ari_base_url}/ari/channels/externalMedia",
                auth=(self._s.ari_username, self._s.ari_password),
                params={
                    "app": self._s.ari_app_name,
                    "external_host": f"{self._s.rtp_bind_host}:{rtp_port}",
                    "format": "alaw",
                    "channelId": f"ext-{channel_id}",
                },
                timeout=5.0,
            )
        resp.raise_for_status()
        return resp.json()["id"]

    async def originate(self, to_number: str) -> None:
        async with httpx.AsyncClient() as client:
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
        async with httpx.AsyncClient() as client:
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
