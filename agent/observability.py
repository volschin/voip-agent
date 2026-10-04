"""Read-only native health and bounded, content-free Prometheus telemetry."""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

_STAGES = ("stt", "llm", "tts")


@dataclass
class AgentStatus:
    initialized: bool = False
    detector_loaded: bool = False
    registered: bool = False
    last_heartbeat: float = field(default_factory=time.monotonic)
    active_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    priority_errors: int = 0
    stages: dict = field(default_factory=lambda: {name: [0, 0.0, 0] for name in _STAGES})

    def initialize(self, *, detector_loaded: bool) -> None:
        self.initialized = True
        self.detector_loaded = detector_loaded
        self.tick()

    def tick(self) -> None:
        self.last_heartbeat = time.monotonic()

    def live(self) -> bool:
        return time.monotonic() - self.last_heartbeat < 30

    def ready(self) -> bool:
        return self.initialized and self.detector_loaded and self.registered and self.live()

    def priority_error(self) -> None:
        self.priority_errors += 1

    def observe_stage(self, stage: str, duration: float, *, error: bool = False) -> None:
        item = self.stages[stage]
        item[0] += 1
        item[1] += duration
        item[2] += int(error)

    @asynccontextmanager
    async def measure(self, stage: str):
        start = time.monotonic()
        failed = False
        try:
            yield
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
            raise
        finally:
            self.observe_stage(stage, time.monotonic() - start, error=failed)

    def wrap(self, stage, call):
        async def measured(*args, **kwargs):
            async with self.measure(stage):
                return await call(*args, **kwargs)

        return measured

    def wrap_stream(self, stage, call):
        async def measured(*args, **kwargs):
            async with self.measure(stage):
                async for item in call(*args, **kwargs):
                    yield item

        return measured

    def metrics(self) -> str:
        scalars = {
            "sip_registered": ("gauge", int(self.registered)),
            "detector_loaded": ("gauge", int(self.detector_loaded)),
            "calls_active": ("gauge", self.active_calls),
            "calls_completed_total": ("counter", self.completed_calls),
            "calls_failed_total": ("counter", self.failed_calls),
            "priority_errors_total": ("counter", self.priority_errors),
        }
        lines = []
        for name, (kind, value) in scalars.items():
            name = "voip_agent_" + name
            lines.extend([f"# TYPE {name} {kind}", f"{name} {value}"])
        lines.extend(
            [
                "# TYPE voip_agent_stage_duration_seconds summary",
                "# TYPE voip_agent_stage_errors_total counter",
            ]
        )
        for stage, (count, duration, errors) in self.stages.items():
            label = f'{{stage="{stage}"}}'
            lines.extend(
                [
                    f"voip_agent_stage_duration_seconds_count{label} {count}",
                    f"voip_agent_stage_duration_seconds_sum{label} {duration}",
                    f"voip_agent_stage_errors_total{label} {errors}",
                ]
            )
        return "\n".join(lines) + "\n"


class StatusServer:
    def __init__(self, status: AgentStatus, host: str, port: int):
        self.status, self.host, self.port = status, host, port
        self._server = None

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._handle, self.host, self.port, limit=8192)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_):
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader, writer):
        code, body, content_type = 400, "bad request\n", "text/plain"
        try:
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
            method, path, version = headers.split(b"\r\n", 1)[0].decode("ascii").split()
            if method != "GET":
                code, body = 405, "method not allowed\n"
            elif path == "/livez":
                code, body = (200 if self.status.live() else 503), "liveness\n"
            elif path == "/readyz":
                code, body = (200 if self.status.ready() else 503), "readiness\n"
            elif path == "/metrics":
                code, body, content_type = 200, self.status.metrics(), "text/plain; version=0.0.4"
            else:
                code, body = 404, "not found\n"
        except (
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            ValueError,
            UnicodeError,
        ):
            pass
        payload = body.encode()
        writer.write(
            (
                f"HTTP/1.1 {code} Status\r\nContent-Type: {content_type}\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
            ).encode()
            + payload
        )
        try:
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
