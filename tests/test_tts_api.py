import asyncio
import io
import json
import threading
import time
import wave
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from dgx.tts.api import (
    MAX_SPEECH_INPUT_CHARACTERS,
    HealthMetadata,
    SpeechRequest,
    _ClientDisconnected,
    _run_stable_synthesis,
    create_app,
    encode_pcm_stream,
)
from dgx.tts.clone_runtime import (
    CloneRuntime,
    SynthesisAdmissionTimeout,
    SynthesisCancelled,
)
from dgx.tts.profiles import ProfileError, VoiceProfile


@dataclass
class FakeRuntime:
    selected_voice: str | None = None
    selected_language: str | None = None
    fail: Exception | None = None

    def synthesize(
        self,
        text: str,
        voice: str | None,
        language: str | None,
        **_admission,
    ):
        if self.fail:
            raise self.fail
        self.selected_voice = voice
        self.selected_language = language
        return [np.array([0.0, 0.5, -0.5], dtype=np.float32)], 24_000

    def stream(self, text: str, voice: str | None, language: str | None):
        if self.fail:
            raise self.fail
        self.selected_voice = voice
        self.selected_language = language
        yield np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32), 24_000, {}


def _health() -> HealthMetadata:
    return HealthMetadata(
        model_revision="fd4b254389122332181a7c3db7f27e918eec64e3",
        default_profile="shared-female-de-v1",
        profiles_loaded=("shared-female-de-v1",),
        device="NVIDIA GB10",
    )


async def _client(runtime: FakeRuntime) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(runtime, _health())),
        base_url="http://test",
    )


async def test_health_reports_warm_base_profile_contract() -> None:
    async with await _client(FakeRuntime()) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_revision": "fd4b254389122332181a7c3db7f27e918eec64e3",
        "default_profile": "shared-female-de-v1",
        "profiles_loaded": ["shared-female-de-v1"],
        "device": "NVIDIA GB10",
    }


async def test_non_streaming_endpoint_returns_valid_24khz_wav() -> None:
    runtime = FakeRuntime()
    async with await _client(runtime) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "Hallo",
                "voice": "shared-female-de-v1",
                "language": "german",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    with wave.open(io.BytesIO(response.content), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (
            1,
            2,
            24_000,
        )
        assert audio.getnframes() == 3
    assert runtime.selected_voice == "shared-female-de-v1"
    assert runtime.selected_language == "german"


@pytest.mark.parametrize(
    "audio",
    [
        np.array([[0.0, 0.25], [-0.25, 0.0]], dtype=np.float32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([0.0, np.inf], dtype=np.float32),
        np.array([0.0, 1.01], dtype=np.float32),
        np.array([-32_769, 0], dtype=np.int32),
        np.array([0, 32_768], dtype=np.int32),
    ],
    ids=["stereo", "nan", "infinite", "float-range", "int16-low", "int16-high"],
)
async def test_non_streaming_endpoint_rejects_unsafe_model_audio(audio: np.ndarray) -> None:
    class InvalidAudioRuntime(FakeRuntime):
        def synthesize(self, *_args, **_kwargs):
            return [audio], 24_000

    async with await _client(InvalidAudioRuntime()) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "Hallo",
                "voice": "shared-female-de-v1",
                "language": "german",
            },
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "synthesis failed"}


async def test_streaming_endpoint_scales_and_clips_float_pcm() -> None:
    runtime = FakeRuntime()
    async with await _client(runtime) as client:
        response = await client.post(
            "/v1/audio/speech/stream",
            json={"input": "Hallo", "voice": "shared-female-de-v1", "language": "de"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert np.frombuffer(response.content, dtype="<i2").tolist() == [
        -32767,
        -16383,
        16383,
        32767,
    ]


@pytest.mark.parametrize("voice", ["", "unknown-profile"])
async def test_unknown_profile_returns_bounded_422(voice: str) -> None:
    runtime = FakeRuntime(fail=ProfileError("unsupported voice profile"))
    async with await _client(runtime) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={"input": "Hallo", "voice": voice, "language": "german"},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "unsupported voice profile"}


async def test_model_failure_returns_bounded_500() -> None:
    runtime = FakeRuntime(fail=RuntimeError("secret model path"))
    async with await _client(runtime) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "Hallo",
                "voice": "shared-female-de-v1",
                "language": "german",
            },
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "synthesis failed"}
    assert "secret model path" not in response.text


async def test_non_streaming_endpoint_rejects_oversized_input_before_generation() -> None:
    runtime = FakeRuntime()
    async with await _client(runtime) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "x" * (MAX_SPEECH_INPUT_CHARACTERS + 1),
                "voice": "shared-female-de-v1",
                "language": "german",
            },
        )

    assert response.status_code == 422
    assert runtime.selected_voice is None


def _runtime_profile() -> VoiceProfile:
    return VoiceProfile(
        profile_id="shared-female-de-v1",
        audio_path=Path("/private/reference.wav"),
        reference_text="Guten Tag.",
        language="german",
        source_type="licensed-human-reference-private",
        source_model=None,
        source_revision="private-v1",
        source_sha256="1" * 64,
        sha256="2" * 64,
        selected_at="2026-07-29T12:00:00Z",
        evaluation_score=91.25,
        selected_candidate_id="candidate-a",
        design_instruction=None,
    )


class _StableRuntime:
    def __init__(self, failure: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.failure = failure

    def synthesize(
        self,
        text: str,
        _voice: str | None,
        _language: str | None,
        **_admission: Any,
    ) -> tuple[list[np.ndarray], int]:
        self.calls.append(text)
        if self.failure is not None:
            raise self.failure
        return [np.array([0.0, 0.25], dtype=np.float32)], 24_000


class _DisconnectRequest:
    def __init__(self, disconnected: bool = False) -> None:
        self.disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _block_executor_worker(started: threading.Event, release: threading.Event) -> None:
    started.set()
    release.wait(timeout=2)


@asynccontextmanager
async def _saturated_default_executor() -> AsyncIterator[None]:
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(executor)
    release = threading.Event()
    started = [threading.Event() for _ in range(2)]
    blockers = [
        loop.run_in_executor(None, _block_executor_worker, worker_started, release)
        for worker_started in started
    ]
    safety_release = threading.Timer(0.5, release.set)
    safety_release.daemon = True
    safety_release.start()
    try:
        async with asyncio.timeout(0.2):
            while not all(worker_started.is_set() for worker_started in started):
                await asyncio.sleep(0.001)
        yield
    finally:
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*blockers)
        safety_release.cancel()


def _background_tasks() -> set[asyncio.Task]:
    current = asyncio.current_task()
    return {task for task in asyncio.all_tasks() if task is not current}


def _stable_helper_tasks() -> set[asyncio.Task]:
    return {
        task
        for task in _background_tasks()
        if any(
            helper in getattr(task.get_coro(), "__qualname__", "")
            for helper in ("_run_stable_synthesis", "_wait_for_disconnect")
        )
    }


async def test_stable_timeout_includes_saturated_executor_queue_time() -> None:
    runtime = _StableRuntime()
    request = _DisconnectRequest()

    async with _saturated_default_executor():
        operation = _run_stable_synthesis(
            runtime,
            SpeechRequest(input="queued"),
            request,
            0.05,
        )
        with pytest.raises(SynthesisAdmissionTimeout):
            await asyncio.wait_for(operation, timeout=0.20)
        assert runtime.calls == []

    await asyncio.sleep(0)
    assert runtime.calls == []


async def test_stable_disconnect_cancels_saturated_executor_work() -> None:
    runtime = _StableRuntime()
    request = _DisconnectRequest(disconnected=True)
    tasks_before = _background_tasks()

    async with _saturated_default_executor():
        operation = _run_stable_synthesis(
            runtime,
            SpeechRequest(input="queued"),
            request,
            1.0,
        )
        with pytest.raises(_ClientDisconnected):
            await asyncio.wait_for(operation, timeout=0.20)
        assert runtime.calls == []

    await asyncio.sleep(0)
    assert runtime.calls == []
    assert _background_tasks() == tasks_before


async def test_stable_monitor_finalization_covers_every_exit() -> None:
    speech = SpeechRequest(input="Hallo")

    tasks_before = _stable_helper_tasks()
    await _run_stable_synthesis(_StableRuntime(), speech, _DisconnectRequest(), 1.0)
    assert _stable_helper_tasks() == tasks_before

    tasks_before = _stable_helper_tasks()
    with pytest.raises(SynthesisAdmissionTimeout):
        await _run_stable_synthesis(
            _StableRuntime(SynthesisAdmissionTimeout()),
            speech,
            _DisconnectRequest(),
            1.0,
        )
    assert _stable_helper_tasks() == tasks_before

    class CancelAwareRuntime(_StableRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()

        def synthesize(self, text, voice, language, *, cancel_event, **admission):
            del voice, language, admission
            self.calls.append(text)
            self.started.set()
            cancel_event.wait(timeout=1)
            raise SynthesisCancelled()

    disconnected_runtime = CancelAwareRuntime()
    tasks_before = _stable_helper_tasks()
    with pytest.raises(_ClientDisconnected):
        await _run_stable_synthesis(
            disconnected_runtime,
            speech,
            _DisconnectRequest(disconnected=True),
            1.0,
        )
    assert _stable_helper_tasks() == tasks_before

    cancelled_runtime = CancelAwareRuntime()
    tasks_before = _stable_helper_tasks()
    operation = asyncio.create_task(
        _run_stable_synthesis(
            cancelled_runtime,
            speech,
            _DisconnectRequest(),
            1.0,
        )
    )
    async with asyncio.timeout(0.2):
        while not cancelled_runtime.started.is_set():
            await asyncio.sleep(0.001)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert _stable_helper_tasks() == tasks_before


async def test_cancelled_queued_stable_route_never_starts_model_generation() -> None:
    profile = _runtime_profile()
    active_started = threading.Event()
    release_active = threading.Event()
    queued_entered = threading.Event()

    class BlockingModel:
        def __init__(self) -> None:
            self.generated_texts: list[str] = []

        def generate_voice_clone(self, **kwargs):
            self.generated_texts.append(kwargs["text"])
            if kwargs["text"] == "active":
                active_started.set()
                release_active.wait(timeout=2)
            return [np.array([0.0, 0.25], dtype=np.float32)], 24_000

    class TrackingRuntime(CloneRuntime):
        def synthesize(self, text, voice, language, **kwargs):
            if text == "queued":
                queued_entered.set()
            return super().synthesize(text, voice, language, **kwargs)

    model = BlockingModel()
    runtime = TrackingRuntime(model, {profile.profile_id: profile}, profile.profile_id)
    app = create_app(runtime, _health())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        active = asyncio.create_task(
            client.post(
                "/v1/audio/speech",
                json={"input": "active", "voice": profile.profile_id, "language": "german"},
            )
        )
        assert await asyncio.to_thread(active_started.wait, 1)
        queued = asyncio.create_task(
            client.post(
                "/v1/audio/speech",
                json={"input": "queued", "voice": profile.profile_id, "language": "german"},
            )
        )
        assert await asyncio.to_thread(queued_entered.wait, 1)

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release_active.set()
        assert (await active).status_code == 200
        following = await client.post(
            "/v1/audio/speech",
            json={"input": "following", "voice": profile.profile_id, "language": "german"},
        )

    assert following.status_code == 200
    assert model.generated_texts == ["active", "following"]


async def test_stable_route_admission_timeout_is_bounded_and_recovers() -> None:
    profile = _runtime_profile()
    active_started = threading.Event()
    release_active = threading.Event()

    class BlockingModel:
        def __init__(self) -> None:
            self.generated_texts: list[str] = []

        def generate_voice_clone(self, **kwargs):
            self.generated_texts.append(kwargs["text"])
            if kwargs["text"] == "active":
                active_started.set()
                release_active.wait(timeout=2)
            return [np.array([0.0, 0.25], dtype=np.float32)], 24_000

    model = BlockingModel()
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)
    app = create_app(runtime, _health(), synthesis_admission_timeout_seconds=0.05)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        active = asyncio.create_task(
            asyncio.to_thread(
                runtime.synthesize,
                "active",
                profile.profile_id,
                "german",
                cancel_event=threading.Event(),
                lock_timeout=1.0,
            )
        )
        assert await asyncio.to_thread(active_started.wait, 1)
        timed_out = await client.post(
            "/v1/audio/speech",
            json={"input": "timed-out", "voice": profile.profile_id, "language": "german"},
        )
        release_active.set()
        await active
        following = await client.post(
            "/v1/audio/speech",
            json={"input": "following", "voice": profile.profile_id, "language": "german"},
        )

    assert timed_out.status_code == 503
    assert timed_out.json() == {"detail": "synthesis admission timed out"}
    assert following.status_code == 200
    assert model.generated_texts == ["active", "following"]


async def test_closing_pcm_stream_closes_model_iterator() -> None:
    closed = False

    def chunks():
        nonlocal closed
        try:
            yield np.array([1, 2], dtype=np.int16), 24_000, {}
            yield np.array([3, 4], dtype=np.int16), 24_000, {}
        finally:
            closed = True

    encoded = encode_pcm_stream(chunks())
    assert await anext(encoded) == b"\x01\x00\x02\x00"

    await encoded.aclose()

    assert closed is True


async def test_asgi_disconnect_closes_model_iterator() -> None:
    closed = threading.Event()

    class DisconnectRuntime(FakeRuntime):
        def stream(self, text: str, voice: str | None, language: str | None):
            del text, voice, language
            try:
                while True:
                    time.sleep(0.05)
                    yield np.array([1, 2], dtype=np.int16), 24_000, {}
            finally:
                closed.set()

    app = create_app(DisconnectRuntime(), _health())
    disconnected = asyncio.Event()
    request_sent = False
    body = json.dumps(
        {
            "input": "Hallo",
            "voice": "shared-female-de-v1",
            "language": "german",
        }
    ).encode()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            disconnected.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/speech/stream",
        "raw_path": b"/v1/audio/speech/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }

    await asyncio.wait_for(app(scope, receive, send), timeout=2)

    assert closed.wait(timeout=1)


async def test_cancelling_inflight_next_waits_before_closing_iterator() -> None:
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def chunks():
        try:
            started.set()
            release.wait(timeout=2)
            yield np.array([1, 2], dtype=np.int16), 24_000, {}
        finally:
            closed.set()

    encoded = encode_pcm_stream(chunks())
    request = asyncio.create_task(anext(encoded))
    assert await asyncio.to_thread(started.wait, 1)

    request.cancel()
    await asyncio.sleep(0.05)
    assert request.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert closed.wait(timeout=1)
