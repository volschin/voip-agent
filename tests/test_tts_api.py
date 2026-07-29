import asyncio
import gc
import inspect
import io
import json
import threading
import time
import wave
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
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
    _cancel_stable_operation,
    _ClientDisconnected,
    _run_stable_synthesis,
    create_app,
    encode_pcm_stream,
)
from dgx.tts.clone_runtime import (
    CloneRuntime,
    SynthesisAdmissionTimeout,
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
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(
        self,
        text: str,
        _voice: str | None,
        _language: str | None,
        **_admission: Any,
    ) -> tuple[list[np.ndarray], int]:
        self.calls.append(text)
        return [np.array([0.0, 0.25], dtype=np.float32)], 24_000


class _NonCooperativeRuntime(_StableRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.cancel_event: threading.Event | None = None

    def synthesize(
        self,
        text: str,
        _voice: str | None,
        _language: str | None,
        *,
        cancel_event: threading.Event,
        **_admission: Any,
    ) -> tuple[list[np.ndarray], int]:
        self.calls.append(text)
        self.cancel_event = cancel_event
        self.started.set()
        try:
            self.release.wait(timeout=2)
            raise RuntimeError("late non-cooperative generation failure")
        finally:
            self.finished.set()


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
    await asyncio.to_thread(lambda: None)
    previous_executor = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    release = threading.Event()
    started = [threading.Event()]
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
        release.set()
        sentinel = loop.run_in_executor(None, lambda: None)
        try:
            await asyncio.gather(*blockers)
            await sentinel
        finally:
            loop.set_default_executor(previous_executor)
            executor.shutdown(wait=True)
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
            for helper in ("_run_stable_synthesis", "_wait_for_disconnect", "to_thread")
        )
    }


async def _wait_for_thread_event(event: threading.Event) -> None:
    async with asyncio.timeout(0.2):
        while not event.is_set():
            await asyncio.sleep(0.001)


async def test_cancel_stable_operation_settles_wrapper_without_waiting_for_worker() -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()

    def blocking_worker() -> None:
        worker_started.set()
        try:
            release_worker.wait(timeout=2)
        finally:
            worker_finished.set()

    operation = asyncio.create_task(asyncio.to_thread(blocking_worker))
    try:
        await _wait_for_thread_event(worker_started)
        cancelled = threading.Event()

        settlement = _cancel_stable_operation(operation, cancelled)

        assert inspect.isawaitable(settlement)
        await asyncio.wait_for(settlement, timeout=0.05)
        assert cancelled.is_set()
        assert operation.cancelled()
        assert worker_finished.is_set() is False
    finally:
        release_worker.set()
        await _wait_for_thread_event(worker_finished)


async def test_cancel_stable_operation_consumes_completed_runtime_error() -> None:
    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

    async def fail_immediately() -> None:
        raise RuntimeError("late synthesis failure")

    operation = asyncio.create_task(fail_immediately())
    try:
        await asyncio.sleep(0)
        assert operation.done()
        cancelled = threading.Event()

        await _cancel_stable_operation(operation, cancelled)

        assert cancelled.is_set()
        del operation
        gc.collect()
        await asyncio.sleep(0)
        assert loop_errors == []
    finally:
        loop.set_exception_handler(previous_exception_handler)


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


async def test_stable_absolute_deadline_prevents_late_executor_model_start() -> None:
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(lambda: None)
    previous_executor = loop._default_executor

    class TrackingExecutor(ThreadPoolExecutor):
        def __init__(self) -> None:
            super().__init__(max_workers=1)
            self._submission_count = 0
            self._submission_lock = threading.Lock()
            self.runtime_queued = threading.Event()

        def submit(self, fn, /, *args, **kwargs):
            with self._submission_lock:
                self._submission_count += 1
                submission = self._submission_count
            future = super().submit(fn, *args, **kwargs)
            if submission == 2:
                self.runtime_queued.set()
            return future

    executor = TrackingExecutor()
    loop.set_default_executor(executor)
    blocker_started = threading.Event()
    release_worker = threading.Event()

    def blocker() -> None:
        blocker_started.set()
        release_worker.wait(timeout=2)

    blocker_future = loop.run_in_executor(None, blocker)
    timer: threading.Timer | None = None
    operation: asyncio.Task | None = None
    try:
        await _wait_for_thread_event(blocker_started)
        profile = _runtime_profile()

        class CountingModel:
            def __init__(self) -> None:
                self.generated_texts: list[str] = []

            def generate_voice_clone(self, **kwargs):
                self.generated_texts.append(kwargs["text"])
                return [np.array([0.0], dtype=np.float32)], 24_000

        model = CountingModel()

        class DeadlineTrackingRuntime(CloneRuntime):
            def __init__(self) -> None:
                super().__init__(model, {profile.profile_id: profile}, profile.profile_id)
                self.entered = threading.Event()
                self.finished = threading.Event()
                self.entered_after_deadline: bool | None = None

            def synthesize(self, *args, **kwargs):
                deadline = kwargs["admission_deadline"]
                self.entered_after_deadline = time.monotonic() >= deadline
                self.entered.set()
                try:
                    return super().synthesize(*args, **kwargs)
                finally:
                    self.finished.set()

        runtime = DeadlineTrackingRuntime()
        operation = asyncio.create_task(
            _run_stable_synthesis(
                runtime,
                SpeechRequest(input="late", voice=profile.profile_id),
                _DisconnectRequest(),
                0.10,
            )
        )
        await _wait_for_thread_event(executor.runtime_queued)
        timer = threading.Timer(0.12, release_worker.set)
        timer.start()

        assert runtime.entered.wait(timeout=1)
        assert runtime.finished.wait(timeout=1)
        assert runtime.entered_after_deadline is True

        with pytest.raises(SynthesisAdmissionTimeout):
            await operation
        assert model.generated_texts == []
    finally:
        release_worker.set()
        await blocker_future
        if timer is not None:
            timer.cancel()
        if operation is not None:
            if not operation.done():
                operation.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await operation
        loop.set_default_executor(previous_executor)
        executor.shutdown(wait=True)


async def test_stable_absolute_deadline_rejects_late_success() -> None:
    class LateSuccessRuntime(_StableRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()
            self.cancel_event: threading.Event | None = None
            self.started_before_deadline: bool | None = None
            self.completed_after_deadline: bool | None = None

        def synthesize(
            self,
            text: str,
            _voice: str | None,
            _language: str | None,
            *,
            cancel_event: threading.Event,
            **_admission: Any,
        ) -> tuple[list[np.ndarray], int]:
            deadline = _admission["admission_deadline"]
            self.calls.append(text)
            self.cancel_event = cancel_event
            self.started_before_deadline = time.monotonic() < deadline
            self.started.set()
            try:
                self.release.wait(timeout=2)
                self.completed_after_deadline = time.monotonic() >= deadline
                return [np.array([0.0], dtype=np.float32)], 24_000
            finally:
                self.finished.set()

    runtime = LateSuccessRuntime()
    operation = asyncio.create_task(
        _run_stable_synthesis(
            runtime,
            SpeechRequest(input="started-in-time"),
            _DisconnectRequest(),
            0.10,
        )
    )
    timer: threading.Timer | None = None
    try:
        await _wait_for_thread_event(runtime.started)
        assert runtime.started_before_deadline is True
        timer = threading.Timer(0.12, runtime.release.set)
        timer.start()

        assert runtime.finished.wait(timeout=1)
        assert runtime.completed_after_deadline is True

        with pytest.raises(SynthesisAdmissionTimeout):
            await operation
        assert runtime.calls == ["started-in-time"]
        assert runtime.cancel_event is not None
        assert runtime.cancel_event.is_set()
    finally:
        runtime.release.set()
        if timer is not None:
            timer.cancel()
        if not operation.done():
            operation.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await operation
        await _wait_for_thread_event(runtime.finished)


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
    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, Any]] = []
    running: list[_NonCooperativeRuntime] = []
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    try:
        tasks_before = _stable_helper_tasks()
        await _run_stable_synthesis(_StableRuntime(), speech, _DisconnectRequest(), 1.0)
        assert _stable_helper_tasks() == tasks_before

        timeout_runtime = _NonCooperativeRuntime()
        running.append(timeout_runtime)
        tasks_before = _stable_helper_tasks()
        timeout_operation = asyncio.create_task(
            _run_stable_synthesis(
                timeout_runtime,
                speech,
                _DisconnectRequest(),
                0.05,
            )
        )
        await _wait_for_thread_event(timeout_runtime.started)
        with pytest.raises(SynthesisAdmissionTimeout):
            await asyncio.wait_for(timeout_operation, timeout=0.20)
        assert timeout_runtime.finished.is_set() is False
        assert timeout_runtime.cancel_event is not None
        assert timeout_runtime.cancel_event.is_set()
        assert _stable_helper_tasks() == tasks_before
        timeout_runtime.release.set()
        await _wait_for_thread_event(timeout_runtime.finished)

        disconnected_runtime = _NonCooperativeRuntime()
        running.append(disconnected_runtime)
        disconnected_request = _DisconnectRequest()
        tasks_before = _stable_helper_tasks()
        disconnect_operation = asyncio.create_task(
            _run_stable_synthesis(
                disconnected_runtime,
                speech,
                disconnected_request,
                1.0,
            )
        )
        await _wait_for_thread_event(disconnected_runtime.started)
        disconnected_request.disconnected = True
        with pytest.raises(_ClientDisconnected):
            await asyncio.wait_for(disconnect_operation, timeout=0.20)
        assert disconnected_runtime.finished.is_set() is False
        assert disconnected_runtime.cancel_event is not None
        assert disconnected_runtime.cancel_event.is_set()
        assert _stable_helper_tasks() == tasks_before
        disconnected_runtime.release.set()
        await _wait_for_thread_event(disconnected_runtime.finished)

        cancelled_runtime = _NonCooperativeRuntime()
        running.append(cancelled_runtime)
        tasks_before = _stable_helper_tasks()
        cancel_operation = asyncio.create_task(
            _run_stable_synthesis(
                cancelled_runtime,
                speech,
                _DisconnectRequest(),
                1.0,
            )
        )
        await _wait_for_thread_event(cancelled_runtime.started)
        cancel_operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancel_operation, timeout=0.20)
        assert cancelled_runtime.finished.is_set() is False
        assert cancelled_runtime.cancel_event is not None
        assert cancelled_runtime.cancel_event.is_set()
        assert _stable_helper_tasks() == tasks_before
        cancelled_runtime.release.set()
        await _wait_for_thread_event(cancelled_runtime.finished)

        gc.collect()
        await asyncio.sleep(0)
        assert loop_errors == []
    finally:
        for runtime in running:
            runtime.release.set()
        for runtime in running:
            if runtime.started.is_set() and not runtime.finished.is_set():
                await _wait_for_thread_event(runtime.finished)
        loop.set_exception_handler(previous_exception_handler)


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
