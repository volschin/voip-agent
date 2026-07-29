import io
import wave
from dataclasses import dataclass

import httpx
import numpy as np
import pytest

from dgx.tts.api import HealthMetadata, create_app, encode_pcm_stream
from dgx.tts.profiles import ProfileError


@dataclass
class FakeRuntime:
    selected_voice: str | None = None
    selected_language: str | None = None
    fail: Exception | None = None

    def synthesize(self, text: str, voice: str | None, language: str | None):
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


def test_closing_pcm_stream_closes_model_iterator() -> None:
    closed = False

    def chunks():
        nonlocal closed
        try:
            yield np.array([1, 2], dtype=np.int16), 24_000, {}
            yield np.array([3, 4], dtype=np.int16), 24_000, {}
        finally:
            closed = True

    encoded = encode_pcm_stream(chunks())
    assert next(encoded) == b"\x01\x00\x02\x00"

    encoded.close()

    assert closed is True
