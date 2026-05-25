import httpx
import numpy as np
import pytest
import respx

from agent.stt import SttClient


@pytest.fixture
def stt(settings):
    return SttClient(base_url=settings.stt_base_url)


def _pcm_16k(duration_ms: int = 500) -> bytes:
    n = 16000 * duration_ms // 1000
    return (np.zeros(n, dtype=np.int16)).tobytes()


@respx.mock
async def test_transcribe_returns_text(stt):
    respx.post("http://stt:8001/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "Hallo Welt"})
    )
    result = await stt.transcribe(_pcm_16k())
    assert result == "Hallo Welt"


@respx.mock
async def test_transcribe_sends_wav_with_language(stt):
    route = respx.post("http://stt:8001/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "test"})
    )
    await stt.transcribe(_pcm_16k())
    request = route.calls[0].request
    assert b"language" in request.content
    assert b"de" in request.content


@respx.mock
async def test_transcribe_raises_on_http_error(stt):
    respx.post("http://stt:8001/v1/audio/transcriptions").mock(
        return_value=httpx.Response(500, text="Server error")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await stt.transcribe(_pcm_16k())
