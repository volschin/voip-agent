import httpx
import numpy as np
import pytest
import respx

from agent.tts import VOICE_INSTRUCT, TtsClient


@pytest.fixture
def tts(settings):
    return TtsClient(base_url=settings.tts_base_url)


def _fake_pcm(n_samples: int = 24000) -> bytes:
    return (np.zeros(n_samples, dtype=np.int16)).tobytes()


@respx.mock
async def test_synthesize_returns_pcm(tts):
    fake_audio = _fake_pcm()
    respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=fake_audio)
    )
    result = await tts.synthesize("Hallo Welt")
    assert result == fake_audio


@respx.mock
async def test_synthesize_sends_text_and_voice(tts):
    route = respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=_fake_pcm())
    )
    await tts.synthesize("Test")
    body = route.calls[0].request.read()
    import json

    payload = json.loads(body)
    assert payload["input"] == "Test"
    # server reads `voice`, not `instruct`
    assert payload["voice"] == VOICE_INSTRUCT


@respx.mock
async def test_synthesize_raises_on_http_error(tts):
    respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(503, text="unavailable")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await tts.synthesize("Test")


async def test_aclose_closes_owned_client():
    client = TtsClient(base_url="http://tts:8002")
    await client.aclose()
    assert client._client.is_closed


async def test_aclose_leaves_injected_client_open():
    shared = httpx.AsyncClient()
    client = TtsClient(base_url="http://tts:8002", client=shared)
    await client.aclose()
    assert not shared.is_closed
    await shared.aclose()
