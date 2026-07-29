import json

import httpx
import numpy as np
import pytest
import respx

from agent.tts import TtsClient

VOICE_PROFILE = "shared-female-de-v1"


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
    payload = json.loads(route.calls[0].request.read())
    assert payload["input"] == "Test"
    assert payload["voice"] == VOICE_PROFILE
    # full language name required — ISO "de" / omitting it both 500 server-side
    assert payload["language"] == "german"


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


@respx.mock
async def test_synthesize_uses_configured_voice_profile():
    route = respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=_fake_pcm())
    )
    client = TtsClient(
        base_url="http://tts:8002",
        voice_profile="private-profile-v2",
    )

    await client.synthesize("Test")

    assert json.loads(route.calls[0].request.read())["voice"] == "private-profile-v2"
    await client.aclose()


async def test_synthesize_stream_yields_pcm_chunks():
    # ASSUMPTION: server streams raw little-endian int16 PCM at 24kHz.
    # Replace this fixture with a captured /v1/audio/speech/stream response
    # once verified on the DGX box (see Task 1 Step 4).
    chunk_a = (np.arange(240, dtype="<i2")).tobytes()
    chunk_b = (np.arange(240, 480, dtype="<i2")).tobytes()

    requests: list[httpx.Request] = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(chunk_a + chunk_b))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    tts = TtsClient(base_url="http://tts:8002", client=client)

    chunks = [c async for c in tts.synthesize_stream("Hallo")]
    joined = np.concatenate(chunks)

    assert joined.dtype == np.int16
    assert joined.size == 480
    assert json.loads(requests[0].content) == {
        "input": "Hallo",
        "voice": VOICE_PROFILE,
        "language": "german",
    }
    await client.aclose()


async def test_synthesize_stream_rejects_trailing_odd_pcm_byte():
    async def handler(_request):
        return httpx.Response(200, stream=httpx.ByteStream(b"\x01\x00\x02"))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    tts = TtsClient(base_url="http://tts:8002", client=client)

    with pytest.raises(ValueError, match="incomplete PCM16 sample"):
        _ = [chunk async for chunk in tts.synthesize_stream("Hallo")]

    await client.aclose()
