import asyncio
import json

import httpx
import numpy as np
import pytest
import respx

from agent import tts as agent_tts
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
        return_value=httpx.Response(500, text="bad language field")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await tts.synthesize("Test")


@pytest.fixture
def no_backoff(monkeypatch):
    """Keep the retry schedule (2 retries) but drop the real sleeps."""
    monkeypatch.setattr(agent_tts, "RETRY_BACKOFF_S", (0.0, 0.0))


@respx.mock
async def test_synthesize_retries_transient_503_and_recovers(tts, no_backoff):
    # Live 503s were brief: the next request 3 s later returned 200 and the
    # caller lost the whole turn to a single failure.
    fake_audio = _fake_pcm()
    route = respx.post("http://tts:8002/v1/audio/speech").mock(
        side_effect=[
            httpx.Response(503, text="unavailable"),
            httpx.Response(200, content=fake_audio),
        ]
    )

    result = await tts.synthesize("Test")

    assert result == fake_audio
    assert route.call_count == 2


@respx.mock
async def test_synthesize_retries_transport_error_and_recovers(tts, no_backoff):
    fake_audio = _fake_pcm()
    route = respx.post("http://tts:8002/v1/audio/speech").mock(
        side_effect=[
            httpx.ConnectError("connection reset"),
            httpx.Response(200, content=fake_audio),
        ]
    )

    result = await tts.synthesize("Test")

    assert result == fake_audio
    assert route.call_count == 2


@respx.mock
async def test_synthesize_gives_up_after_retry_budget(tts, no_backoff):
    route = respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(503, text="unavailable")
    )

    with pytest.raises(httpx.HTTPStatusError):
        await tts.synthesize("Test")

    assert route.call_count == 3


@respx.mock
async def test_synthesize_does_not_retry_deterministic_error(tts, no_backoff):
    # A 500 here means a rejected request body (e.g. the language field), not a
    # blip — retrying only adds silence before the same failure.
    route = respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(500, text="bad language field")
    )

    with pytest.raises(httpx.HTTPStatusError):
        await tts.synthesize("Test")

    assert route.call_count == 1


@respx.mock
async def test_synthesize_retry_stays_cancellable(tts, monkeypatch):
    # Barge-in cancels the synthesis task; the retry sleep must not swallow it.
    monkeypatch.setattr(agent_tts, "RETRY_BACKOFF_S", (30.0, 30.0))
    respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(503, text="unavailable")
    )

    task = asyncio.create_task(tts.synthesize("Test"))
    await asyncio.sleep(0)  # let it reach the backoff sleep
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


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
    # Verified server wire contract: raw little-endian int16 PCM at 24 kHz.
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
