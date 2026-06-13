from collections.abc import AsyncIterator

import httpx
import numpy as np

VOICE_INSTRUCT = "Eine warme, natürliche deutsche Stimme mit angemessenem Sprechtempo."
# faster-qwen3-tts wants full language names, not ISO codes. Omitting it sends
# language=None, which the server rejects (500). This is a German-only agent, so
# pin "german" rather than relying on per-utterance auto-detect (which can
# misfire on short outputs like "Ja", numbers, or names).
LANGUAGE = "german"


class TtsClient:
    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def synthesize(self, text: str) -> bytes:
        resp = await self._client.post(
            f"{self._base_url}/v1/audio/speech",
            json={"input": text, "voice": VOICE_INSTRUCT, "language": LANGUAGE},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.content

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield 24kHz int16 PCM chunks as the server produces them.

        Buffers across reads so each yielded array is whole int16 samples
        (a chunk boundary can fall mid-sample on the wire).

        Wire contract is pinned to the *deployed* qwen3-tts-server, not the
        OpenAI standard. Verified against its live /openapi.json: streaming is a
        custom POST /v1/audio/speech/stream subpath returning raw s16le mono
        24kHz (application/octet-stream, chunked, no RIFF header, no
        x-audio-sample-rate header) — so we read raw int16 and the 24kHz rate is
        hardcoded downstream (resample_24k_to_8k). The server's SpeechRequest has
        no `stream`/`stream_format`/`response_format=pcm` fields; sending the
        OpenAI-canonical body (stream_format on the base endpoint, pcm format)
        would 422/break here. Migrate to stream_format once the server adopts it:
        https://github.com/AEON-7/qwen3-tts-server/issues/1
        """
        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/audio/speech/stream",
            json={"input": text, "voice": VOICE_INSTRUCT, "language": LANGUAGE},
            timeout=30.0,
        ) as resp:
            resp.raise_for_status()
            carry = b""
            async for raw in resp.aiter_bytes():
                buf = carry + raw
                n = len(buf) - (len(buf) % 2)  # whole int16 samples only
                if n:
                    yield np.frombuffer(buf[:n], dtype="<i2")
                carry = buf[n:]
            if carry:
                # Trailing odd byte should not happen; drop with no crash.
                pass
