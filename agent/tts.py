import asyncio
import logging
from collections.abc import AsyncIterator

import httpx
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_VOICE_PROFILE = "shared-female-de-v1"
# faster-qwen3-tts wants full language names, not ISO codes. Omitting it sends
# language=None, which the server rejects (500). This is a German-only agent, so
# pin "german" rather than relying on per-utterance auto-detect (which can
# misfire on short outputs like "Ja", numbers, or names).
LANGUAGE = "german"

# A single 503 from /v1/audio/speech used to drop a whole response turn: the
# caller heard nothing and got no spoken error, while the next request seconds
# later succeeded. Retry the transient classes only — a 4xx, or the 500 the
# server returns for a rejected request body, repeats deterministically, and
# retrying it just adds dead air before the same failure.
RETRY_STATUS = frozenset({502, 503, 504})
# One entry per retry; total attempts = len + 1. Kept short so a retried turn
# still lands inside the caller's patience.
RETRY_BACKOFF_S: tuple[float, ...] = (0.3, 0.9)


class TtsClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        *,
        voice_profile: str = DEFAULT_VOICE_PROFILE,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._voice_profile = voice_profile

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def synthesize(self, text: str) -> bytes:
        for attempt, backoff in enumerate((*RETRY_BACKOFF_S, None)):
            try:
                resp = await self._client.post(
                    f"{self._base_url}/v1/audio/speech",
                    json={
                        "input": text,
                        "voice": self._voice_profile,
                        "language": LANGUAGE,
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                return resp.content
            except httpx.HTTPStatusError as exc:
                if backoff is None or exc.response.status_code not in RETRY_STATUS:
                    raise
                reason: object = exc.response.status_code
            except httpx.TransportError as exc:
                if backoff is None:
                    raise
                reason = exc
            # Warn, not debug: recovered 503s are the signal for root-causing
            # the server-side outage.
            log.warning(
                "TTS attempt %d failed (%s), retrying in %.1fs",
                attempt + 1,
                reason,
                backoff,
            )
            await asyncio.sleep(backoff)
        raise AssertionError("unreachable")  # pragma: no cover

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield 24kHz int16 PCM chunks as the server produces them.

        Buffers across reads so each yielded array is whole int16 samples
        (a chunk boundary can fall mid-sample on the wire).

        Wire contract is pinned to the *deployed* qwen3-tts-server, not the
        OpenAI standard. Verified against its live /openapi.json: streaming is a
        custom POST /v1/audio/speech/stream subpath returning raw s16le mono
        24kHz (application/octet-stream, chunked, no RIFF header, no
        x-audio-sample-rate header) — so we read raw int16 and the 24kHz rate is
        hardcoded downstream (stateful 24-to-16 kHz conversion). The server's SpeechRequest has
        no `stream`/`stream_format`/`response_format=pcm` fields; sending the
        OpenAI-canonical body (stream_format on the base endpoint, pcm format)
        would 422/break here. Migrate to stream_format once the server adopts it:
        https://github.com/AEON-7/qwen3-tts-server/issues/1
        """
        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/audio/speech/stream",
            json={"input": text, "voice": self._voice_profile, "language": LANGUAGE},
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
                raise ValueError("TTS stream ended with incomplete PCM16 sample")
