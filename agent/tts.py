import httpx

VOICE_INSTRUCT = "Eine warme, natürliche deutsche Stimme mit angemessenem Sprechtempo."


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
            json={"input": text, "voice": VOICE_INSTRUCT},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.content
