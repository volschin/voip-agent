import httpx

VOICE_INSTRUCT = "Eine warme, natürliche deutsche Stimme mit angemessenem Sprechtempo."


class TtsClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def synthesize(self, text: str) -> bytes:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/v1/audio/speech",
                json={"input": text, "instruct": VOICE_INSTRUCT},
                timeout=30.0,
            )
        resp.raise_for_status()
        return resp.content
