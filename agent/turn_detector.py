import httpx
import numpy as np


class TurnDetectorClient:
    """Calls a DGX Smart Turn v2 service to classify end-of-turn.

    Mirrors SttClient's ownership model: reuse an injected long-lived client
    (and its pool); only close one we created ourselves.
    """

    def __init__(
        self,
        base_url: str,
        threshold: float = 0.5,
        timeout_ms: int = 150,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._threshold = threshold
        self._timeout_s = timeout_ms / 1000.0
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def classify(self, pcm_16k: np.ndarray) -> bool:
        """True = caller's turn is complete. Raises on timeout / HTTP error."""
        body = pcm_16k.astype(np.int16).tobytes()
        resp = await self._client.post(
            f"{self._base_url}/v1/turn/classify",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        return float(resp.json()["prob"]) >= self._threshold
