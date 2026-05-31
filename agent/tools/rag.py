import asyncpg
import httpx

TOP_K = 5


class RagTool:
    def __init__(
        self,
        pool: asyncpg.Pool,
        embedding_base_url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._pool = pool
        self._embed_url = embedding_base_url.rstrip("/") + "/v1/embeddings"
        # Shared long-lived client; one pool reused across embedding calls.
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def lookup(self, query: str) -> str:
        embedding = await self._embed(query)
        rows = await self._search(embedding)
        if not rows:
            return "Keine relevanten Informationen gefunden."
        return "\n\n".join(row["content"] for row in rows)

    async def _embed(self, text: str) -> list[float]:
        resp = await self._client.post(
            self._embed_url,
            json={"input": text},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def _search(self, embedding: list[float]) -> list[asyncpg.Record]:
        vec_literal = "[" + ",".join(str(x) for x in embedding) + "]"
        async with self._pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT content
                FROM documents
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_literal,
                TOP_K,
            )
