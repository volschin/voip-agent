import httpx
import asyncpg

TOP_K = 5


class RagTool:
    def __init__(self, pool: asyncpg.Pool, embedding_base_url: str) -> None:
        self._pool = pool
        self._embed_url = embedding_base_url.rstrip("/") + "/embed"

    async def lookup(self, query: str) -> str:
        embedding = await self._embed(query)
        rows = await self._search(embedding)
        if not rows:
            return "Keine relevanten Informationen gefunden."
        return "\n\n".join(row["content"] for row in rows)

    async def _embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._embed_url,
                json={"text": text},
                timeout=10.0,
            )
        resp.raise_for_status()
        return resp.json()["embedding"]

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
