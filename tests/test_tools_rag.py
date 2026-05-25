from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.rag import RagTool


@pytest.fixture
def pool():
    mock_pool = AsyncMock()
    conn = AsyncMock()

    # Create a proper async context manager for pool.acquire()
    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(return_value=conn)
    async_cm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=async_cm)

    return mock_pool, conn


async def test_lookup_returns_joined_chunks(settings, pool):
    mock_pool, conn = pool
    embedding_response = MagicMock()
    embedding_response.json = MagicMock(return_value={"embedding": [0.1] * 1024})

    conn.fetch.return_value = [
        {"content": "Chunk A"},
        {"content": "Chunk B"},
    ]

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = AsyncMock(return_value=embedding_response)
        mock_client_cls.return_value = mock_http

        rag = RagTool(pool=mock_pool, embedding_base_url=settings.embedding_base_url)
        result = await rag.lookup("Was ist X?")

    assert "Chunk A" in result
    assert "Chunk B" in result


async def test_lookup_empty_result(settings, pool):
    mock_pool, conn = pool
    embedding_response = MagicMock()
    embedding_response.json = MagicMock(return_value={"embedding": [0.0] * 1024})
    conn.fetch.return_value = []

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = AsyncMock(return_value=embedding_response)
        mock_client_cls.return_value = mock_http

        rag = RagTool(pool=mock_pool, embedding_base_url=settings.embedding_base_url)
        result = await rag.lookup("unbekannt")

    assert result == "Keine relevanten Informationen gefunden."
