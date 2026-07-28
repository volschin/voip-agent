"""Authenticated Traefik client isolated from local embedding traffic."""

import ssl
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from agent.ai_http import build_ai_client
from agent.config import Settings
from agent.llm import LlmClient
from agent.stt import SttClient
from agent.tools.rag import RagTool
from agent.tts import TtsClient

EXPECTED_AUTHORIZATION = "Basic dm9pcC1hZ2VudDptYWNoaW5lLXNlY3JldA=="


def _settings(tmp_path: Path, **overrides: str) -> Settings:
    password = tmp_path / "password"
    ca = tmp_path / "ca.crt"
    token = tmp_path / "priority-token"
    password.write_text("machine-secret\n", encoding="utf-8")
    ca.write_text("test CA fixture", encoding="utf-8")
    token.write_text("priority-secret\n", encoding="utf-8")
    values = {
        "fritzbox_sip_username": "agent-phone",
        "fritzbox_sip_password": "strong-secret",
        "stt_base_url": "https://mate.olcon.de",
        "tts_base_url": "https://mate.olcon.de",
        "llm_base_url": "https://mate.olcon.de",
        "ai_proxy_username": "voip-agent",
        "ai_proxy_password_file": str(password),
        "ai_proxy_ca_file": str(ca),
        "voice_priority_token_file": str(token),
        "voice_priority_base_url": "https://mate.olcon.de",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def ssl_context(monkeypatch: pytest.MonkeyPatch) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr(ssl, "create_default_context", lambda *, cafile: context)
    return context


def test_build_ai_client_loads_secret_files_and_ca_context(
    tmp_path: Path,
    ssl_context: ssl.SSLContext,
) -> None:
    client = build_ai_client(_settings(tmp_path))

    assert client._transport._pool._ssl_context is ssl_context
    assert "machine-secret" not in repr(client)


@pytest.mark.parametrize(
    ("override", "value", "error"),
    [
        ("ai_proxy_username", " ", "username"),
        ("ai_proxy_password_file", "/missing/password", "password file"),
        ("ai_proxy_ca_file", "/missing/ca", "CA file"),
    ],
)
def test_build_ai_client_rejects_missing_credentials(
    tmp_path: Path,
    ssl_context: ssl.SSLContext,
    override: str,
    value: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        build_ai_client(_settings(tmp_path, **{override: value}))


def test_build_ai_client_rejects_empty_password(
    tmp_path: Path,
    ssl_context: ssl.SSLContext,
) -> None:
    settings = _settings(tmp_path)
    Path(settings.ai_proxy_password_file).write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="password"):
        build_ai_client(settings)


@pytest.mark.asyncio
@respx.mock
async def test_ai_client_authenticates_stt_tts_and_llm_only(
    tmp_path: Path,
    ssl_context: ssl.SSLContext,
) -> None:
    ai_client = build_ai_client(_settings(tmp_path))
    local_client = httpx.AsyncClient()
    stt_route = respx.post("https://mate.olcon.de/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "Hallo"})
    )
    tts_route = respx.post("https://mate.olcon.de/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"\x00\x00")
    )
    llm_route = respx.post("https://mate.olcon.de/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hallo",
                            "tool_calls": None,
                        }
                    }
                ]
            },
        )
    )
    embedding_route = respx.post("http://dgx-spark:8003/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
    )
    pool = MagicMock()
    connection = AsyncMock()
    connection.fetch.return_value = []
    acquire = AsyncMock()
    acquire.__aenter__.return_value = connection
    acquire.__aexit__.return_value = False
    pool.acquire.return_value = acquire
    rag = RagTool(pool, "http://dgx-spark:8003", client=local_client)
    llm = LlmClient(
        base_url="https://mate.olcon.de",
        model="companion-gemma",
        system_prompt="Deutsch",
        rag=rag.lookup,
        calendar=AsyncMock(),
        client=ai_client,
    )

    assert await SttClient("https://mate.olcon.de", ai_client).transcribe(b"\x00\x00") == "Hallo"
    assert await TtsClient("https://mate.olcon.de", ai_client).synthesize("Hallo") == b"\x00\x00"
    assert await llm.complete([{"role": "user", "content": "Hallo"}]) == "Hallo"
    assert await rag.lookup("Wissen") == "Keine relevanten Informationen gefunden."

    for route in (stt_route, tts_route, llm_route):
        assert route.calls[0].request.headers["authorization"] == (EXPECTED_AUTHORIZATION)
    assert "authorization" not in embedding_route.calls[0].request.headers
    await ai_client.aclose()
    await local_client.aclose()
