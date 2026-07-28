"""Authenticated HTTPS client for the shared Traefik AI boundary."""

import ssl
from pathlib import Path

import httpx

from agent.config import Settings

_AI_ORIGIN = "https://mate.olcon.de"


def build_ai_client(settings: Settings) -> httpx.AsyncClient:
    """Load protected credentials and build the only authenticated AI client."""
    username = settings.ai_proxy_username.strip()
    if not username:
        raise ValueError("AI proxy username must not be blank")

    for name, url in (
        ("STT", settings.stt_base_url),
        ("TTS", settings.tts_base_url),
        ("LLM", settings.llm_base_url),
        ("voice priority", settings.voice_priority_base_url),
    ):
        if url.rstrip("/") != _AI_ORIGIN:
            raise ValueError(f"{name} base URL must use {_AI_ORIGIN}")

    password = _read_password(settings.ai_proxy_password_file)
    ca_file = Path(settings.ai_proxy_ca_file)
    if not ca_file.is_file():
        raise ValueError("AI proxy CA file is unavailable")
    try:
        ssl_context = ssl.create_default_context(cafile=str(ca_file))
    except ssl.SSLError as error:
        raise ValueError("AI proxy CA file is invalid") from error
    except OSError as error:
        raise ValueError("AI proxy CA file is unavailable") from error

    return httpx.AsyncClient(
        auth=httpx.BasicAuth(username, password),
        verify=ssl_context,
        timeout=httpx.Timeout(60.0, connect=5.0),
    )


def _read_password(path: str) -> str:
    try:
        password = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("AI proxy password file is unavailable") from error
    if password.endswith("\r\n"):
        password = password[:-2]
    elif password.endswith("\n"):
        password = password[:-1]
    if not password.strip():
        raise ValueError("AI proxy password must not be blank")
    return password
