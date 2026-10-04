"""Origin- and route-limited clients for existing GX10 authentication contracts."""

import re
import ssl

import httpx

from agent.config import Settings
from agent.private_files import validate_private_file


def _tls_context(settings: Settings) -> ssl.SSLContext:
    context = ssl.create_default_context()
    if settings.ai_proxy_ca_file.strip():
        ca_file = validate_private_file(
            settings.ai_proxy_ca_file, label="AI proxy CA", forbid_group_other_read=False
        )
        try:
            context.load_verify_locations(cafile=str(ca_file))
        except (ssl.SSLError, OSError) as error:
            raise ValueError("AI proxy CA file is invalid") from error
    return context


def _guard(settings: Settings, *, bearer: bool):
    origin = httpx.URL(settings.ai_origin)

    async def guard(request: httpx.Request) -> None:
        if (request.url.scheme, request.url.host, request.url.port) != (
            origin.scheme,
            origin.host,
            origin.port,
        ):
            raise ValueError("AI request outside configured origin")
        method, path = request.method, request.url.path
        if bearer:
            allowed = (method == "GET" and path == "/v1/models") or (
                method == "POST" and path == "/v1/chat/completions"
            )
        else:
            allowed = (
                method == "POST"
                and path
                in {
                    "/v1/audio/transcriptions",
                    "/v1/audio/speech",
                    "/v1/audio/speech/stream",
                    "/voice/priority/lease",
                }
            ) or (
                method in {"PUT", "DELETE"}
                and re.fullmatch(r"/voice/priority/lease/[0-9a-f-]{36}", path) is not None
            )
        if not allowed or request.url.query or request.url.fragment:
            raise ValueError("AI request outside allowed route")

    return guard


def build_ai_client(settings: Settings) -> httpx.AsyncClient:
    username = settings.ai_proxy_username.strip()
    if not username:
        raise ValueError("AI proxy username must not be blank")
    for name in ("stt_base_url", "tts_base_url", "llm_base_url", "voice_priority_base_url"):
        if getattr(settings, name).rstrip("/") != settings.ai_origin:
            raise ValueError(f"{name} must use {settings.ai_origin}")
    return httpx.AsyncClient(
        auth=httpx.BasicAuth(
            username, _read_secret(settings.ai_proxy_password_file, "AI proxy password")
        ),
        verify=_tls_context(settings),
        timeout=httpx.Timeout(60.0, connect=5.0),
        follow_redirects=False,
        event_hooks={"request": [_guard(settings, bearer=False)]},
    )


def build_llm_client(settings: Settings) -> httpx.AsyncClient:
    if settings.llm_base_url.rstrip("/") != settings.ai_origin:
        raise ValueError("LLM base URL must use configured origin")
    key = _read_secret(settings.llm_api_key_file, "Liter API key")
    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}"},
        verify=_tls_context(settings),
        timeout=httpx.Timeout(60.0, connect=5.0),
        follow_redirects=False,
        event_hooks={"request": [_guard(settings, bearer=True)]},
    )


def _read_secret(path: str, label: str) -> str:
    file = validate_private_file(path, label=label, forbid_group_other_read=True)
    try:
        value = file.read_text(encoding="utf-8").removesuffix("\n").removesuffix("\r")
    except OSError as error:
        raise ValueError(f"{label} file is unavailable") from error
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value
