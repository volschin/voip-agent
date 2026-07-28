"""VoIP client for the Companion Core priority lease."""

from pathlib import Path
from uuid import UUID

import httpx
import pytest
import respx

from agent.priority import PriorityLeaseClient, PriorityUnavailable

TOKEN_HEADER = "X-Voice-Priority-Token"
LEASE_ID = "12345678-1234-5678-1234-567812345678"


def _client(tmp_path: Path, http_client: httpx.AsyncClient) -> PriorityLeaseClient:
    token = tmp_path / "priority-token"
    token.write_text("priority-secret\n", encoding="utf-8")
    return PriorityLeaseClient(
        base_url="https://mate.olcon.de",
        client=http_client,
        token_file=str(token),
    )


@pytest.mark.asyncio
@respx.mock
async def test_priority_client_acquires_renews_and_releases_lease(
    tmp_path: Path,
) -> None:
    http_client = httpx.AsyncClient()
    acquire = respx.post("https://mate.olcon.de/voice/priority/lease").mock(
        return_value=httpx.Response(200, json={"lease_id": LEASE_ID, "expires_in": 30})
    )
    renew = respx.put(f"https://mate.olcon.de/voice/priority/lease/{LEASE_ID}").mock(
        return_value=httpx.Response(200, json={"lease_id": LEASE_ID, "expires_in": 30})
    )
    release = respx.delete(f"https://mate.olcon.de/voice/priority/lease/{LEASE_ID}").mock(
        return_value=httpx.Response(204)
    )
    client = _client(tmp_path, http_client)

    handle = await client.acquire()
    await handle.renew()
    await handle.release()
    await handle.release()

    assert handle.lease_id == UUID(LEASE_ID)
    assert handle.expires_in == 30
    assert acquire.calls[0].request.headers[TOKEN_HEADER] == "priority-secret"
    assert renew.calls[0].request.headers[TOKEN_HEADER] == "priority-secret"
    assert release.calls[0].request.headers[TOKEN_HEADER] == "priority-secret"
    assert release.call_count == 1
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403, text="priority-secret denied"),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"lease_id": "secret-lease", "expires_in": 30}),
        httpx.Response(200, json={"lease_id": LEASE_ID, "expires_in": 31}),
    ],
)
@respx.mock
async def test_priority_acquire_errors_are_stable_and_redacted(
    tmp_path: Path,
    response: httpx.Response,
) -> None:
    http_client = httpx.AsyncClient()
    respx.post("https://mate.olcon.de/voice/priority/lease").mock(return_value=response)
    client = _client(tmp_path, http_client)

    with pytest.raises(PriorityUnavailable) as error:
        await client.acquire()

    assert str(error.value) == "voice priority service is unavailable"
    assert "priority-secret" not in str(error.value)
    assert "secret-lease" not in str(error.value)
    await http_client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_priority_timeout_is_redacted(tmp_path: Path) -> None:
    http_client = httpx.AsyncClient()
    respx.post("https://mate.olcon.de/voice/priority/lease").mock(
        side_effect=httpx.ReadTimeout("priority-secret upstream")
    )
    client = _client(tmp_path, http_client)

    with pytest.raises(PriorityUnavailable, match="service is unavailable"):
        await client.acquire()

    await http_client.aclose()


@pytest.mark.parametrize("contents", ["", "\n"])
def test_priority_client_rejects_empty_token_file(
    tmp_path: Path,
    contents: str,
) -> None:
    token = tmp_path / "priority-token"
    token.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="priority token"):
        PriorityLeaseClient(
            base_url="https://mate.olcon.de",
            client=httpx.AsyncClient(),
            token_file=str(token),
        )


def test_priority_client_rejects_missing_token_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="priority token file is unavailable"):
        PriorityLeaseClient(
            base_url="https://mate.olcon.de",
            client=httpx.AsyncClient(),
            token_file=str(tmp_path / "missing"),
        )
