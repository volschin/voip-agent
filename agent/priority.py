"""Fail-closed client for the Companion Core voice-priority lease."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import httpx

_TOKEN_HEADER = "X-Voice-Priority-Token"


class PriorityUnavailable(RuntimeError):
    """Stable error without upstream body, URL, token, or lease details."""

    def __init__(self) -> None:
        super().__init__("voice priority service is unavailable")


@dataclass
class LeaseHandle:
    """One acquired lease with bounded renewal and idempotent release."""

    lease_id: UUID
    expires_in: int
    _client: "PriorityLeaseClient" = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    async def renew(self) -> None:
        if self._released:
            raise PriorityUnavailable()
        self.expires_in = await self._client._renew(self.lease_id)

    async def release(self) -> None:
        if self._released:
            return
        await self._client._release(self.lease_id)
        self._released = True


class PriorityLeaseClient:
    """Acquire a Core lease through the authenticated shared AI client."""

    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.AsyncClient,
        token_file: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._token = _read_token(token_file)

    async def acquire(self) -> LeaseHandle:
        payload = await self._request("POST", "/voice/priority/lease")
        lease_id, expires_in = _validate_lease(payload)
        return LeaseHandle(lease_id, expires_in, self)

    async def _renew(self, lease_id: UUID) -> int:
        payload = await self._request("PUT", f"/voice/priority/lease/{lease_id}")
        renewed_id, expires_in = _validate_lease(payload)
        if renewed_id != lease_id:
            raise PriorityUnavailable()
        return expires_in

    async def _release(self, lease_id: UUID) -> None:
        await self._request("DELETE", f"/voice/priority/lease/{lease_id}")

    async def _request(
        self,
        method: str,
        path: str,
    ) -> object:
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}{path}",
                headers={_TOKEN_HEADER: self._token},
                timeout=5.0,
            )
            response.raise_for_status()
            if method == "DELETE":
                return None
            return response.json()
        except (httpx.HTTPError, ValueError, TypeError):
            raise PriorityUnavailable() from None


def _read_token(path: str) -> str:
    try:
        token = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("voice priority token file is unavailable") from error
    if token.endswith("\r\n"):
        token = token[:-2]
    elif token.endswith("\n"):
        token = token[:-1]
    if not token.strip():
        raise ValueError("voice priority token must not be blank")
    return token


def _validate_lease(payload: object) -> tuple[UUID, int]:
    if not isinstance(payload, dict):
        raise PriorityUnavailable()
    try:
        lease_id = UUID(payload["lease_id"])
        expires_in = payload["expires_in"]
    except (KeyError, TypeError, ValueError):
        raise PriorityUnavailable() from None
    if isinstance(expires_in, bool) or not isinstance(expires_in, int):
        raise PriorityUnavailable()
    if not 1 <= expires_in <= 30:
        raise PriorityUnavailable()
    return lease_id, expires_in
