from typing import Protocol
import httpx


GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class CalendarBackend(Protocol):
    async def get_events(self, start: str, end: str) -> str: ...
    async def create_event(self, title: str, start: str, end: str, description: str = "") -> str: ...


class MSGraphCalendar:
    def __init__(self, msal_app, user_email: str) -> None:
        self._msal = msal_app
        self._user_email = user_email
        self._scope = ["https://graph.microsoft.com/.default"]

    async def _token(self) -> str:
        import asyncio
        result = await asyncio.to_thread(
            self._msal.acquire_token_for_client, scopes=self._scope
        )
        if "access_token" not in result:
            raise RuntimeError(result.get("error_description", str(result)))
        return result["access_token"]

    async def get_events(self, start: str, end: str) -> str:
        token = await self._token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Prefer": 'outlook.timezone="Europe/Berlin"',
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GRAPH_BASE}/users/{self._user_email}/calendarView",
                params={"startDateTime": start, "endDateTime": end,
                        "$select": "subject,start,end"},
                headers=headers,
                timeout=15.0,
            )
        resp.raise_for_status()
        events = resp.json().get("value", [])
        if not events:
            return "Keine Termine in diesem Zeitraum."
        lines = []
        for e in events:
            start_raw = e["start"]["dateTime"][:16]  # "2026-05-25T10:00"
            date_part, time_part = start_raw.split("T")
            lines.append(f"- {e['subject']} am {date_part} um {time_part} Uhr")
        return "\n".join(lines)

    async def create_event(self, title: str, start: str, end: str, description: str = "") -> str:
        token = await self._token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = {
            "subject": title,
            "body": {"contentType": "text", "content": description},
            "start": {"dateTime": start, "timeZone": "Europe/Berlin"},
            "end": {"dateTime": end, "timeZone": "Europe/Berlin"},
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{GRAPH_BASE}/users/{self._user_email}/events",
                json=body,
                headers=headers,
                timeout=15.0,
            )
        resp.raise_for_status()
        subject = resp.json().get("subject", title)
        return f"Termin '{subject}' wurde erstellt."
