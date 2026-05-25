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

    def _token(self) -> str:
        result = self._msal.acquire_token_for_client(scopes=self._scope)
        return result["access_token"]

    async def get_events(self, start: str, end: str) -> str:
        headers = {"Authorization": f"Bearer {self._token()}"}
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
            dt = e["start"]["dateTime"][:16].replace("T", " ")
            lines.append(f"- {e['subject']} um {dt.split()[1]} Uhr")
        return "\n".join(lines)

    async def create_event(self, title: str, start: str, end: str, description: str = "") -> str:
        headers = {
            "Authorization": f"Bearer {self._token()}",
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
