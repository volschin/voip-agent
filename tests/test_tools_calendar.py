# tests/test_tools_calendar.py
import pytest
import respx
import httpx
from unittest.mock import MagicMock
from agent.tools.calendar import MSGraphCalendar


@pytest.fixture
def calendar(settings):
    msal_app = MagicMock()
    msal_app.acquire_token_for_client.return_value = {"access_token": "tok123"}
    return MSGraphCalendar(
        msal_app=msal_app,
        user_email=settings.calendar_user_email,
    )


@respx.mock
async def test_get_events_returns_formatted_string(calendar):
    respx.get(
        f"https://graph.microsoft.com/v1.0/users/{calendar._user_email}/calendarView"
    ).mock(return_value=httpx.Response(200, json={
        "value": [{
            "subject": "Team Meeting",
            "start": {"dateTime": "2026-05-25T10:00:00"},
            "end": {"dateTime": "2026-05-25T11:00:00"},
        }]
    }))
    result = await calendar.get_events("2026-05-25T00:00:00", "2026-05-25T23:59:59")
    assert "Team Meeting" in result
    assert "10:00" in result


@respx.mock
async def test_get_events_no_events(calendar):
    respx.get(
        f"https://graph.microsoft.com/v1.0/users/{calendar._user_email}/calendarView"
    ).mock(return_value=httpx.Response(200, json={"value": []}))
    result = await calendar.get_events("2026-05-25T00:00:00", "2026-05-25T23:59:59")
    assert result == "Keine Termine in diesem Zeitraum."


@respx.mock
async def test_create_event_returns_confirmation(calendar):
    respx.post(
        f"https://graph.microsoft.com/v1.0/users/{calendar._user_email}/events"
    ).mock(return_value=httpx.Response(201, json={"id": "evt1", "subject": "Arzt"}))
    result = await calendar.create_event(
        title="Arzt",
        start="2026-05-26T09:00:00",
        end="2026-05-26T09:30:00",
    )
    assert "Arzt" in result
