from agent.tools.calendar import UnavailableCalendar


async def test_unavailable_calendar_fails_safe_for_reads_and_writes():
    calendar = UnavailableCalendar()

    assert "nicht verfügbar" in await calendar.get_events("start", "end")
    assert "nicht verfügbar" in await calendar.create_event("title", "start", "end")
