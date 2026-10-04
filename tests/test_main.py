from unittest.mock import AsyncMock

from agent import main


async def test_disabled_tools_make_no_backend_connections(settings, monkeypatch):
    settings = settings.model_copy(update={"trusted_callers": ""})
    pool = AsyncMock()
    monkeypatch.setattr(main.asyncpg, "create_pool", pool)
    result = await main.create_rag_pool(settings)
    assert result is None
    assert pool.call_count == 0
    calendar = main.create_calendar(settings)
    assert isinstance(calendar, main.UnavailableCalendar)
