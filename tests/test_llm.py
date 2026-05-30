import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from agent.llm import _FALLBACK_MSG, _NEEDS_CONFIRM_MSG, _WRITE_DISABLED_MSG, LlmClient


def _make_llm(settings, **over):
    rag = AsyncMock(return_value="RAG result")
    calendar = AsyncMock()
    calendar.get_events = AsyncMock(return_value="No events")
    calendar.create_event = AsyncMock(return_value="Event created")
    kwargs = dict(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        system_prompt=settings.llm_system_prompt,
        rag=rag,
        calendar=calendar,
    )
    kwargs.update(over)
    return LlmClient(**kwargs)


@pytest.fixture
def llm(settings):
    return _make_llm(settings)


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": None}}]}


def _tool_call_response(name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }


@respx.mock
async def test_complete_no_tool(llm):
    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("Hallo!"))
    )
    result = await llm.complete([{"role": "user", "content": "Hi"}])
    assert result == "Hallo!"


@respx.mock
async def test_complete_with_rag_tool_call(llm):
    responses = [
        httpx.Response(200, json=_tool_call_response("rag_lookup", {"query": "Was ist X?"})),
        httpx.Response(200, json=_chat_response("X ist Y.")),
    ]
    respx.post("http://llm:8000/v1/chat/completions").mock(side_effect=responses)
    result = await llm.complete([{"role": "user", "content": "Was ist X?"}])
    assert result == "X ist Y."
    llm._rag.assert_awaited_once_with("Was ist X?")


@respx.mock
async def test_complete_raises_on_http_error(llm):
    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="error")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await llm.complete([{"role": "user", "content": "test"}])


@respx.mock
async def test_calendar_write_disabled_by_default(settings):
    llm = _make_llm(settings)  # calendar_write_enabled defaults False
    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_tool_call_response(
                    "calendar_create_event",
                    {"title": "X", "start": "s", "end": "e", "confirmed": True},
                ),
            ),
            httpx.Response(200, json=_chat_response("ok")),
        ]
    )
    result = await llm.complete([{"role": "user", "content": "Termin anlegen"}])
    assert result == "ok"
    llm._calendar.create_event.assert_not_awaited()


@respx.mock
async def test_calendar_write_enabled_needs_confirmation(settings):
    llm = _make_llm(settings, calendar_write_enabled=True)
    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_tool_call_response(
                    "calendar_create_event",
                    {"title": "X", "start": "s", "end": "e"},  # confirmed omitted
                ),
            ),
            httpx.Response(200, json=_chat_response("done")),
        ]
    )
    result = await llm.complete([{"role": "user", "content": "Termin"}])
    assert result == "done"
    llm._calendar.create_event.assert_not_awaited()


@respx.mock
async def test_calendar_write_enabled_and_confirmed_creates(settings):
    llm = _make_llm(settings, calendar_write_enabled=True)
    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_tool_call_response(
                    "calendar_create_event",
                    {"title": "X", "start": "s", "end": "e", "confirmed": True},
                ),
            ),
            httpx.Response(200, json=_chat_response("angelegt")),
        ]
    )
    result = await llm.complete([{"role": "user", "content": "Termin"}])
    assert result == "angelegt"
    llm._calendar.create_event.assert_awaited_once()


@respx.mock
async def test_tool_loop_is_bounded(settings):
    llm = _make_llm(settings, max_tool_rounds=2)
    # Model keeps emitting tool calls forever; the cap must stop it.
    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json=_tool_call_response("rag_lookup", {"query": "loop"})
        )
    )
    result = await llm.complete([{"role": "user", "content": "x"}])
    assert result == _FALLBACK_MSG
    assert llm._rag.await_count == 2  # rounds 0 and 1; round 2 sends no tools


@respx.mock
async def test_malformed_tool_arguments_do_not_crash(llm):
    bad = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "rag_lookup", "arguments": "{not json"},
                        }
                    ],
                }
            }
        ]
    }
    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=bad),
            httpx.Response(200, json=_chat_response("recovered")),
        ]
    )
    result = await llm.complete([{"role": "user", "content": "x"}])
    assert result == "recovered"
    llm._rag.assert_not_awaited()
