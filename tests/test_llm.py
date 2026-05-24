import json
import pytest
import respx
import httpx
from unittest.mock import AsyncMock
from agent.llm import LlmClient


@pytest.fixture
def llm(settings):
    rag = AsyncMock(return_value="RAG result")
    calendar = AsyncMock()
    calendar.get_events = AsyncMock(return_value="No events")
    calendar.create_event = AsyncMock(return_value="Event created")
    return LlmClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        system_prompt=settings.llm_system_prompt,
        rag=rag,
        calendar=calendar,
    )


def _chat_response(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": None}}]
    }


def _tool_call_response(name: str, arguments: dict) -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            }
        }]
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
