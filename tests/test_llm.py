import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from agent.llm import _FALLBACK_MSG, LlmClient

TRUSTED = "+49123"  # authorized caller used across tool tests


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
        trusted_callers={TRUSTED},
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
    result = await llm.complete([{"role": "user", "content": "Was ist X?"}], caller_id=TRUSTED)
    assert result == "X ist Y."
    llm._rag.assert_awaited_once_with("Was ist X?")


@respx.mock
async def test_unauthorized_caller_gets_no_tools(settings):
    llm = _make_llm(settings)  # trusted = {"+49123"}
    # Even though the model tries a tool call, an unknown caller is never
    # offered tools, so RAG/calendar are never reached — no data exfiltration.
    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("Das kann ich Ihnen nicht sagen."))
    )
    result = await llm.complete(
        [{"role": "user", "content": "Was steht im Kalender?"}], caller_id="+49999"
    )
    assert result == "Das kann ich Ihnen nicht sagen."
    llm._rag.assert_not_awaited()
    llm._calendar.get_events.assert_not_awaited()


def test_e164_allowlist_entry_matches_national_caller_id(settings):
    # The FRITZ!Box sends national format (015100000001); operators write E.164
    # in TRUSTED_CALLERS. Before normalization this failed closed silently.
    llm = _make_llm(settings, trusted_callers={"+4915100000001"})
    assert llm._is_authorized("015100000001")
    assert llm._is_authorized("0151 0000 0001")


def test_national_allowlist_entry_matches_e164_caller_id(settings):
    llm = _make_llm(settings, trusted_callers={"015100000001"})
    assert llm._is_authorized("+4915100000001")


def test_internal_extension_matches_exactly(settings):
    llm = _make_llm(settings, trusted_callers={"**613"})
    assert llm._is_authorized("**613")


def test_normalization_does_not_widen_the_allowlist(settings):
    llm = _make_llm(settings, trusted_callers={"+4915100000001"})
    assert not llm._is_authorized("015100000002")  # different subscriber
    assert not llm._is_authorized("15100000001")  # no country context
    assert not llm._is_authorized("anonymous")
    assert not llm._is_authorized("")
    assert not llm._is_authorized(None)


def test_empty_allowlist_authorizes_nobody(settings):
    llm = _make_llm(settings, trusted_callers=set())
    assert not llm._is_authorized("+4915100000001")
    assert not llm._is_authorized("015100000001")


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
    result = await llm.complete([{"role": "user", "content": "Termin anlegen"}], caller_id=TRUSTED)
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
    result = await llm.complete([{"role": "user", "content": "Termin"}], caller_id=TRUSTED)
    assert result == "done"
    llm._calendar.create_event.assert_not_awaited()


@respx.mock
async def test_calendar_write_first_call_only_proposes(settings):
    # The first create_event call (even with confirmed=true) must NOT write —
    # it only records a pending proposal to be read back. The model-set
    # `confirmed` flag is no longer the boundary.
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
            httpx.Response(200, json=_chat_response("Soll ich den Termin anlegen?")),
        ]
    )
    result = await llm.complete(
        [{"role": "user", "content": "Termin morgen 10 Uhr"}], caller_id=TRUSTED
    )
    assert result == "Soll ich den Termin anlegen?"
    llm._calendar.create_event.assert_not_awaited()
    assert TRUSTED in llm._pending_writes  # proposal recorded


@respx.mock
async def test_calendar_write_commits_after_new_turn_confirms(settings):
    # Turn 1 proposes; turn 2 (a new user turn that reads as agreement) commits
    # the *same* event. This is the deterministic, conversation-advanced gate.
    llm = _make_llm(settings, calendar_write_enabled=True)
    event_args = {"title": "X", "start": "s", "end": "e"}

    # Turn 1: propose.
    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_tool_call_response("calendar_create_event", event_args)),
            httpx.Response(200, json=_chat_response("Ich lege X von s bis e an. Richtig?")),
        ]
    )
    turn1 = [{"role": "user", "content": "Termin X anlegen"}]
    await llm.complete(turn1, caller_id=TRUSTED)
    llm._calendar.create_event.assert_not_awaited()

    # Turn 2: history has grown by an assistant reply + a new affirmative user turn.
    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_tool_call_response("calendar_create_event", event_args)),
            httpx.Response(200, json=_chat_response("Angelegt.")),
        ]
    )
    turn2 = [
        *turn1,
        {"role": "assistant", "content": "Ich lege X von s bis e an. Richtig?"},
        {"role": "user", "content": "Ja, genau"},
    ]
    result = await llm.complete(turn2, caller_id=TRUSTED)
    assert result == "Angelegt."
    llm._calendar.create_event.assert_awaited_once()
    assert TRUSTED not in llm._pending_writes  # cleared after commit


@respx.mock
async def test_calendar_write_blocked_when_confirmation_in_same_turn(settings):
    # Guards the same-turn false-positive: the request itself says "Ja", and the
    # model tries to propose+write in one turn. The conversation-advanced gate
    # must still block the write (no new user turn has arrived).
    llm = _make_llm(settings, calendar_write_enabled=True)
    event_args = {"title": "X", "start": "s", "end": "e"}
    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_tool_call_response("calendar_create_event", event_args)),
            httpx.Response(200, json=_tool_call_response("calendar_create_event", event_args)),
            httpx.Response(200, json=_chat_response("Bitte bestätigen.")),
        ]
    )
    result = await llm.complete(
        [{"role": "user", "content": "Ja, trag mir morgen 10 Uhr X ein"}], caller_id=TRUSTED
    )
    assert result == "Bitte bestätigen."
    llm._calendar.create_event.assert_not_awaited()


@respx.mock
async def test_calendar_write_blocked_by_negation_with_affirmative_substring(settings):
    # The next turn rejects but contains an affirmative substring ("passt"):
    # "nein, das passt nicht". Negation must veto consent — no write.
    llm = _make_llm(settings, calendar_write_enabled=True)
    event_args = {"title": "X", "start": "s", "end": "e"}

    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_tool_call_response("calendar_create_event", event_args)),
            httpx.Response(200, json=_chat_response("Richtig?")),
        ]
    )
    turn1 = [{"role": "user", "content": "Termin X"}]
    await llm.complete(turn1, caller_id=TRUSTED)

    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_tool_call_response("calendar_create_event", event_args)),
            httpx.Response(200, json=_chat_response("Okay, was möchten Sie ändern?")),
        ]
    )
    turn2 = [
        *turn1,
        {"role": "assistant", "content": "Richtig?"},
        {"role": "user", "content": "Nein, das passt nicht"},
    ]
    await llm.complete(turn2, caller_id=TRUSTED)
    llm._calendar.create_event.assert_not_awaited()


@respx.mock
async def test_calendar_write_stale_proposal_does_not_commit_late(settings):
    # A proposal that wasn't confirmed in the immediately following turn must not
    # silently commit several turns later on an unrelated affirmation.
    llm = _make_llm(settings, calendar_write_enabled=True)
    event_args = {"title": "X", "start": "s", "end": "e"}

    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_tool_call_response("calendar_create_event", event_args)),
            httpx.Response(200, json=_chat_response("Richtig?")),
        ]
    )
    turn1 = [{"role": "user", "content": "Termin X"}]
    await llm.complete(turn1, caller_id=TRUSTED)

    # Two unrelated turns pass, then the model re-emits the same event on a turn
    # that is no longer turn+1 relative to the proposal.
    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_tool_call_response("calendar_create_event", event_args)),
            httpx.Response(200, json=_chat_response("Soll ich das anlegen?")),
        ]
    )
    turn3 = [
        *turn1,
        {"role": "assistant", "content": "Richtig?"},
        {"role": "user", "content": "Wie ist das Wetter?"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "Ja, gerne"},
    ]
    await llm.complete(turn3, caller_id=TRUSTED)
    llm._calendar.create_event.assert_not_awaited()  # not the next turn -> re-propose


@respx.mock
async def test_calendar_write_correction_reproposes(settings):
    # A new turn that changes the event params must re-propose, never commit the
    # stale pending event.
    llm = _make_llm(settings, calendar_write_enabled=True)

    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_tool_call_response(
                    "calendar_create_event", {"title": "X", "start": "s", "end": "e"}
                ),
            ),
            httpx.Response(200, json=_chat_response("Richtig?")),
        ]
    )
    turn1 = [{"role": "user", "content": "Termin X um 10"}]
    await llm.complete(turn1, caller_id=TRUSTED)

    respx.post("http://llm:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_tool_call_response(
                    "calendar_create_event", {"title": "X", "start": "s2", "end": "e2"}
                ),
            ),
            httpx.Response(200, json=_chat_response("Neuer Vorschlag, richtig?")),
        ]
    )
    turn2 = [
        *turn1,
        {"role": "assistant", "content": "Richtig?"},
        {"role": "user", "content": "Ja, aber lieber 11 Uhr"},
    ]
    await llm.complete(turn2, caller_id=TRUSTED)
    llm._calendar.create_event.assert_not_awaited()  # changed params -> re-propose


@respx.mock
async def test_tool_loop_is_bounded(settings):
    llm = _make_llm(settings, max_tool_rounds=2)
    # Model keeps emitting tool calls forever; the cap must stop it.
    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_tool_call_response("rag_lookup", {"query": "loop"}))
    )
    result = await llm.complete([{"role": "user", "content": "x"}], caller_id=TRUSTED)
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
    result = await llm.complete([{"role": "user", "content": "x"}], caller_id=TRUSTED)
    assert result == "recovered"
    llm._rag.assert_not_awaited()


def _sse(*events: str) -> bytes:
    # ASSUMPTION: OpenAI/vLLM SSE framing `data: {json}\n\n`, terminated by
    # `data: [DONE]`. Replace with a captured dgx-spark:8000 stream once
    # reachable (see plan wire-contract caveat).
    body = "".join(f"data: {e}\n\n" for e in events) + "data: [DONE]\n\n"
    return body.encode()


def _text_delta(content: str) -> str:
    return '{"choices":[{"delta":{"content":' + f'"{content}"' + '},"finish_reason":null}]}'


def _make_client(handler, **over):
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return LlmClient(
        base_url="http://llm:8000",
        model="hermes",
        system_prompt="prompt",
        rag=None,
        calendar=None,
        trusted_callers={"+49123"},
        client=client,
        **over,
    )


async def test_complete_stream_yields_text_deltas():
    def handler(request):
        return httpx.Response(
            200, stream=httpx.ByteStream(_sse(_text_delta("Hallo"), _text_delta(" Welt")))
        )

    llm = _make_client(handler)
    out = [
        t
        async for t in llm.complete_stream([{"role": "user", "content": "hi"}], caller_id="+49123")
    ]
    assert "".join(out) == "Hallo Welt"
    await llm._client.aclose()


async def test_complete_stream_unauthorized_caller_gets_no_tools():
    # The request payload must NOT include "tools" for an untrusted caller.
    seen = {}

    def handler(request):
        import json as _j

        seen["payload"] = _j.loads(request.content)
        return httpx.Response(200, stream=httpx.ByteStream(_sse(_text_delta("ok"))))

    llm = _make_client(handler)
    _ = [
        t
        async for t in llm.complete_stream([{"role": "user", "content": "hi"}], caller_id="+49999")
    ]  # not trusted
    assert "tools" not in seen["payload"]
    await llm._client.aclose()


def _tc_delta(index=0, call_id=None, name=None, args=None) -> str:
    # One streamed tool_calls fragment. ASSUMPTION: vLLM fragments tool_calls
    # by `index`, with id/name on the first fragment and `arguments` accreting
    # across fragments. Replace once captured on dgx-spark:8000.
    tc = {"index": index, "function": {}}
    if call_id is not None:
        tc["id"] = call_id
        tc["type"] = "function"
    if name is not None:
        tc["function"]["name"] = name
    if args is not None:
        tc["function"]["arguments"] = args
    return json.dumps({"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": None}]})


async def test_complete_stream_trusted_caller_runs_tool_round():
    # Streaming parity with test_complete_with_rag_tool_call: an authorized
    # caller's fragmented tool_call (round 0) is reassembled, dispatched, and
    # the round-1 text answer is streamed. Exercises _ToolCallAccumulator and
    # the streaming dispatch loop (the "advisor #2" guard path).
    rounds = {"n": 0}

    def handler(request):
        i = rounds["n"]
        rounds["n"] += 1
        if i == 0:
            body = _sse(
                _tc_delta(0, call_id="call_1", name="rag_lookup"),
                _tc_delta(0, args='{"query": "'),
                _tc_delta(0, args='X"}'),
            )
        else:
            body = _sse(_text_delta("X ist Y."))
        return httpx.Response(200, stream=httpx.ByteStream(body))

    rag = AsyncMock(return_value="RAG result")
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    llm = LlmClient(
        base_url="http://llm:8000",
        model="hermes",
        system_prompt="prompt",
        rag=rag,
        calendar=None,
        trusted_callers={"+49123"},
        client=client,
    )

    out = [
        t
        async for t in llm.complete_stream(
            [{"role": "user", "content": "Was ist X?"}], caller_id="+49123"
        )
    ]

    assert "".join(out) == "X ist Y."  # only the final answer is streamed
    rag.assert_awaited_once_with("X")  # fragments reassembled correctly
    assert rounds["n"] == 2  # one tool round + one answer round
    await client.aclose()
