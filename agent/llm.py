import json
from typing import Any

import httpx


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_lookup",
            "description": "Durchsuche die Wissensdatenbank nach relevanten Informationen.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_get_events",
            "description": "Kalendertermine für einen Zeitraum abrufen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "ISO 8601 datetime"},
                    "end": {"type": "string", "description": "ISO 8601 datetime"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "Neuen Kalendertermin erstellen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "ISO 8601 datetime"},
                    "end": {"type": "string", "description": "ISO 8601 datetime"},
                    "description": {"type": "string", "default": ""},
                },
                "required": ["title", "start", "end"],
            },
        },
    },
]


class LlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str,
        rag: Any,
        calendar: Any,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._system_prompt = system_prompt
        self._rag = rag
        self._calendar = calendar

    async def complete(self, messages: list[dict]) -> str:
        full_messages = [
            {"role": "system", "content": self._system_prompt},
            *messages,
        ]
        async with httpx.AsyncClient() as client:
            while True:
                resp = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={"model": self._model, "messages": full_messages, "tools": TOOLS},
                    timeout=60.0,
                )
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]

                if not msg.get("tool_calls"):
                    return msg["content"] or ""

                full_messages.append(msg)
                for tc in msg["tool_calls"]:
                    result = await self._dispatch(
                        tc["function"]["name"],
                        json.loads(tc["function"]["arguments"]),
                    )
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

    async def _dispatch(self, name: str, args: dict) -> str:
        if name == "rag_lookup":
            return await self._rag(args["query"])
        if name == "calendar_get_events":
            return await self._calendar.get_events(args["start"], args["end"])
        if name == "calendar_create_event":
            return await self._calendar.create_event(
                title=args["title"],
                start=args["start"],
                end=args["end"],
                description=args.get("description", ""),
            )
        return f"Unknown tool: {name}"
