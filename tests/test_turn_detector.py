import httpx
import numpy as np
import pytest
import respx

from agent.turn_detector import TurnDetectorClient


@respx.mock
async def test_classify_complete_above_threshold():
    respx.post("http://td:8004/v1/turn/classify").mock(
        return_value=httpx.Response(200, json={"complete": True, "prob": 0.9})
    )
    c = TurnDetectorClient(base_url="http://td:8004", threshold=0.5)
    assert await c.classify(np.zeros(1600, dtype=np.int16)) is True
    await c.aclose()


@respx.mock
async def test_classify_incomplete_below_threshold():
    respx.post("http://td:8004/v1/turn/classify").mock(
        return_value=httpx.Response(200, json={"complete": False, "prob": 0.2})
    )
    c = TurnDetectorClient(base_url="http://td:8004", threshold=0.5)
    assert await c.classify(np.zeros(1600, dtype=np.int16)) is False
    await c.aclose()


@respx.mock
async def test_classify_raises_on_server_error():
    respx.post("http://td:8004/v1/turn/classify").mock(return_value=httpx.Response(500))
    c = TurnDetectorClient(base_url="http://td:8004")
    with pytest.raises(httpx.HTTPStatusError):
        await c.classify(np.zeros(1600, dtype=np.int16))
    await c.aclose()
