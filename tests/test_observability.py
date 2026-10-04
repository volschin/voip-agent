import httpx

from agent import observability


def test_registration_loss_and_stall_are_distinct():
    status = observability.AgentStatus()
    status.initialize(detector_loaded=True)
    status.registered = True
    assert status.ready()
    status.registered = False
    assert not status.ready()
    assert status.live()
    status.last_heartbeat -= 31
    assert not status.live()


def test_initialization_requires_detector():
    status = observability.AgentStatus()
    status.registered = True
    assert not status.ready()
    status.initialize(detector_loaded=False)
    assert not status.ready()


async def test_http_routes_reflect_real_status_and_metrics():
    status = observability.AgentStatus()
    async with observability.StatusServer(status, "127.0.0.1", 0) as server:
        origin = f"http://127.0.0.1:{server.port}"
        async with httpx.AsyncClient() as client:
            assert (await client.get(origin + "/readyz")).status_code == 503
            status.initialize(detector_loaded=True)
            status.registered = True
            assert (await client.get(origin + "/readyz")).status_code == 200
            status.observe_stage("stt", 0.5, error=True)
            status.priority_error()
            body = (await client.get(origin + "/metrics")).text
            assert 'voip_agent_stage_errors_total{stage="stt"} 1' in body
            assert "voip_agent_sip_registered 1" in body
            assert "voip_agent_priority_errors_total 1" in body
            assert (await client.post(origin + "/metrics")).status_code == 405
            assert (await client.get(origin + "/unknown")).status_code == 404
            status.last_heartbeat -= 31
            assert (await client.get(origin + "/livez")).status_code == 503
