import pytest

from agent.pjsip_poc import DelayedAnswerService, PjsipPocSettings


class FakeCall:
    def __init__(self, *, fail_answer: bool = False, fail_terminate: bool = False):
        self.ring_count = 0
        self.answer_count = 0
        self.busy_count = 0
        self.terminate_count = 0
        self.fail_answer = fail_answer
        self.fail_terminate = fail_terminate

    def signal_ringing(self):
        self.ring_count += 1

    def accept(self):
        self.answer_count += 1
        if self.fail_answer:
            raise RuntimeError("call disappeared")

    def reject_busy(self):
        self.busy_count += 1

    def terminate(self):
        self.terminate_count += 1
        if self.fail_terminate:
            raise RuntimeError("call already disappeared")


def _service(now):
    return DelayedAnswerService(
        answer_delay_seconds=20,
        max_call_seconds=30,
        clock=lambda: now[0],
    )


def test_call_rings_then_answers_after_delay():
    now = [100.0]
    call = FakeCall()
    service = _service(now)

    assert service.offer(1, "+49123", call) is True
    assert call.ring_count == 1

    now[0] = 119.9
    service.tick()
    assert call.answer_count == 0

    now[0] = 120.0
    service.tick()
    assert call.answer_count == 1
    assert service.call_count == 1


def test_human_answer_disconnect_prevents_agent_answer():
    now = [100.0]
    call = FakeCall()
    service = _service(now)
    service.offer(1, "+49123", call)

    service.disconnected(1, 487, "Request Terminated")
    now[0] = 125.0
    service.tick()

    assert call.answer_count == 0
    assert service.call_count == 0


def test_active_call_is_terminated_at_poc_limit():
    now = [100.0]
    call = FakeCall()
    service = _service(now)
    service.offer(1, "+49123", call)

    now[0] = 120.0
    service.tick()
    now[0] = 149.9
    service.tick()
    assert call.terminate_count == 0

    now[0] = 150.0
    service.tick()
    assert call.terminate_count == 1
    assert service.call_count == 0


def test_second_call_is_rejected_when_capacity_is_reached():
    now = [100.0]
    first = FakeCall()
    second = FakeCall()
    service = _service(now)

    service.offer(1, "+49123", first)
    assert service.offer(2, "+49456", second) is False

    assert second.ring_count == 0
    assert second.busy_count == 1
    assert service.call_count == 1


def test_failed_answer_is_removed():
    now = [100.0]
    call = FakeCall(fail_answer=True)
    service = _service(now)
    service.offer(1, "+49123", call)

    now[0] = 120.0
    service.tick()

    assert service.call_count == 0


def test_failed_timeout_hangup_is_removed():
    now = [100.0]
    call = FakeCall(fail_terminate=True)
    service = _service(now)
    service.offer(1, "+49123", call)

    now[0] = 120.0
    service.tick()
    now[0] = 150.0
    service.tick()

    assert call.terminate_count == 1
    assert service.call_count == 0


def test_settings_load_safe_defaults(monkeypatch):
    monkeypatch.setenv("FRITZBOX_SIP_USERNAME", "agent-phone")
    monkeypatch.setenv("FRITZBOX_SIP_PASSWORD", "strong-secret")
    monkeypatch.delenv("FRITZBOX_HOST", raising=False)
    monkeypatch.delenv("ANSWER_DELAY_SECONDS", raising=False)

    settings = PjsipPocSettings.from_env()

    assert settings.fritzbox_host == "fritz.box"
    assert settings.answer_delay_seconds == 20
    assert settings.identity_uri == "sip:agent-phone@fritz.box"
    assert settings.registrar_uri == "sip:fritz.box"


def test_tcp_settings_pin_registrar_transport(monkeypatch):
    monkeypatch.setenv("FRITZBOX_SIP_USERNAME", "agent-phone")
    monkeypatch.setenv("FRITZBOX_SIP_PASSWORD", "strong-secret")
    monkeypatch.setenv("PJSIP_TRANSPORT", "tcp")

    settings = PjsipPocSettings.from_env()

    assert settings.registrar_uri == "sip:fritz.box;transport=tcp"


@pytest.mark.parametrize("password", ["", "changeme", "CHANGEME"])
def test_settings_reject_placeholder_password(monkeypatch, password):
    monkeypatch.setenv("FRITZBOX_SIP_USERNAME", "agent-phone")
    monkeypatch.setenv("FRITZBOX_SIP_PASSWORD", password)

    with pytest.raises(ValueError, match="FRITZBOX_SIP_PASSWORD"):
        PjsipPocSettings.from_env()
