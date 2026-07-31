import logging

from agent.answer_policy import DelayedAnswerService, caller_id_from_uri


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


def test_active_call_is_terminated_at_duration_limit():
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


def test_caller_id_is_logged_across_the_call_lifecycle(caplog):
    now = [100.0]
    call = FakeCall()
    service = _service(now)

    with caplog.at_level(logging.INFO, logger="agent.answer_policy"):
        service.offer(1, "+4930123", call)
        now[0] = 120.0
        service.tick()
        service.disconnected(1, 200, "Normal call clearing")

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Call 1 from +4930123 waiting 20.0 seconds before answer",
        "Call 1 from +4930123 answered by agent",
        "Call 1 from +4930123 ended (status=200 Normal call clearing)",
    ]


def test_cancelled_call_logs_caller_id(caplog):
    now = [100.0]
    call = FakeCall()
    service = _service(now)

    with caplog.at_level(logging.INFO, logger="agent.answer_policy"):
        service.offer(1, "+4930123", call)
        service.disconnected(1, 487, "Request Terminated")

    assert (
        "Call 1 from +4930123 cancelled before agent answer (status=487 Request Terminated)"
        in [record.getMessage() for record in caplog.records]
    )


def test_missing_caller_id_falls_back_to_unknown(caplog):
    now = [100.0]
    call = FakeCall()
    service = _service(now)

    with caplog.at_level(logging.INFO, logger="agent.answer_policy"):
        service.offer(1, "", call)

    assert "Call 1 from unknown waiting" in caplog.records[0].getMessage()


def test_caller_id_from_uri_extracts_the_sip_user():
    assert caller_id_from_uri("<sip:+4930123@fritz.box>;tag=abc") == "+4930123"
    assert caller_id_from_uri("anonymous") == ""


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
