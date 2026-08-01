import pytest

from agent.pjsip_poc import PjsipPocSettings


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
