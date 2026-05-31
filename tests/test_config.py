import pytest

from agent.config import Settings


def _valid_kwargs(**over):
    kwargs = dict(
        ari_password="strong-secret",
        azure_tenant_id="t",
        azure_client_id="c",
        azure_client_secret="s",
        calendar_user_email="x@x.com",
    )
    kwargs.update(over)
    return kwargs


@pytest.mark.parametrize("bad", ["changeme", "CHANGEME", "", "  "])
def test_insecure_ari_password_rejected(monkeypatch, bad):
    monkeypatch.delenv("ARI_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="ari_password"):
        Settings(**_valid_kwargs(ari_password=bad))


@pytest.mark.parametrize("bad", ["0.0.0.0", "", "  "])
def test_unroutable_rtp_advertise_host_rejected(monkeypatch, bad):
    monkeypatch.delenv("RTP_ADVERTISE_HOST", raising=False)
    with pytest.raises(ValueError, match="rtp_advertise_host"):
        Settings(**_valid_kwargs(rtp_advertise_host=bad))


def test_rtp_advertise_host_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("RTP_ADVERTISE_HOST", raising=False)
    s = Settings(**_valid_kwargs())
    assert s.rtp_advertise_host == "127.0.0.1"


def test_calendar_write_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CALENDAR_WRITE_ENABLED", raising=False)
    s = Settings(**_valid_kwargs())
    assert s.calendar_write_enabled is False
    assert s.max_tool_rounds == 5


def test_trusted_callers_empty_by_default(monkeypatch):
    monkeypatch.delenv("TRUSTED_CALLERS", raising=False)
    s = Settings(**_valid_kwargs())
    assert s.trusted_caller_set == frozenset()  # fail closed: tools off for all


def test_trusted_caller_set_parses_and_trims(monkeypatch):
    monkeypatch.delenv("TRUSTED_CALLERS", raising=False)
    s = Settings(**_valid_kwargs(trusted_callers=" +49123, +49999 ,"))
    assert s.trusted_caller_set == frozenset({"+49123", "+49999"})


def test_settings_load_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("ARI_BASE_URL", raising=False)
    s = Settings(
        ari_base_url="http://test:8088",
        ari_username="u",
        ari_password="p",
        ari_app_name="app",
        rtp_bind_host="0.0.0.0",
        rtp_port=5001,
        stt_base_url="http://stt",
        tts_base_url="http://tts",
        llm_base_url="http://llm",
        llm_model="hermes",
        embedding_base_url="http://emb",
        db_dsn="postgresql://x",
        azure_tenant_id="t",
        azure_client_id="c",
        azure_client_secret="s",
        calendar_user_email="x@x.com",
        greeting_text="Hi!",
        llm_system_prompt="prompt",
    )
    assert s.ari_base_url == "http://test:8088"
    assert s.rtp_port == 5001
    assert s.llm_model == "hermes"
    assert s.greeting_text == "Hi!"
