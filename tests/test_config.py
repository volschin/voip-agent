import pytest

from agent.config import Settings


def _valid_kwargs(**over):
    kwargs = dict(
        fritzbox_sip_username="agent-phone",
        fritzbox_sip_password="strong-secret",
        azure_tenant_id="t",
        azure_client_id="c",
        azure_client_secret="s",
        calendar_user_email="x@x.com",
    )
    kwargs.update(over)
    return kwargs


@pytest.mark.parametrize("bad", ["changeme", "CHANGEME", "", "  "])
def test_insecure_sip_password_rejected(monkeypatch, bad):
    monkeypatch.delenv("FRITZBOX_SIP_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="fritzbox_sip_password"):
        Settings(**_valid_kwargs(fritzbox_sip_password=bad))


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
        fritzbox_sip_username="agent-phone",
        fritzbox_sip_password="strong-secret",
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
        ai_proxy_username="",
        ai_proxy_password_file="",
        ai_proxy_ca_file="",
        voice_priority_token_file="",
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


def test_turn_detection_defaults(settings):
    assert settings.turn_detection_enabled is True
    assert settings.turn_complete_threshold == 0.70
    assert settings.turn_vad_silence_ms == 200
    assert settings.turn_model_repo == "pipecat-ai/smart-turn-v3"
    assert settings.turn_model_filename == "smart-turn-v3.2-cpu.onnx"
    assert settings.turn_model_revision == "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
    assert settings.turn_onnx_providers == "CPUExecutionProvider"
    assert settings.turn_onnx_provider_list == ["CPUExecutionProvider"]
    # Removed HTTP-only fields must be gone.
    assert not hasattr(settings, "turn_detector_url")
    assert not hasattr(settings, "turn_classify_timeout_ms")


def test_shared_ai_credentials_require_exact_traefik_origins(tmp_path):
    password = tmp_path / "password"
    ca = tmp_path / "ca.crt"
    token = tmp_path / "priority-token"
    password.write_text("secret", encoding="utf-8")
    ca.write_text("ca", encoding="utf-8")
    token.write_text("token", encoding="utf-8")
    shared = {
        "ai_proxy_username": "voip-agent",
        "ai_proxy_password_file": str(password),
        "ai_proxy_ca_file": str(ca),
        "voice_priority_token_file": str(token),
    }

    settings = Settings(
        **_valid_kwargs(
            stt_base_url="https://mate.olcon.de",
            tts_base_url="https://mate.olcon.de",
            llm_base_url="https://mate.olcon.de",
            **shared,
        )
    )

    assert settings.voice_priority_base_url == "https://mate.olcon.de"


def test_shared_ai_defaults_are_fail_closed_to_traefik() -> None:
    settings = Settings(**_valid_kwargs())

    assert settings.ai_origin == "https://mate.olcon.de"
    assert settings.stt_base_url == "https://mate.olcon.de"
    assert settings.tts_base_url == "https://mate.olcon.de"
    assert settings.llm_base_url == "https://mate.olcon.de"
    assert settings.llm_model == "companion-gemma"
    assert settings.ai_proxy_username == "voip-agent"
    assert settings.ai_proxy_password_file == "/run/secrets/shared_ai_password"
    assert settings.ai_proxy_ca_file == "/run/secrets/mate_ca.crt"
    assert settings.voice_priority_token_file == "/run/secrets/voice_priority_token"
    assert settings.voice_priority_base_url == "https://mate.olcon.de"
    assert settings.tts_voice_profile == "shared-female-de-v1"


_UNSET_SERVICE_URLS = {
    "stt_base_url": "",
    "tts_base_url": "",
    "llm_base_url": "",
    "voice_priority_base_url": "",
}


def test_unset_service_urls_follow_configured_ai_origin() -> None:
    settings = Settings(
        **_valid_kwargs(ai_origin="https://voice.example.test/", **_UNSET_SERVICE_URLS)
    )

    assert settings.ai_origin == "https://voice.example.test"
    assert settings.stt_base_url == "https://voice.example.test"
    assert settings.tts_base_url == "https://voice.example.test"
    assert settings.llm_base_url == "https://voice.example.test"
    assert settings.voice_priority_base_url == "https://voice.example.test"


def test_explicit_service_urls_must_match_configured_ai_origin(tmp_path) -> None:
    password = tmp_path / "password"
    ca = tmp_path / "ca.crt"
    token = tmp_path / "priority-token"
    password.write_text("secret", encoding="utf-8")
    ca.write_text("ca", encoding="utf-8")
    token.write_text("token", encoding="utf-8")
    shared = {
        **_UNSET_SERVICE_URLS,
        "ai_origin": "https://voice.example.test",
        "ai_proxy_username": "voip-agent",
        "ai_proxy_password_file": str(password),
        "ai_proxy_ca_file": str(ca),
        "voice_priority_token_file": str(token),
    }

    settings = Settings(
        **_valid_kwargs(**{**shared, "stt_base_url": "https://voice.example.test/"})
    )
    assert settings.stt_base_url == "https://voice.example.test/"

    with pytest.raises(ValueError, match="voice.example.test"):
        Settings(**_valid_kwargs(**{**shared, "stt_base_url": "https://mate.olcon.de"}))


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "http://mate.olcon.de", "mate.olcon.de", "https://mate.olcon.de/v1"],
)
def test_ai_origin_rejects_non_https_or_path_values(monkeypatch, bad: str) -> None:
    monkeypatch.delenv("AI_ORIGIN", raising=False)

    with pytest.raises(ValueError, match="ai_origin"):
        Settings(**_valid_kwargs(ai_origin=bad))


@pytest.mark.parametrize("value", ["", " "])
def test_tts_voice_profile_rejects_blank_values(monkeypatch, value: str) -> None:
    monkeypatch.delenv("TTS_VOICE_PROFILE", raising=False)

    with pytest.raises(ValueError, match="tts_voice_profile"):
        Settings(**_valid_kwargs(tts_voice_profile=value))


@pytest.mark.parametrize(
    "field",
    ["stt_base_url", "tts_base_url", "llm_base_url", "voice_priority_base_url"],
)
def test_shared_ai_credentials_reject_direct_or_different_origins(
    tmp_path,
    field,
):
    password = tmp_path / "password"
    ca = tmp_path / "ca.crt"
    token = tmp_path / "priority-token"
    password.write_text("secret", encoding="utf-8")
    ca.write_text("ca", encoding="utf-8")
    token.write_text("token", encoding="utf-8")
    values = {
        "stt_base_url": "https://mate.olcon.de",
        "tts_base_url": "https://mate.olcon.de",
        "llm_base_url": "https://mate.olcon.de",
        "voice_priority_base_url": "https://mate.olcon.de",
        "ai_proxy_username": "voip-agent",
        "ai_proxy_password_file": str(password),
        "ai_proxy_ca_file": str(ca),
        "voice_priority_token_file": str(token),
    }
    values[field] = "http://dgx-spark:8001"

    with pytest.raises(ValueError, match="mate.olcon.de"):
        Settings(**_valid_kwargs(**values))
