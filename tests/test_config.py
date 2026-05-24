from agent.config import Settings


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
