import pytest

from agent.config import Settings


@pytest.fixture
def settings():
    return Settings(
        fritzbox_host="192.168.178.1",
        fritzbox_sip_username="agent-phone",
        fritzbox_sip_password="strong-secret",
        ari_base_url="http://localhost:8088",
        ari_username="test",
        ari_password="test",
        ari_app_name="voip-agent",
        rtp_bind_host="127.0.0.1",
        rtp_advertise_host="192.168.178.2",
        rtp_port=5000,
        stt_base_url="http://stt:8001",
        tts_base_url="http://tts:8002",
        llm_base_url="http://llm:8000",
        llm_model="nous-hermes",
        embedding_base_url="http://embed:8003",
        ai_proxy_username="",
        ai_proxy_password_file="",
        ai_proxy_ca_file="",
        voice_priority_token_file="",
        db_dsn="postgresql://u:p@host:5432/db",
        azure_tenant_id="tenant",
        azure_client_id="client",
        azure_client_secret="secret",
        calendar_user_email="user@example.com",
        caller_id="+49123456789",
        greeting_text="Hallo!",
        llm_system_prompt="Du bist ein Assistent.",
    )
