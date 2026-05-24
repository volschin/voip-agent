"""Configuration management for VoIP agent."""
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Agent settings loaded from environment variables."""

    model_config = ConfigDict(env_file=".env", case_sensitive=False)

    # Asterisk ARI
    ari_base_url: str = "http://localhost:8088"
    ari_username: str = "voip-agent"
    ari_password: str = "changeme"
    ari_app_name: str = "voip-agent"

    # RTP
    rtp_bind_host: str = "0.0.0.0"
    rtp_port: int = 5000

    # DGX Spark AI services
    stt_base_url: str = "http://dgx-spark:8001"
    tts_base_url: str = "http://dgx-spark:8002"
    llm_base_url: str = "http://dgx-spark:8000"
    llm_model: str = "nous-hermes"
    embedding_base_url: str = "http://dgx-spark:8003"

    # pgvector RAG
    db_dsn: str = "postgresql://user:pass@dgx-spark:5432/voip"

    # MS Graph Calendar
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""
    calendar_user_email: str = ""

    # Agent behaviour
    caller_id: str = "+49123456789"
    greeting_text: str = "Hallo, wie kann ich Ihnen helfen?"
    llm_system_prompt: str = (
        "Du bist ein hilfreicher Telefonassistent. "
        "Antworte immer auf Deutsch. Sei freundlich und präzise. "
        "Nutze rag_lookup für Wissensfragen und die Kalender-Werkzeuge für Terminanfragen."
    )
