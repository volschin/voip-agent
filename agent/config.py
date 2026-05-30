"""Configuration management for VoIP agent."""

from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings

_INSECURE_PASSWORDS = {"", "changeme"}


class Settings(BaseSettings):
    """Agent settings loaded from environment variables."""

    model_config = ConfigDict(env_file=".env", case_sensitive=False)

    # Asterisk ARI
    ari_base_url: str = "http://localhost:8088"
    ari_username: str = "voip-agent"
    ari_password: str = "changeme"
    ari_app_name: str = "voip-agent"

    @field_validator("ari_password")
    @classmethod
    def _reject_insecure_ari_password(cls, v: str) -> str:
        # Fail closed: the ARI socket grants full call control. Refuse to start
        # with the placeholder password so a default install is never exposed.
        if v.strip().lower() in _INSECURE_PASSWORDS:
            raise ValueError(
                "ari_password is unset or 'changeme'. Set ARI_PASSWORD to a strong "
                "secret matching asterisk/ari.conf."
            )
        return v

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

    # Tool / LLM safety
    calendar_write_enabled: bool = False  # fail-closed: callers cannot create events unless opted in
    max_tool_rounds: int = 5  # cap LLM tool-call loop to prevent runaway dispatch
    # Comma-separated caller numbers allowed to use tools (RAG + calendar).
    # Empty = no caller is authorized = tools off for everyone (fail closed).
    trusted_callers: str = ""

    @property
    def trusted_caller_set(self) -> frozenset[str]:
        return frozenset(c.strip() for c in self.trusted_callers.split(",") if c.strip())

    # Agent behaviour
    caller_id: str = "+49123456789"
    greeting_text: str = "Hallo, wie kann ich Ihnen helfen?"
    llm_system_prompt: str = (
        "Du bist ein hilfreicher Telefonassistent. "
        "Antworte immer auf Deutsch. Sei freundlich und präzise. "
        "Nutze rag_lookup für Wissensfragen und die Kalender-Werkzeuge für Terminanfragen."
    )
