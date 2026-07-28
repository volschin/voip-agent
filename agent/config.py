"""Configuration management for VoIP agent."""

from pydantic import ConfigDict, field_validator, model_validator
from pydantic_settings import BaseSettings

_INSECURE_PASSWORDS = {"", "changeme"}


class Settings(BaseSettings):
    """Agent settings loaded from environment variables."""

    model_config = ConfigDict(env_file=".env", case_sensitive=False)

    # Direct FRITZ!Box PJSIP transport
    fritzbox_host: str = "fritz.box"
    fritzbox_sip_username: str = ""
    fritzbox_sip_password: str = ""
    pjsip_transport: str = "udp"
    pjsip_local_port: int = 5062
    pjsip_log_level: int = 2
    pjsip_event_poll_ms: int = 10
    answer_delay_seconds: float = 20.0
    max_call_seconds: float = 900.0
    max_concurrent_calls: int = 1

    @field_validator("fritzbox_sip_username")
    @classmethod
    def _require_sip_username(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("fritzbox_sip_username must not be empty")
        return value.strip()

    @field_validator("fritzbox_sip_password")
    @classmethod
    def _reject_insecure_sip_password(cls, value: str) -> str:
        if value.strip().lower() in _INSECURE_PASSWORDS:
            raise ValueError("fritzbox_sip_password must be a non-placeholder secret")
        return value

    @field_validator("pjsip_transport")
    @classmethod
    def _validate_pjsip_transport(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"udp", "tcp"}:
            raise ValueError("pjsip_transport must be 'udp' or 'tcp'")
        return value

    # Legacy Asterisk settings retained only for the rollback adapter/tests.
    # The production entry point no longer reads or validates them.
    ari_base_url: str = "http://localhost:8088"
    ari_username: str = "voip-agent"
    ari_password: str = "changeme"
    ari_app_name: str = "voip-agent"

    # RTP
    rtp_bind_host: str = "0.0.0.0"
    # IP that Asterisk's ExternalMedia sends RTP *to* (the destination it dials
    # out to with connection_type=client). Must be an address reachable from
    # Asterisk — never 0.0.0.0, which is a valid bind but an invalid
    # destination. Defaults to loopback because the agent is co-located with
    # Asterisk (see ari_base_url=localhost). Set to the agent host's LAN IP if
    # the agent runs on a different machine than Asterisk.
    rtp_advertise_host: str = "127.0.0.1"
    rtp_port: int = 5000

    @field_validator("rtp_advertise_host")
    @classmethod
    def _reject_unroutable_advertise_host(cls, v: str) -> str:
        # 0.0.0.0 binds every interface but cannot be a media destination:
        # Asterisk would have nowhere to send RTP and the call would have no
        # audio. Fail fast instead of shipping a silent call.
        if v.strip() in {"", "0.0.0.0"}:
            raise ValueError(
                "rtp_advertise_host must be an address reachable from Asterisk "
                "(e.g. 127.0.0.1 if co-located, or the agent host's LAN IP). "
                "0.0.0.0 is a bind address, not a routable media destination."
            )
        return v

    # DGX Spark AI services
    stt_base_url: str = "http://dgx-spark:8001"
    tts_base_url: str = "http://dgx-spark:8002"
    llm_base_url: str = "http://dgx-spark:8000"
    llm_model: str = "nous-hermes"
    embedding_base_url: str = "http://dgx-spark:8003"
    ai_proxy_username: str = ""
    ai_proxy_password_file: str = ""
    ai_proxy_ca_file: str = ""
    voice_priority_token_file: str = ""
    voice_priority_base_url: str = "https://mate.olcon.de"

    @model_validator(mode="after")
    def _validate_shared_ai_boundary(self) -> "Settings":
        credentials = (
            self.ai_proxy_username,
            self.ai_proxy_password_file,
            self.ai_proxy_ca_file,
            self.voice_priority_token_file,
        )
        if not any(value.strip() for value in credentials):
            return self
        if not all(value.strip() for value in credentials):
            raise ValueError(
                "AI proxy username, password file, CA file, and priority token "
                "file must be configured together"
            )
        for name, url in (
            ("stt_base_url", self.stt_base_url),
            ("tts_base_url", self.tts_base_url),
            ("llm_base_url", self.llm_base_url),
            ("voice_priority_base_url", self.voice_priority_base_url),
        ):
            if url.rstrip("/") != "https://mate.olcon.de":
                raise ValueError(f"{name} must use https://mate.olcon.de")
        return self

    # pgvector RAG
    db_dsn: str = "postgresql://user:pass@dgx-spark:5432/voip"

    # MS Graph Calendar
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""
    calendar_user_email: str = ""

    # Tool / LLM safety
    # fail-closed: callers cannot create events unless opted in
    calendar_write_enabled: bool = False
    max_tool_rounds: int = 5  # cap LLM tool-call loop to prevent runaway dispatch
    # Comma-separated caller numbers allowed to use tools (RAG + calendar).
    # Empty = no caller is authorized = tools off for everyone (fail closed).
    trusted_callers: str = ""

    @property
    def trusted_caller_set(self) -> frozenset[str]:
        return frozenset(c.strip() for c in self.trusted_callers.split(",") if c.strip())

    # Turn detection (Smart Turn v3, in-process ONNX). ON by default: the model
    # is downloaded once (revision-pinned) at startup and run on CPU. If it
    # cannot be loaded, startup logs the reason and falls back to the legacy
    # single-buffer 800ms silence path.
    # German verified offline at ~95% (synthetic test split); a real-call
    # smoke test on the live trunk is still recommended.
    turn_detection_enabled: bool = True
    # prob >= this => caller's turn is complete. Default 0.70 (not 0.50): on the
    # telephony 8 kHz aLaw path this lowers false cut-ins (~14.5%->10.6%) at a
    # small added-wait cost — fewer talk-overs is the right bias for a phone
    # agent. See docs/research/2026-06-14-smart-turn-german-accuracy.md.
    turn_complete_threshold: float = 0.70
    turn_vad_silence_ms: int = 200  # endpoint-candidate floor for the turn-end VAD
    turn_model_repo: str = "pipecat-ai/smart-turn-v3"
    turn_model_filename: str = "smart-turn-v3.2-cpu.onnx"
    turn_model_revision: str = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
    # Comma-separated onnxruntime execution providers. Default CPU; for the
    # NUC iGPU install onnxruntime-openvino and set
    # "OpenVINOExecutionProvider" + turn_model_filename=smart-turn-v3.2-gpu.onnx.
    turn_onnx_providers: str = "CPUExecutionProvider"

    @property
    def turn_onnx_provider_list(self) -> list[str]:
        return [p.strip() for p in self.turn_onnx_providers.split(",") if p.strip()]

    # Agent behaviour
    caller_id: str = "+49123456789"
    greeting_text: str = "Hallo, wie kann ich Ihnen helfen?"
    llm_system_prompt: str = (
        "Du bist ein hilfreicher Telefonassistent. "
        "Antworte immer auf Deutsch. Sei freundlich und präzise. "
        "Nutze rag_lookup für Wissensfragen und die Kalender-Werkzeuge für Terminanfragen."
    )
