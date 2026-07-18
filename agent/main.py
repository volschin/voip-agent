import asyncio
import logging
import signal

import asyncpg
import httpx
import msal

from agent.config import Settings
from agent.conversation import ConversationManager
from agent.llm import LlmClient
from agent.pipeline import VoicePipeline
from agent.pjsip import PjsipClient
from agent.stt import SttClient
from agent.tools.calendar import MSGraphCalendar, UnavailableCalendar
from agent.tools.rag import RagTool
from agent.tts import TtsClient
from agent.turn_detector import TurnDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

log = logging.getLogger(__name__)


async def _rag_unavailable(_query: str) -> str:
    return "Die Wissensdatenbank ist derzeit nicht verfügbar."


async def main() -> None:
    s = Settings()

    pg_pool = None
    try:
        pg_pool = await asyncio.wait_for(
            asyncpg.create_pool(s.db_dsn, min_size=2, max_size=5),
            timeout=5,
        )
    except Exception:
        # Database-backed RAG must not prevent the phone from registering or
        # handling calls. Tool invocations degrade to an explicit unavailable
        # result while the voice path remains operational.
        log.warning("pgvector unavailable; RAG disabled")

    if all(
        (
            s.azure_tenant_id,
            s.azure_client_id,
            s.azure_client_secret,
            s.calendar_user_email,
        )
    ):
        msal_app = msal.ConfidentialClientApplication(
            client_id=s.azure_client_id,
            authority=f"https://login.microsoftonline.com/{s.azure_tenant_id}",
            client_credential=s.azure_client_secret,
        )
        calendar = MSGraphCalendar(msal_app=msal_app, user_email=s.calendar_user_email)
    else:
        log.info("Microsoft Graph is not configured; calendar tools disabled")
        calendar = UnavailableCalendar()

    # One HTTP client shared by every DGX service call, closed on shutdown
    # alongside the pg pool and the direct PJSIP client.
    http_client = httpx.AsyncClient()
    stt = SttClient(base_url=s.stt_base_url, client=http_client)
    tts = TtsClient(base_url=s.tts_base_url, client=http_client)
    # Build the detector (downloads the model) only when the feature is on;
    # otherwise pass None and ConversationManager runs the legacy silence path.
    turn_detector = None
    if s.turn_detection_enabled:
        try:
            turn_detector = TurnDetector(
                model_repo=s.turn_model_repo,
                model_filename=s.turn_model_filename,
                model_revision=s.turn_model_revision,
                providers=s.turn_onnx_provider_list,
                threshold=s.turn_complete_threshold,
            )
        except Exception as exc:
            log.warning("Smart Turn unavailable; using fixed-silence VAD: %s", exc)
    rag_lookup = (
        RagTool(pool=pg_pool, embedding_base_url=s.embedding_base_url, client=http_client).lookup
        if pg_pool is not None
        else _rag_unavailable
    )
    llm = LlmClient(
        base_url=s.llm_base_url,
        model=s.llm_model,
        system_prompt=s.llm_system_prompt,
        rag=rag_lookup,
        calendar=calendar,
        calendar_write_enabled=s.calendar_write_enabled,
        max_tool_rounds=s.max_tool_rounds,
        trusted_callers=s.trusted_caller_set,
        client=http_client,
    )
    pipeline = VoicePipeline(
        stt=stt.transcribe,
        llm=llm.complete,
        tts=tts.synthesize,
        llm_stream=llm.complete_stream,
        tts_stream=tts.synthesize_stream,
    )
    conversations = ConversationManager(
        settings=s,
        pipeline=pipeline,
        turn_detector=turn_detector,
    )
    pjsip = PjsipClient(settings=s, conversations=conversations)
    loop = asyncio.get_running_loop()
    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(shutdown_signal, pjsip.request_stop)

    try:
        await pjsip.run()
    finally:
        pjsip.request_stop()
        for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(shutdown_signal)
        await http_client.aclose()
        if pg_pool is not None:
            await pg_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
