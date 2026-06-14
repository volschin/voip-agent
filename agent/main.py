import asyncio
import logging

import asyncpg
import httpx
import msal

from agent.ari import AriClient
from agent.config import Settings
from agent.llm import LlmClient
from agent.pipeline import VoicePipeline
from agent.stt import SttClient
from agent.tools.calendar import MSGraphCalendar
from agent.tools.rag import RagTool
from agent.tts import TtsClient
from agent.turn_detector import TurnDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


async def main() -> None:
    s = Settings()

    pg_pool = await asyncpg.create_pool(s.db_dsn, min_size=2, max_size=5)

    msal_app = msal.ConfidentialClientApplication(
        client_id=s.azure_client_id,
        authority=f"https://login.microsoftonline.com/{s.azure_tenant_id}",
        client_credential=s.azure_client_secret,
    )

    # One HTTP client shared by every DGX service call, closed on shutdown
    # alongside the pg pool and the ARI client.
    http_client = httpx.AsyncClient()
    stt = SttClient(base_url=s.stt_base_url, client=http_client)
    tts = TtsClient(base_url=s.tts_base_url, client=http_client)
    # Build the detector (downloads the model) only when the feature is on;
    # otherwise pass None and AriClient runs the legacy silence path.
    turn_detector = (
        TurnDetector(
            model_repo=s.turn_model_repo,
            model_filename=s.turn_model_filename,
            model_revision=s.turn_model_revision,
            providers=s.turn_onnx_provider_list,
            threshold=s.turn_complete_threshold,
        )
        if s.turn_detection_enabled
        else None
    )
    rag = RagTool(pool=pg_pool, embedding_base_url=s.embedding_base_url, client=http_client)
    calendar = MSGraphCalendar(msal_app=msal_app, user_email=s.calendar_user_email)
    llm = LlmClient(
        base_url=s.llm_base_url,
        model=s.llm_model,
        system_prompt=s.llm_system_prompt,
        rag=rag.lookup,
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
    ari = AriClient(settings=s, pipeline=pipeline, turn_detector=turn_detector)

    try:
        await ari.run()
    finally:
        await ari.aclose()
        await http_client.aclose()
        await pg_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
