import asyncio
import logging

import asyncpg
import msal

from agent.config import Settings
from agent.ari import AriClient
from agent.pipeline import VoicePipeline
from agent.stt import SttClient
from agent.tts import TtsClient
from agent.llm import LlmClient
from agent.tools.rag import RagTool
from agent.tools.calendar import MSGraphCalendar

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

    stt = SttClient(base_url=s.stt_base_url)
    tts = TtsClient(base_url=s.tts_base_url)
    rag = RagTool(pool=pg_pool, embedding_base_url=s.embedding_base_url)
    calendar = MSGraphCalendar(msal_app=msal_app, user_email=s.calendar_user_email)
    llm = LlmClient(
        base_url=s.llm_base_url,
        model=s.llm_model,
        system_prompt=s.llm_system_prompt,
        rag=rag.lookup,
        calendar=calendar,
    )
    pipeline = VoicePipeline(stt=stt.transcribe, llm=llm.complete, tts=tts.synthesize)
    ari = AriClient(settings=s, pipeline=pipeline)

    await ari.run()


if __name__ == "__main__":
    asyncio.run(main())
