import logging

import numpy as np

from agent.audio import alaw_encode, resample_24k_to_8k
from agent.session import CallSession, SessionState

log = logging.getLogger(__name__)

_SILENCE_FRAME = b"\xd5" * 160

FALLBACK_ASR = "Ich habe Sie leider nicht verstanden."
FALLBACK_LLM = "Technischer Fehler, bitte später erneut anrufen."


class VoicePipeline:
    def __init__(self, stt, llm, tts) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts

    async def synthesize_alaw(self, text: str) -> bytes:
        try:
            pcm_24k_bytes = await self._tts(text)
            pcm_24k = np.frombuffer(pcm_24k_bytes, dtype=np.int16)
            return alaw_encode(resample_24k_to_8k(pcm_24k))
        except Exception:
            log.exception("TTS failed for text: %r", text[:50])
            return _SILENCE_FRAME

    async def process_turn(self, session: CallSession, pcm_16k: np.ndarray) -> bytes:
        session.transition(SessionState.PROCESSING)

        try:
            transcript = await self._stt(pcm_16k.tobytes())
        except Exception:
            log.exception("STT failed")
            session.transition(SessionState.LISTENING)
            return await self.synthesize_alaw(FALLBACK_ASR)

        if not transcript.strip():
            session.transition(SessionState.LISTENING)
            return await self.synthesize_alaw(FALLBACK_ASR)

        session.history.append({"role": "user", "content": transcript})

        try:
            response_text = await self._llm(session.history)
        except Exception:
            log.exception("LLM failed")
            session.history.pop()
            session.transition(SessionState.LISTENING)
            return await self.synthesize_alaw(FALLBACK_LLM)

        session.history.append({"role": "assistant", "content": response_text})
        session.transition(SessionState.SPEAKING)

        alaw_bytes = await self.synthesize_alaw(response_text)
        session.transition(SessionState.LISTENING)
        return alaw_bytes
