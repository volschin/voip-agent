import io
import logging
import wave

import numpy as np

from agent.audio import alaw_encode, resample_24k_to_8k
from agent.session import CallSession, SessionState

log = logging.getLogger(__name__)

_SILENCE_FRAME = b"\xd5" * 160
_TTS_SAMPLE_RATE = 24000


def _decode_wav(data: bytes) -> np.ndarray:
    """Decode the WAV returned by the TTS server into mono int16 samples.

    The server (dgx/tts/server.py) returns a 24 kHz PCM_16 WAV. Treating the
    raw bytes as int16 — as the old code did — fed the 44-byte RIFF header into
    the resampler as audio. Parse the container instead.
    """
    with wave.open(io.BytesIO(data), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {wf.getsampwidth() * 8}-bit")
        if wf.getframerate() != _TTS_SAMPLE_RATE:
            log.warning(
                "TTS WAV is %d Hz, resampler expects %d Hz",
                wf.getframerate(),
                _TTS_SAMPLE_RATE,
            )
        frames = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16)
        if wf.getnchannels() > 1:  # collapse to mono
            pcm = pcm.reshape(-1, wf.getnchannels())[:, 0].copy()
    return pcm

FALLBACK_ASR = "Ich habe Sie leider nicht verstanden."
FALLBACK_LLM = "Technischer Fehler, bitte später erneut anrufen."


class VoicePipeline:
    def __init__(self, stt, llm, tts) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts

    async def synthesize_alaw(self, text: str) -> bytes:
        try:
            wav_bytes = await self._tts(text)
            pcm_24k = _decode_wav(wav_bytes)
            return alaw_encode(resample_24k_to_8k(pcm_24k))
        except Exception:
            log.exception("TTS failed for text: %r", text[:50])
            return _SILENCE_FRAME

    async def process_turn(self, session: CallSession, pcm_16k: np.ndarray) -> bytes:
        # Owns the PROCESSING state only. Returns synthesized audio and leaves
        # the session in PROCESSING; the caller (AriClient._play_audio) drives
        # PROCESSING -> SPEAKING -> LISTENING around actual playback. This keeps
        # SPEAKING reachable only from PROCESSING/ANSWER, as the FSM enforces.
        session.transition(SessionState.PROCESSING)

        try:
            transcript = await self._stt(pcm_16k.tobytes())
        except Exception:
            log.exception("STT failed")
            return await self.synthesize_alaw(FALLBACK_ASR)

        if not transcript.strip():
            return await self.synthesize_alaw(FALLBACK_ASR)

        session.history.append({"role": "user", "content": transcript})

        try:
            response_text = await self._llm(session.history, session.caller_id)
        except Exception:
            log.exception("LLM failed")
            session.history.pop()
            return await self.synthesize_alaw(FALLBACK_LLM)

        session.history.append({"role": "assistant", "content": response_text})
        return await self.synthesize_alaw(response_text)
