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
FILLER_TEXT = "Einen Moment, ich schaue nach."
FALLBACK_RECOVERY = "Entschuldigung, da ist etwas schiefgelaufen."


class VoicePipeline:
    def __init__(self, stt, llm, tts, llm_stream=None, tts_stream=None) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._llm_stream = llm_stream
        self._tts_stream = tts_stream

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

    async def _tts_alaw_chunks(self, text):
        """Synthesize text and yield aLaw byte blobs (resampled 24k->8k)."""
        async for pcm_24k in self._tts_stream(text):
            yield alaw_encode(resample_24k_to_8k(pcm_24k))

    async def process_turn_stream(self, session: CallSession, pcm_16k: np.ndarray):
        """Stream a turn: STT (full) -> LLM tokens -> segments -> TTS -> aLaw.

        Yields aLaw byte blobs. Owns the PROCESSING entry; the caller drives
        SPEAKING (on first chunk) and LISTENING. On any mid-stream failure,
        yields the recovery prompt audio instead of raising.
        """
        from agent.segmenter import SentenceSegmenter

        session.transition(SessionState.PROCESSING)
        try:
            transcript = await self._stt(pcm_16k.tobytes())
        except Exception:
            log.exception("STT failed")
            async for c in self._tts_alaw_chunks(FALLBACK_ASR):
                yield c
            return

        if not transcript.strip():
            async for c in self._tts_alaw_chunks(FALLBACK_ASR):
                yield c
            return

        session.history.append({"role": "user", "content": transcript})

        seg = SentenceSegmenter()
        parts: list[str] = []
        tool_round = {"hit": False}

        def _on_tool_round() -> None:
            tool_round["hit"] = True

        try:
            async for token in self._llm_stream(
                session.history, session.caller_id, on_tool_round=_on_tool_round
            ):
                # Play filler once, the first time a tool round is signalled.
                if tool_round["hit"] and tool_round.get("filler_played") is not True:
                    tool_round["filler_played"] = True
                    async for c in self._tts_alaw_chunks(FILLER_TEXT):
                        yield c
                parts.append(token)
                for sentence in seg.feed(token):
                    async for c in self._tts_alaw_chunks(sentence):
                        yield c
            tail = seg.flush()
            if tail:
                async for c in self._tts_alaw_chunks(tail):
                    yield c
        except Exception:
            log.exception("LLM/TTS failed mid-stream")
            # Leave the user turn in history (the model saw it); do not append
            # a partial assistant turn. Emit the recovery prompt and stop.
            async for c in self._tts_alaw_chunks(FALLBACK_RECOVERY):
                yield c
            return

        session.history.append({"role": "assistant", "content": "".join(parts)})
