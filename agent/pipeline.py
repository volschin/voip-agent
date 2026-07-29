import asyncio
import io
import logging
import wave
from contextlib import suppress

import numpy as np

from agent.audio import resample_pcm16
from agent.session import CallSession, SessionState

log = logging.getLogger(__name__)

_TTS_SAMPLE_RATE = 24000
_OUTPUT_SAMPLE_RATE = 16000
_SENTENCE_PREFETCH_DEPTH = 2


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

    async def synthesize_pcm16(self, text: str) -> bytes:
        try:
            wav_bytes = await self._tts(text)
            pcm_24k = _decode_wav(wav_bytes)
            return resample_pcm16(pcm_24k, _TTS_SAMPLE_RATE, _OUTPUT_SAMPLE_RATE)
        except Exception:
            log.exception("TTS failed for text: %r", text[:50])
            return b""

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
            return await self.synthesize_pcm16(FALLBACK_ASR)

        if not transcript.strip():
            return await self.synthesize_pcm16(FALLBACK_ASR)

        session.history.append({"role": "user", "content": transcript})

        try:
            response_text = await self._llm(session.history, session.caller_id)
        except Exception:
            log.exception("LLM failed")
            session.history.pop()
            return await self.synthesize_pcm16(FALLBACK_LLM)

        session.history.append({"role": "assistant", "content": response_text})
        return await self.synthesize_pcm16(response_text)

    async def _tts_pcm16_chunks(self, text):
        """Synthesize one stable sentence and yield 16 kHz mono PCM16."""
        pcm_16k = await self.synthesize_pcm16(text)
        if pcm_16k:
            yield pcm_16k

    async def process_turn_stream(self, session: CallSession, pcm_16k: np.ndarray):
        """Stream a turn: STT (full) -> LLM tokens -> segments -> 16 kHz PCM.

        Yields PCM16 byte blobs. Owns the PROCESSING entry; the caller drives
        SPEAKING (on first chunk) and LISTENING. On any mid-stream failure,
        yields the recovery prompt audio instead of raising.
        """
        from agent.segmenter import SentenceSegmenter

        session.transition(SessionState.PROCESSING)
        try:
            transcript = await self._stt(pcm_16k.tobytes())
        except Exception:
            log.exception("STT failed")
            async for c in self._tts_pcm16_chunks(FALLBACK_ASR):
                yield c
            return

        if not transcript.strip():
            async for c in self._tts_pcm16_chunks(FALLBACK_ASR):
                yield c
            return

        session.history.append({"role": "user", "content": transcript})

        parts: list[str] = []

        # Segment in the LLM producer so at most two complete sentences can
        # wait while the sole consumer streams the current TTS request.
        queue: asyncio.Queue = asyncio.Queue(maxsize=_SENTENCE_PREFETCH_DEPTH)
        filler_task: asyncio.Task | None = None
        filler_scheduled = False

        def _on_tool_round() -> None:
            nonlocal filler_scheduled, filler_task
            if filler_scheduled:
                return
            filler_scheduled = True
            filler_task = asyncio.create_task(queue.put(("text", FILLER_TEXT)))

        async def _produce() -> None:
            seg = SentenceSegmenter()
            try:
                async for token in self._llm_stream(
                    session.history, session.caller_id, on_tool_round=_on_tool_round
                ):
                    if filler_task is not None and not filler_task.done():
                        await asyncio.sleep(0)
                    parts.append(token)
                    for sentence in seg.feed(token):
                        await queue.put(("text", sentence))
                if filler_task is not None and not filler_task.done():
                    await asyncio.sleep(0)
                tail = seg.flush()
                if tail:
                    await queue.put(("text", tail))
                await queue.put(("done", None))
            except Exception:
                log.exception("LLM stream failed")
                await queue.put(("error", None))

        producer = asyncio.create_task(_produce())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "error":
                    raise RuntimeError("llm stream failed")
                if kind == "done":
                    break
                emitted = False
                async for c in self._tts_pcm16_chunks(payload):
                    emitted = True
                    yield c
                if not emitted:
                    raise RuntimeError("TTS returned no audio")
        except Exception:
            log.exception("LLM/TTS failed mid-stream")
            # Leave the user turn in history (the model saw it); do not append
            # a partial assistant turn. Emit the recovery prompt and stop.
            async for c in self._tts_pcm16_chunks(FALLBACK_RECOVERY):
                yield c
            return
        finally:
            if not producer.done():
                producer.cancel()
            if filler_task is not None and not filler_task.done():
                filler_task.cancel()
            with suppress(asyncio.CancelledError):
                await producer
            if filler_task is not None:
                with suppress(asyncio.CancelledError):
                    await filler_task

        session.history.append({"role": "assistant", "content": "".join(parts)})
