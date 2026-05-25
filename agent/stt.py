import io
import wave

import httpx


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class SttClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def transcribe(self, pcm_16k: bytes) -> str:
        wav = _pcm_to_wav(pcm_16k)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"language": "de"},
                timeout=30.0,
            )
        resp.raise_for_status()
        return resp.json()["text"]
