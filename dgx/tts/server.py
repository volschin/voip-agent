"""FastAPI wrapper exposing OpenAI-compatible /v1/audio/speech endpoint
backed by Qwen3-TTS-12Hz-1.7B-VoiceDesign.

OpenAI body field mapping:
  - input        -> text to synthesize
  - voice        -> qwen-tts `instruct` (voice description, free-form natural language)
  - response_format -> wav, flac, or mp3 (mp3 needs system libmp3lame)
"""
import io
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch
import soundfile as sf

MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
TOKENIZER_ID = os.environ.get("QWEN_TTS_TOKENIZER", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
DEFAULT_INSTRUCT = os.environ.get(
    "QWEN_TTS_DEFAULT_VOICE",
    "A neutral, friendly adult voice with clear pronunciation, moderate pace, and natural intonation.",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# Honor explicit override; otherwise auto-detect flash-attn availability.
try:
    import flash_attn  # noqa: F401
    _FA_AVAILABLE = True
except Exception:
    _FA_AVAILABLE = False
ATTN_IMPL = os.environ.get(
    "QWEN_TTS_ATTN_IMPL",
    "flash_attention_2" if (DEVICE == "cuda" and _FA_AVAILABLE) else ("sdpa" if DEVICE == "cuda" else "eager"),
)

_model = None
_tokenizer = None


def _load() -> None:
    global _model, _tokenizer
    if _model is not None:
        return
    from faster_qwen3_tts import FasterQwen3TTS
    print(f"[load] faster-qwen3-tts model {MODEL_ID} device={DEVICE}", flush=True)
    _model = FasterQwen3TTS.from_pretrained(MODEL_ID, device=DEVICE, dtype=DTYPE)
    _tokenizer = None  # faster runtime manages tokenization internally
    print("[load] OK", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="Qwen3-TTS Server", version="0.1.0", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str = Field(default="qwen3-tts", description="Ignored — single served model")
    input: str = Field(..., description="Text to synthesize")
    voice: str | None = Field(
        default=None,
        description="Voice description (free-form natural language). Maps to qwen-tts `instruct`. "
                    "If omitted, uses QWEN_TTS_DEFAULT_VOICE env default.",
    )
    response_format: str = Field(default="wav", description="wav | flac | mp3")
    language: str | None = Field(
        default=None,
        description="Optional language hint. One of: zh, en, ja, ko, de, fr, ru, pt, es, it. "
                    "If None the model auto-detects.",
    )


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "qwen3-tts",
            "object": "model",
            "owned_by": "qwen",
            "permission": [],
        }],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/v1/audio/speech")
def synthesize(req: SpeechRequest):
    if _model is None:
        _load()
    instruct = req.voice or DEFAULT_INSTRUCT
    try:
        audios, sample_rate = _model.generate_voice_design(
            text=req.input,
            instruct=instruct,
            language=req.language,
            non_streaming_mode=True,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"synthesis failed: {type(e).__name__}: {e}")

    if not audios:
        raise HTTPException(500, "synthesis returned empty result")
    audio = audios[0]
    fmt = req.response_format.lower()
    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, audio, sample_rate, format="WAV")
        media = "audio/wav"
    elif fmt == "flac":
        sf.write(buf, audio, sample_rate, format="FLAC")
        media = "audio/flac"
    elif fmt == "mp3":
        try:
            sf.write(buf, audio, sample_rate, format="MP3")
            media = "audio/mpeg"
        except Exception:
            raise HTTPException(400, "mp3 encoding requires system libmp3lame; use wav or flac")
    else:
        raise HTTPException(400, f"unsupported response_format: {fmt}")
    buf.seek(0)
    return Response(content=buf.read(), media_type=media)


@app.post("/v1/audio/speech/stream")
def synthesize_stream(req: SpeechRequest):
    if _model is None:
        _load()
    instruct = req.voice or DEFAULT_INSTRUCT

    def gen():
        import numpy as np
        for audio_chunk, sample_rate, _timing in _model.generate_voice_design_streaming(
            text=req.input,
            instruct=instruct,
            language=req.language,
            chunk_size=8,  # ~667ms audio per chunk; tune later
        ):
            arr = np.asarray(audio_chunk)
            if np.issubdtype(arr.dtype, np.floating):
                # Normalized float [-1, 1] → int16 PCM (mirror soundfile's WAV scaling).
                # Direct astype("<i2") would truncate every in-range sample to 0 → silence.
                arr = np.clip(arr, -1.0, 1.0) * 32767.0
            yield arr.astype("<i2").tobytes()

    return StreamingResponse(gen(), media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
