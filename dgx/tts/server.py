import io
import os
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModel

MODEL_ID = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI(title="Qwen3-TTS")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
).to(DEVICE)


class SpeechRequest(BaseModel):
    model: str = MODEL_ID
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/audio/speech")
async def synthesize(req: SpeechRequest):
    inputs = processor(text=req.input, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs)
    audio = output.squeeze().float().cpu().numpy()

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=24000, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
