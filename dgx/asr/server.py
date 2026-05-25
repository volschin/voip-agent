import io
import os
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI(title="Qwen3-ASR")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True
).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID)
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=DTYPE,
    device=DEVICE,
)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default="de"),
):
    audio_bytes = await file.read()
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
    result = asr_pipe(
        {"array": audio_array.astype(np.float32), "sampling_rate": sample_rate},
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    return JSONResponse({"text": result["text"]})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
