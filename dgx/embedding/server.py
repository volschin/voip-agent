import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

app = FastAPI(title="Embedding")
model = SentenceTransformer(MODEL_ID)


class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str = MODEL_ID


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/embeddings")
async def embed(req: EmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = model.encode(texts, normalize_embeddings=True).tolist()
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
        "model": MODEL_ID,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
