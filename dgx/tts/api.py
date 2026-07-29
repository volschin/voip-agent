"""Testable HTTP contract for the private Qwen voice-clone service."""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dgx.tts.profiles import ProfileError


@dataclass(frozen=True)
class HealthMetadata:
    model_revision: str
    default_profile: str
    profiles_loaded: tuple[str, ...]
    device: str


class SpeechRequest(BaseModel):
    model: str = Field(default="qwen3-tts", description="Ignored; one served model")
    input: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str | None = Field(default=None, description="Server-owned voice profile ID")
    response_format: str = Field(default="wav", description="wav | flac | mp3")
    language: str | None = Field(default=None, description="Optional language hint")


def create_app(runtime: Any, health: HealthMetadata) -> FastAPI:
    app = FastAPI(title="Qwen3-TTS Clone Server", version="0.2.0")

    @app.get("/health")
    def get_health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": True,
            "model_revision": health.model_revision,
            "default_profile": health.default_profile,
            "profiles_loaded": list(health.profiles_loaded),
            "device": health.device,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "qwen3-tts",
                    "object": "model",
                    "owned_by": "qwen",
                    "permission": [],
                }
            ],
        }

    @app.post("/v1/audio/speech")
    def synthesize(request: SpeechRequest) -> Response:
        try:
            audios, sample_rate = runtime.synthesize(
                request.input,
                request.voice,
                request.language,
            )
            if not audios or np.asarray(audios[0]).size == 0:
                raise RuntimeError("empty audio")
            return _audio_response(
                np.asarray(audios[0]),
                sample_rate,
                request.response_format,
            )
        except ProfileError as error:
            raise HTTPException(422, detail="unsupported voice profile") from error
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(500, detail="synthesis failed") from None

    @app.post("/v1/audio/speech/stream")
    def synthesize_stream(request: SpeechRequest) -> StreamingResponse:
        try:
            chunks = runtime.stream(request.input, request.voice, request.language)
        except ProfileError as error:
            raise HTTPException(422, detail="unsupported voice profile") from error
        except Exception:
            raise HTTPException(500, detail="synthesis failed") from None
        return StreamingResponse(
            encode_pcm_stream(chunks),
            media_type="application/octet-stream",
        )

    return app


async def encode_pcm_stream(
    chunks: Iterator[tuple[Any, int, dict]],
) -> AsyncIterator[bytes]:
    """Bridge blocking model chunks to async PCM and always close upstream."""
    iterator = iter(chunks)
    try:
        while True:
            next_task = asyncio.create_task(asyncio.to_thread(_next_chunk, iterator))
            try:
                item = await next_task
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(next_task)
                except BaseException:
                    pass
                raise
            if item is None:
                break
            audio_chunk, sample_rate, _timing = item
            if sample_rate != 24_000:
                raise RuntimeError("stream must return 24 kHz audio")
            array = np.asarray(audio_chunk)
            if array.size == 0:
                continue
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, -1.0, 1.0) * 32767.0
            yield array.astype("<i2").tobytes()
    finally:
        await asyncio.shield(asyncio.to_thread(_close_iterator, iterator))


def _next_chunk(
    iterator: Iterator[tuple[Any, int, dict]],
) -> tuple[Any, int, dict] | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _close_iterator(iterator: Iterator[tuple[Any, int, dict]]) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def _audio_response(audio: np.ndarray, sample_rate: int, response_format: str) -> Response:
    if sample_rate != 24_000:
        raise RuntimeError("synthesis must return 24 kHz audio")
    formats = {
        "wav": ("WAV", "audio/wav"),
        "flac": ("FLAC", "audio/flac"),
        "mp3": ("MP3", "audio/mpeg"),
    }
    selected = formats.get(response_format.lower())
    if selected is None:
        raise HTTPException(400, detail="unsupported response_format")
    output = io.BytesIO()
    try:
        sf.write(output, audio, sample_rate, format=selected[0], subtype="PCM_16")
    except Exception:
        if response_format.lower() == "mp3":
            raise HTTPException(400, detail="mp3 encoding unavailable") from None
        raise
    return Response(content=output.getvalue(), media_type=selected[1])
