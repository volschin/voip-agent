"""Testable HTTP contract for the private Qwen voice-clone service."""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from dataclasses import dataclass
from threading import Event, Lock
from time import monotonic
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dgx.tts.clone_runtime import SynthesisAdmissionTimeout
from dgx.tts.profiles import ProfileError

MAX_SPEECH_INPUT_CHARACTERS = 2_000
DEFAULT_SYNTHESIS_ADMISSION_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class HealthMetadata:
    model_revision: str
    default_profile: str
    profiles_loaded: tuple[str, ...]
    device: str


class SpeechRequest(BaseModel):
    model: str = Field(default="qwen3-tts", description="Ignored; one served model")
    input: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SPEECH_INPUT_CHARACTERS,
        description="Text to synthesize",
    )
    voice: str | None = Field(default=None, description="Server-owned voice profile ID")
    response_format: str = Field(default="wav", description="wav | flac | mp3")
    language: str | None = Field(default=None, description="Optional language hint")


def create_app(
    runtime: Any,
    health: HealthMetadata,
    *,
    synthesis_admission_timeout_seconds: float = DEFAULT_SYNTHESIS_ADMISSION_TIMEOUT_SECONDS,
) -> FastAPI:
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
    async def synthesize(speech: SpeechRequest, request: Request) -> Response:
        try:
            audios, sample_rate = await _run_stable_synthesis(
                runtime,
                speech,
                request,
                synthesis_admission_timeout_seconds,
            )
            if not audios or np.asarray(audios[0]).size == 0:
                raise RuntimeError("empty audio")
            return _audio_response(
                np.asarray(audios[0]),
                sample_rate,
                speech.response_format,
            )
        except ProfileError as error:
            raise HTTPException(422, detail="unsupported voice profile") from error
        except SynthesisAdmissionTimeout as error:
            raise HTTPException(503, detail="synthesis admission timed out") from error
        except _ClientDisconnected:
            return Response(status_code=499)
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


class _ClientDisconnected(RuntimeError):
    pass


async def _run_stable_synthesis(
    runtime: Any,
    speech: SpeechRequest,
    request: Request,
    admission_timeout_seconds: float,
) -> tuple[list[Any], int]:
    deadline = monotonic() + admission_timeout_seconds
    cancelled = Event()
    operation = asyncio.create_task(
        asyncio.to_thread(
            runtime.synthesize,
            speech.input,
            speech.voice,
            speech.language,
            cancel_event=cancelled,
            lock_timeout=admission_timeout_seconds,
            admission_deadline=deadline,
        )
    )
    try:
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                await _cancel_stable_operation(operation, cancelled)
                raise SynthesisAdmissionTimeout()
            done, _pending = await asyncio.wait(
                {operation},
                timeout=min(0.025, remaining),
            )
            if operation in done:
                result = operation.result()
                if monotonic() >= deadline:
                    await _cancel_stable_operation(operation, cancelled)
                    raise SynthesisAdmissionTimeout()
                return result
            if await request.is_disconnected():
                await _cancel_stable_operation(operation, cancelled)
                raise _ClientDisconnected()
    except asyncio.CancelledError:
        await _cancel_stable_operation(operation, cancelled)
        raise


async def _cancel_stable_operation(operation: asyncio.Task, cancelled: Event) -> None:
    cancelled.set()
    operation.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await operation


async def encode_pcm_stream(
    chunks: Iterator[tuple[Any, int, dict]],
) -> AsyncIterator[bytes]:
    """Bridge blocking model chunks to async PCM and always close upstream."""
    iterator = _SerializedIterator(iter(chunks))
    try:
        while True:
            next_task = asyncio.create_task(asyncio.to_thread(_next_chunk, iterator))
            try:
                item = await asyncio.shield(next_task)
            except asyncio.CancelledError:
                _cancel_iterator(iterator)
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


def _cancel_iterator(iterator: Iterator[tuple[Any, int, dict]]) -> None:
    cancel = getattr(iterator, "cancel", None)
    if cancel is not None:
        cancel()


class _SerializedIterator:
    """Never close a blocking iterator while its ``next`` call is executing."""

    def __init__(self, iterator: Iterator[tuple[Any, int, dict]]) -> None:
        self._iterator = iterator
        self._operation_lock = Lock()

    def __iter__(self) -> "_SerializedIterator":
        return self

    def __next__(self) -> tuple[Any, int, dict]:
        with self._operation_lock:
            return next(self._iterator)

    def cancel(self) -> None:
        cancel = getattr(self._iterator, "cancel", None)
        if cancel is not None:
            cancel()

    def close(self) -> None:
        with self._operation_lock:
            close = getattr(self._iterator, "close", None)
            if close is not None:
                close()


def _audio_response(audio: np.ndarray, sample_rate: int, response_format: str) -> Response:
    if sample_rate != 24_000:
        raise RuntimeError("synthesis must return 24 kHz audio")
    audio = _validate_stable_audio(audio)
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


def _validate_stable_audio(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim != 1:
        raise RuntimeError("synthesis must return mono audio")
    if array.size == 0:
        raise RuntimeError("synthesis returned empty audio")
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.signedinteger)
        or np.issubdtype(array.dtype, np.unsignedinteger)
    ):
        raise RuntimeError("synthesis returned unsupported audio samples")
    if not np.all(np.isfinite(array)):
        raise RuntimeError("synthesis returned non-finite audio")
    if np.issubdtype(array.dtype, np.floating):
        if np.any(array < -1.0) or np.any(array > 1.0):
            raise RuntimeError("synthesis returned out-of-range audio")
        return array
    if np.any(array < -32_768) or np.any(array > 32_767):
        raise RuntimeError("synthesis returned out-of-range audio")
    return array.astype(np.int16, copy=False)
