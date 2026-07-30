"""Model adapter for one stable, named Qwen voice-clone profile."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from threading import Event, Lock
from time import monotonic
from typing import Any

import numpy as np

from dgx.tts.profiles import VoiceProfile, resolve_profile
from dgx.tts.runtime import normalize_language


class SynthesisAdmissionTimeout(RuntimeError):
    """The stable model stayed busy beyond the bounded admission wait."""


class SynthesisCancelled(RuntimeError):
    """A stable request was cancelled before or during model generation."""


class CloneRuntime:
    def __init__(
        self,
        model: Any,
        profiles: Mapping[str, VoiceProfile],
        default_profile_id: str,
    ) -> None:
        self._model = model
        self._profiles = profiles
        self._default_profile_id = default_profile_id
        self._model_lock = Lock()

    def warm(self) -> None:
        """Prepare and validate each clone prompt before health turns green."""
        with self._model_lock:
            for profile in self._profiles.values():
                audios, sample_rate = self._model.generate_voice_clone(
                    text="Bereit.",
                    language=profile.language,
                    ref_audio=profile.audio_path,
                    ref_text=profile.reference_text,
                    non_streaming_mode=True,
                )
                if not audios or np.asarray(audios[0]).size == 0:
                    raise RuntimeError("voice profile warmup returned empty audio")
                if sample_rate != 24_000:
                    raise RuntimeError("voice profile warmup must return 24 kHz audio")

    def synthesize(
        self,
        text: str,
        voice: str | None,
        language: str | None,
        *,
        cancel_event: Event | None = None,
        lock_timeout: float | None = None,
        admission_deadline: float | None = None,
    ) -> tuple[list[Any], int]:
        profile = resolve_profile(voice, self._profiles, self._default_profile_id)
        deadline = self._acquire_synthesis(cancel_event, lock_timeout, admission_deadline)
        try:
            if deadline is not None and monotonic() >= deadline:
                raise SynthesisAdmissionTimeout()
            try:
                return self._model.generate_voice_clone(
                    text=text,
                    language=normalize_language(language) or profile.language,
                    ref_audio=profile.audio_path,
                    ref_text=profile.reference_text,
                    non_streaming_mode=True,
                    cancel_event=cancel_event,
                )
            except InterruptedError:
                if cancel_event is not None and cancel_event.is_set():
                    raise SynthesisCancelled() from None
                raise
        finally:
            self._model_lock.release()

    def _acquire_synthesis(
        self,
        cancel_event: Event | None,
        lock_timeout: float | None,
        admission_deadline: float | None,
    ) -> float | None:
        deadline = None if lock_timeout is None else monotonic() + lock_timeout
        if admission_deadline is not None:
            deadline = admission_deadline if deadline is None else min(deadline, admission_deadline)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise SynthesisCancelled()
            wait_seconds = 0.05
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise SynthesisAdmissionTimeout()
                wait_seconds = min(wait_seconds, remaining)
            if not self._model_lock.acquire(timeout=wait_seconds):
                continue
            if cancel_event is not None and cancel_event.is_set():
                self._model_lock.release()
                raise SynthesisCancelled()
            if deadline is not None and monotonic() >= deadline:
                self._model_lock.release()
                raise SynthesisAdmissionTimeout()
            return deadline

    def stream(
        self,
        text: str,
        voice: str | None,
        language: str | None,
    ) -> Iterator[tuple[Any, int, dict]]:
        profile = resolve_profile(voice, self._profiles, self._default_profile_id)

        return _LockedModelStream(
            self._model_lock,
            lambda: self._model.generate_voice_clone_streaming(
                text=text,
                language=normalize_language(language) or profile.language,
                ref_audio=profile.audio_path,
                ref_text=profile.reference_text,
                chunk_size=8,
            ),
        )


class _LockedModelStream:
    """Hold the model lock across a stream and abort queued acquisition."""

    def __init__(
        self,
        model_lock: Any,
        factory: Callable[[], Iterator[tuple[Any, int, dict]]],
    ) -> None:
        self._model_lock = model_lock
        self._factory = factory
        self._cancelled = Event()
        self._upstream: Iterator[tuple[Any, int, dict]] | None = None
        self._owns_lock = False
        self._closed = False

    def __iter__(self) -> "_LockedModelStream":
        return self

    def __next__(self) -> tuple[Any, int, dict]:
        if self._closed:
            raise StopIteration
        if self._upstream is None:
            while not self._cancelled.is_set():
                if self._model_lock.acquire(timeout=0.05):
                    self._owns_lock = True
                    break
            if not self._owns_lock or self._cancelled.is_set():
                self.close()
                raise StopIteration
            self._upstream = iter(self._factory())
        if self._cancelled.is_set():
            self.close()
            raise StopIteration
        try:
            return next(self._upstream)
        except BaseException:
            self.close()
            raise

    def cancel(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        upstream = self._upstream
        self._upstream = None
        try:
            close = getattr(upstream, "close", None)
            if close is not None:
                close()
        finally:
            if self._owns_lock:
                self._owns_lock = False
                self._model_lock.release()
