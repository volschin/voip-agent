"""Model adapter for one stable, named Qwen voice-clone profile."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from dgx.tts.profiles import VoiceProfile, resolve_profile
from dgx.tts.runtime import normalize_language


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

    def warm(self) -> None:
        """Prepare and validate each clone prompt before health turns green."""
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
    ) -> tuple[list[Any], int]:
        profile = resolve_profile(voice, self._profiles, self._default_profile_id)
        return self._model.generate_voice_clone(
            text=text,
            language=normalize_language(language) or profile.language,
            ref_audio=profile.audio_path,
            ref_text=profile.reference_text,
            non_streaming_mode=True,
        )

    def stream(
        self,
        text: str,
        voice: str | None,
        language: str | None,
    ) -> Iterator[tuple[Any, int, dict]]:
        profile = resolve_profile(voice, self._profiles, self._default_profile_id)
        return self._model.generate_voice_clone_streaming(
            text=text,
            language=normalize_language(language) or profile.language,
            ref_audio=profile.audio_path,
            ref_text=profile.reference_text,
            chunk_size=8,
        )
