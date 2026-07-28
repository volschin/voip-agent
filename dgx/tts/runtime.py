"""Fail-closed runtime gate for GX10 Qwen3-TTS."""

from typing import Any

LANGUAGE_NAMES = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}


def normalize_language(language: str | None) -> str | None:
    """Map OpenAI-style language codes to faster-qwen3-tts names."""
    if language is None:
        return None
    return LANGUAGE_NAMES.get(language.lower(), language.lower())


def require_gb10_cuda(torch_module: Any) -> str:
    """Return the only supported device after verifying the live GPU."""
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-TTS")
    device_name = torch_module.cuda.get_device_name(0)
    if "NVIDIA GB10" not in device_name:
        raise RuntimeError("Qwen3-TTS requires NVIDIA GB10")
    return "cuda"
