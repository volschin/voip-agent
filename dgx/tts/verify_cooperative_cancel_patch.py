"""Build-time contract check for the patched faster-qwen3-tts runtime."""

from __future__ import annotations

import inspect
import threading
from unittest.mock import patch

import torch
from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.generate import _raise_if_cancelled, fast_generate


def main() -> None:
    assert "cancel_event" in inspect.signature(FasterQwen3TTS.generate_voice_clone).parameters
    assert "cancel_event" in inspect.signature(fast_generate).parameters

    cancelled = threading.Event()
    cancelled.set()
    with patch.object(torch.cuda, "synchronize") as synchronize:
        try:
            _raise_if_cancelled(cancelled)
        except InterruptedError:
            pass
        else:
            raise AssertionError("cancelled generation was not interrupted")
    synchronize.assert_called_once_with()


if __name__ == "__main__":
    main()
