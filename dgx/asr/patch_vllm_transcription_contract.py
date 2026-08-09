"""Keep the inherited vLLM transcription JSON compatible with the voice client."""

import importlib.util
import sys
from pathlib import Path

_FIELD = "    usage: TranscriptionUsageAudio\n"
_EXCLUDED_FIELD = "    usage: TranscriptionUsageAudio = Field(exclude=True)\n"
_MODULE = "vllm.entrypoints.speech_to_text.transcription.protocol"
_CLASS = "class TranscriptionResponse("


def _protocol_path() -> Path:
    spec = importlib.util.find_spec(_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {_MODULE}")
    return Path(spec.origin)


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} [protocol.py]")
    protocol = Path(sys.argv[1]) if len(sys.argv) == 2 else _protocol_path()
    source = protocol.read_text(encoding="utf-8")
    if source.count(_CLASS) != 1:
        raise RuntimeError("expected exactly one inherited TranscriptionResponse class")
    class_start = source.index(_CLASS)
    class_end = source.find("\n\nclass ", class_start)
    if class_end == -1:
        class_end = len(source)
    response_class = source[class_start:class_end]
    if response_class.count(_FIELD) != 1:
        raise RuntimeError("expected one usage field in inherited TranscriptionResponse")
    patched_class = response_class.replace(_FIELD, _EXCLUDED_FIELD)
    protocol.write_text(
        source[:class_start] + patched_class + source[class_end:],
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
