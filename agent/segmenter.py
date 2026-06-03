"""Buffer an LLM token stream into clause/sentence units for TTS.

Feeding raw tokens to TTS produces robotic prosody — the model needs
coherent units. Emit on sentence-final punctuation, but never split on a
period that belongs to a known German abbreviation.
"""

_SENTENCE_END = {".", "!", "?"}
# Abbreviations whose trailing period must not end a sentence.
_ABBREVIATIONS = {
    "z.b.",
    "u.a.",
    "d.h.",
    "u.s.w.",
    "usw.",
    "etc.",
    "bzw.",
    "ca.",
    "nr.",
    "abs.",
    "vgl.",
    "z.t.",
    "evtl.",
    "ggf.",
    "inkl.",
    "max.",
    "min.",
}


class SentenceSegmenter:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """Append a token; return any completed sentences (possibly empty)."""
        self._buf += token
        out: list[str] = []
        while True:
            idx = self._next_boundary(self._buf)
            if idx is None:
                break
            sentence = self._buf[: idx + 1].strip()
            self._buf = self._buf[idx + 1 :]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        """Return any buffered trailing text (no terminal punctuation)."""
        tail = self._buf.strip()
        self._buf = ""
        return tail or None

    def _next_boundary(self, text: str) -> int | None:
        for i, ch in enumerate(text):
            if ch in _SENTENCE_END:
                # The char after the boundary must exist and not be a digit
                # (avoid splitting "3.14"); require it to be whitespace/end.
                if i + 1 < len(text) and not text[i + 1].isspace():
                    continue
                if self._ends_with_abbreviation(text[: i + 1]):
                    continue
                return i
        return None

    @staticmethod
    def _ends_with_abbreviation(text: str) -> bool:
        last = text.strip().split()[-1].lower() if text.strip() else ""
        return last in _ABBREVIATIONS
