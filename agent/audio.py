import audioop

import numpy as np
import webrtcvad
from scipy.signal import resample_poly


def alaw_decode(data: bytes) -> np.ndarray:
    return np.frombuffer(audioop.alaw2lin(data, 2), dtype=np.int16)


def alaw_encode(samples: np.ndarray) -> bytes:
    return audioop.lin2alaw(samples.astype(np.int16).tobytes(), 2)


def resample_8k_to_16k(samples: np.ndarray) -> np.ndarray:
    return resample_poly(samples, up=2, down=1).astype(np.int16)


def resample_24k_to_8k(samples: np.ndarray) -> np.ndarray:
    return resample_poly(samples, up=1, down=3).astype(np.int16)


class VadBuffer:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        silence_threshold_ms: int = 800,
        max_speech_ms: int = 15000,
    ) -> None:
        self._vad = webrtcvad.Vad(2)
        self._sample_rate = sample_rate
        self._silence_threshold = silence_threshold_ms // frame_ms
        self._max_speech_frames = max_speech_ms // frame_ms
        self._speech_frames: list[np.ndarray] = []
        self._silence_count = 0
        self._in_speech = False

    def add_frame(self, frame: np.ndarray) -> np.ndarray | None:
        is_speech = self._vad.is_speech(frame.astype(np.int16).tobytes(), self._sample_rate)
        if is_speech:
            self._speech_frames.append(frame)
            self._silence_count = 0
            self._in_speech = True
        elif self._in_speech:
            self._speech_frames.append(frame)
            self._silence_count += 1
            if self._silence_count >= self._silence_threshold:
                return self._flush()
        if len(self._speech_frames) >= self._max_speech_frames:
            return self._flush()
        return None

    @property
    def at_cap(self) -> bool:
        """True when buffered frames have reached the hard max_speech_frames cap."""
        return len(self._speech_frames) >= self._max_speech_frames

    def add_frame_candidate(self, frame: np.ndarray) -> np.ndarray | None:
        # Like add_frame, but returns a COPY of the buffered speech without
        # resetting, so the caller (a turn detector) can decide whether the
        # turn is really over. Returns the candidate on silence floor OR hard
        # cap; the cap guarantees the continue_speech loop terminates.
        is_speech = self._vad.is_speech(frame.astype(np.int16).tobytes(), self._sample_rate)
        if is_speech:
            self._speech_frames.append(frame)
            self._silence_count = 0
            self._in_speech = True
        elif self._in_speech:
            self._speech_frames.append(frame)
            self._silence_count += 1
        # Pre-speech silence: nothing to propose yet.
        if not self._in_speech:
            return None
        if self._silence_count >= self._silence_threshold or self.at_cap:
            return np.concatenate(self._speech_frames)
        return None

    def continue_speech(self) -> None:
        # Keep listening after an "incomplete" verdict: drop the silence count
        # but retain buffered frames so a later pause re-proposes a candidate.
        self._silence_count = 0

    def force_flush(self) -> np.ndarray | None:
        if self._speech_frames:
            return self._flush()
        return None

    def reset(self) -> None:
        self._speech_frames = []
        self._silence_count = 0
        self._in_speech = False

    def _flush(self) -> np.ndarray:
        result = np.concatenate(self._speech_frames)
        self.reset()
        return result
