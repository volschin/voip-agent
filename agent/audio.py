import audioop

import numpy as np
import webrtcvad
from scipy.signal import resample_poly

PCM16_SAMPLE_RATE = 16_000
PCM16_BYTES_PER_SAMPLE = 2
PCM16_PLAYBACK_BLOCK_MS = 20
PCM16_PLAYBACK_BLOCK_BYTES = (
    PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * PCM16_PLAYBACK_BLOCK_MS // 1000
)


def alaw_decode(data: bytes) -> np.ndarray:
    return np.frombuffer(audioop.alaw2lin(data, 2), dtype=np.int16)


def alaw_encode(samples: np.ndarray) -> bytes:
    return audioop.lin2alaw(samples.astype(np.int16).tobytes(), 2)


def resample_8k_to_16k(samples: np.ndarray) -> np.ndarray:
    return resample_poly(samples, up=2, down=1).astype(np.int16)


def resample_24k_to_8k(samples: np.ndarray) -> np.ndarray:
    return resample_poly(samples, up=1, down=3).astype(np.int16)


class StreamingPcm16Resampler:
    """Resample one PCM utterance while retaining state across input chunks."""

    def __init__(self, input_rate: int, output_rate: int) -> None:
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("sample rates must be positive")
        self._input_rate = input_rate
        self._output_rate = output_rate
        self._state = None
        self._closed = False

    def process(self, samples: np.ndarray) -> bytes:
        if self._closed:
            raise RuntimeError("resampler is closed")
        if samples.ndim != 1 or samples.dtype != np.int16:
            raise ValueError("PCM must be one-dimensional int16")
        output, self._state = audioop.ratecv(
            samples.astype("<i2", copy=False).tobytes(),
            2,
            1,
            self._input_rate,
            self._output_rate,
            self._state,
        )
        return output

    def close(self) -> None:
        self._closed = True
        self._state = None


def resample_pcm16(samples: np.ndarray, input_rate: int, output_rate: int) -> bytes:
    """Resample a complete mono int16 buffer to little-endian PCM bytes."""

    stream = StreamingPcm16Resampler(input_rate, output_rate)
    try:
        return stream.process(samples)
    finally:
        stream.close()


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
