import numpy as np
import pytest

from agent.audio import (
    StreamingPcm16Resampler,
    VadBuffer,
    alaw_decode,
    alaw_encode,
    resample_8k_to_16k,
    resample_24k_to_8k,
    resample_pcm16,
)


def _sine_8k(duration_ms: int = 200, freq: int = 440) -> np.ndarray:
    n = int(8000 * duration_ms / 1000)
    t = np.arange(n) / 8000
    return (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)


def test_alaw_roundtrip():
    original = _sine_8k(100)
    encoded = alaw_encode(original)
    decoded = alaw_decode(encoded)
    assert decoded.dtype == np.int16
    assert len(decoded) == len(original)
    rms_orig = np.sqrt(np.mean(original.astype(np.float32) ** 2))
    rms_diff = np.sqrt(np.mean((decoded.astype(np.float32) - original.astype(np.float32)) ** 2))
    assert rms_diff / rms_orig < 0.02  # aLaw is lossy; ~1-1.5% RMS error on real hardware


def test_resample_8k_to_16k_shape():
    samples = _sine_8k(100)
    out = resample_8k_to_16k(samples)
    assert len(out) == 1600


def test_resample_24k_to_8k_shape():
    n = int(24000 * 0.1)
    samples = np.zeros(n, dtype=np.int16)
    out = resample_24k_to_8k(samples)
    assert len(out) == 800


def test_streaming_pcm_resampler_matches_one_shot_across_arbitrary_splits():
    samples = np.arange(-5000, 5000, dtype=np.int16)
    one_shot = resample_pcm16(samples, 24_000, 16_000)
    stream = StreamingPcm16Resampler(24_000, 16_000)
    chunks = [samples[:1], samples[1:101], samples[101:4097], samples[4097:]]

    chunked = b"".join(stream.process(chunk) for chunk in chunks)
    stream.close()

    assert chunked == one_shot
    assert len(chunked) == 13_334


def test_streaming_pcm_resampler_rejects_invalid_or_closed_input():
    stream = StreamingPcm16Resampler(24_000, 16_000)
    with pytest.raises(ValueError, match="one-dimensional int16"):
        stream.process(np.zeros((2, 2), dtype=np.int16))
    stream.close()
    with pytest.raises(RuntimeError, match="closed"):
        stream.process(np.zeros(3, dtype=np.int16))


def test_vad_buffer_returns_none_during_silence():
    buf = VadBuffer(sample_rate=16000, frame_ms=20, silence_threshold_ms=200)
    frame = np.zeros(320, dtype=np.int16)
    for _ in range(20):
        result = buf.add_frame(frame)
    assert result is None


def test_vad_buffer_flushes_after_speech_then_silence():
    buf = VadBuffer(sample_rate=16000, frame_ms=20, silence_threshold_ms=200)
    t = np.arange(320) / 16000
    speech_frame = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)
    silence_frame = np.zeros(320, dtype=np.int16)

    for _ in range(15):
        buf.add_frame(speech_frame)

    result = None
    for _ in range(30):  # webrtcvad has ~6-frame VAD hangover; use extra headroom
        result = buf.add_frame(silence_frame)
        if result is not None:
            break

    assert result is not None
    assert result.dtype == np.int16
    assert len(result) > 0


def test_vad_buffer_hard_cap_flushes_at_15s():
    """VadBuffer must flush at max_speech_ms even without trailing silence."""
    buf = VadBuffer(sample_rate=16000, frame_ms=20, silence_threshold_ms=800, max_speech_ms=200)
    t = np.arange(320) / 16000
    speech_frame = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)

    results = []
    for _ in range(15):
        r = buf.add_frame(speech_frame)
        if r is not None:
            results.append(r)

    assert len(results) == 1
    assert results[0].dtype == np.int16


class _FakeVad:
    """Scripted webrtcvad replacement: is_speech returns the next scripted bool."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def is_speech(self, _frame_bytes, _sample_rate):
        v = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return v


def test_add_frame_candidate_returns_without_reset():
    vad = VadBuffer(silence_threshold_ms=40, frame_ms=20)  # _silence_threshold = 2
    vad._vad = _FakeVad([False, True, True, False, False])
    f = np.zeros(320, dtype=np.int16)
    assert vad.add_frame_candidate(f) is None  # pre-speech silence -> None, nothing buffered
    assert vad.add_frame_candidate(f) is None  # speech 1
    assert vad.add_frame_candidate(f) is None  # speech 2
    assert vad.add_frame_candidate(f) is None  # silence 1
    cand = vad.add_frame_candidate(f)  # silence 2 -> threshold
    assert cand is not None
    assert len(cand) == 320 * 4  # 4 buffered (pre-speech silence dropped), no reset
    assert vad._in_speech is True  # buffer retained


def test_continue_speech_retains_buffer():
    vad = VadBuffer(silence_threshold_ms=40, frame_ms=20)  # _silence_threshold = 2
    vad._vad = _FakeVad([True, False, False, True, False, False])
    f = np.ones(320, dtype=np.int16)
    vad.add_frame_candidate(f)  # speech 1
    vad.add_frame_candidate(f)  # silence 1
    cand1 = vad.add_frame_candidate(f)  # silence 2 -> candidate (3 frames)
    assert cand1 is not None and len(cand1) == 320 * 3
    vad.continue_speech()  # keep listening
    vad.add_frame_candidate(f)  # speech 2 (frame 4)
    vad.add_frame_candidate(f)  # silence 1
    cand2 = vad.add_frame_candidate(f)  # silence 2 -> candidate (6 frames)
    assert cand2 is not None and len(cand2) == 320 * 6  # earlier frames retained


def test_at_cap_forces_candidate_without_silence():
    vad = VadBuffer(silence_threshold_ms=10000, frame_ms=20, max_speech_ms=60)  # cap = 3
    vad._vad = _FakeVad([True, True, True])
    f = np.zeros(320, dtype=np.int16)
    assert vad.at_cap is False
    assert vad.add_frame_candidate(f) is None
    assert vad.add_frame_candidate(f) is None
    cand = vad.add_frame_candidate(f)  # 3rd frame -> at cap
    assert cand is not None
    assert vad.at_cap is True
