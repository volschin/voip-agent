import audioop
import numpy as np
import pytest
from agent.audio import alaw_decode, alaw_encode, resample_8k_to_16k, resample_24k_to_8k, VadBuffer


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
    assert rms_diff / rms_orig < 0.01


def test_resample_8k_to_16k_shape():
    samples = _sine_8k(100)
    out = resample_8k_to_16k(samples)
    assert len(out) == 1600


def test_resample_24k_to_8k_shape():
    n = int(24000 * 0.1)
    samples = np.zeros(n, dtype=np.int16)
    out = resample_24k_to_8k(samples)
    assert len(out) == 800


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
    for _ in range(15):
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
