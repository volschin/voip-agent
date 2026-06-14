import numpy as np
import pytest

from agent.turn_detector import TurnDetector


class _FakeSession:
    def __init__(self, prob):
        self._prob = prob
        self.feeds = []

    def run(self, _outputs, feeds):
        self.feeds.append(feeds)
        return [np.array([[self._prob]], dtype=np.float32)]


class _BoomSession:
    def run(self, _outputs, _feeds):
        raise RuntimeError("inference boom")


class _FakeFeatureExtractor:
    def __init__(self):
        self.last_audio_len = None

    def __call__(self, audio, **_kwargs):
        self.last_audio_len = len(audio)

        class _Batch:
            input_features = np.zeros((1, 80, 800), dtype=np.float32)

        return _Batch()


def _detector(session, fx, threshold=0.5):
    return TurnDetector(
        model_repo="r",
        model_filename="f",
        model_revision="rev",
        providers=["CPUExecutionProvider"],
        threshold=threshold,
        session=session,
        feature_extractor=fx,
    )


async def test_classify_complete_above_threshold():
    det = _detector(_FakeSession(0.9), _FakeFeatureExtractor())
    assert await det.classify(np.zeros(2000, dtype=np.int16)) is True


async def test_classify_incomplete_below_threshold():
    det = _detector(_FakeSession(0.2), _FakeFeatureExtractor())
    assert await det.classify(np.zeros(2000, dtype=np.int16)) is False


async def test_classify_truncates_to_last_8s():
    fx = _FakeFeatureExtractor()
    det = _detector(_FakeSession(0.9), fx)
    await det.classify(np.zeros(10 * 16000, dtype=np.int16))  # 10 s in
    assert fx.last_audio_len == 8 * 16000  # truncated to model max


async def test_classify_passes_input_features_to_session():
    session = _FakeSession(0.9)
    det = _detector(session, _FakeFeatureExtractor())
    await det.classify(np.zeros(2000, dtype=np.int16))
    assert "input_features" in session.feeds[0]


async def test_classify_raises_on_session_error():
    det = _detector(_BoomSession(), _FakeFeatureExtractor())
    with pytest.raises(RuntimeError):
        await det.classify(np.zeros(2000, dtype=np.int16))
