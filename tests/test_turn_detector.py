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


async def test_classify_threshold_boundary_is_inclusive():
    # prob == threshold counts as complete (matches config "prob >= this").
    det = _detector(_FakeSession(0.5), _FakeFeatureExtractor(), threshold=0.5)
    assert await det.classify(np.zeros(2000, dtype=np.int16)) is True


async def test_classify_truncates_to_last_8s():
    fx = _FakeFeatureExtractor()
    det = _detector(_FakeSession(0.9), fx)
    await det.classify(np.zeros(10 * 16000, dtype=np.int16))  # 10 s in
    assert fx.last_audio_len == 8 * 16000  # truncated to model max


async def test_classify_left_pads_short_audio_to_window_end():
    # The model was trained with speech anchored to the END of the 8 s window
    # (left zero-pad). Short turns must be padded at the FRONT, not the back,
    # or the model reads trailing silence as "complete" and the agent cuts in.
    captured = {}

    class _CapturingFx:
        def __call__(self, audio, **_kwargs):
            captured["audio"] = np.asarray(audio)

            class _Batch:
                input_features = np.zeros((1, 80, 800), dtype=np.float32)

            return _Batch()

    det = _detector(_FakeSession(0.9), _CapturingFx())
    pcm = np.ones(16000, dtype=np.int16)  # 1 s of non-zero speech
    await det.classify(pcm)
    a = captured["audio"]
    assert len(a) == 8 * 16000  # padded up to the full window
    assert a[0] == 0.0 and a[100] == 0.0  # zeros at the FRONT
    assert a[-1] != 0.0  # speech anchored at the END


async def test_classify_passes_input_features_to_session():
    session = _FakeSession(0.9)
    det = _detector(session, _FakeFeatureExtractor())
    await det.classify(np.zeros(2000, dtype=np.int16))
    assert "input_features" in session.feeds[0]


async def test_classify_raises_on_session_error():
    det = _detector(_BoomSession(), _FakeFeatureExtractor())
    with pytest.raises(RuntimeError):
        await det.classify(np.zeros(2000, dtype=np.int16))
