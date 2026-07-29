import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from dgx.tts.clone_runtime import CloneRuntime
from dgx.tts.profiles import ProfileError, VoiceProfile


def _profile(tmp_path: Path) -> VoiceProfile:
    return VoiceProfile(
        profile_id="shared-female-de-v1",
        audio_path=tmp_path / "reference.wav",
        reference_text="Guten Tag.",
        language="german",
        source_type="licensed-human-reference-private",
        source_model=None,
        source_revision="private-v1",
        source_sha256="1" * 64,
        sha256="2" * 64,
        selected_at="2026-07-29T12:00:00Z",
        evaluation_score=91.25,
        selected_candidate_id="candidate-a",
        design_instruction=None,
    )


class ExactCloneModel:
    def __init__(self, profile: VoiceProfile) -> None:
        self.profile = profile
        self.calls: list[tuple[str, str]] = []
        self.warm_audio = [np.array([0.25, -0.25], dtype=np.float32)]
        self.sample_rate = 24_000

    def _verify(self, *, text, language, ref_audio, ref_text, **kwargs) -> None:
        assert language == "german"
        assert ref_audio == self.profile.audio_path
        assert ref_text == self.profile.reference_text
        assert "instruct" not in kwargs
        self.calls.append((text, kwargs.pop("mode")))

    def generate_voice_clone(self, **kwargs):
        mode = "warm" if kwargs["text"] == "Bereit." else "synthesize"
        self._verify(**kwargs, mode=mode)
        return self.warm_audio, self.sample_rate

    def generate_voice_clone_streaming(self, **kwargs):
        self._verify(**kwargs, mode="stream")
        yield np.array([1, 2], dtype=np.int16), self.sample_rate, {"first": True}


def test_clone_runtime_warms_and_synthesizes_with_fixed_reference(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    model = ExactCloneModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    runtime.warm()
    audios, sample_rate = runtime.synthesize("Hallo", profile.profile_id, "de")

    assert sample_rate == 24_000
    assert np.array_equal(audios[0], np.array([0.25, -0.25], dtype=np.float32))
    assert model.calls == [("Bereit.", "warm"), ("Hallo", "synthesize")]


def test_clone_runtime_streams_with_fixed_reference_and_chunk_size(tmp_path: Path) -> None:
    profile = _profile(tmp_path)

    class StreamingModel(ExactCloneModel):
        def generate_voice_clone_streaming(self, **kwargs):
            assert kwargs.pop("chunk_size") == 8
            self._verify(**kwargs, mode="stream")
            yield np.array([1, 2], dtype=np.int16), self.sample_rate, {"first": True}

    model = StreamingModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    chunks = list(runtime.stream("Hallo", profile.profile_id, "german"))

    assert len(chunks) == 1
    assert np.array_equal(chunks[0][0], np.array([1, 2], dtype=np.int16))
    assert model.calls == [("Hallo", "stream")]


@pytest.mark.parametrize(
    ("warm_audio", "sample_rate", "message"),
    [
        ([], 24_000, "empty"),
        ([np.array([1], dtype=np.float32)], 16_000, "24 kHz"),
    ],
)
def test_clone_runtime_rejects_invalid_warm_result(
    tmp_path: Path, warm_audio: list[np.ndarray], sample_rate: int, message: str
) -> None:
    profile = _profile(tmp_path)
    model = ExactCloneModel(profile)
    model.warm_audio = warm_audio
    model.sample_rate = sample_rate
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    with pytest.raises(RuntimeError, match=message):
        runtime.warm()


def test_clone_runtime_rejects_unknown_profile_before_model_call(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    model = ExactCloneModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    with pytest.raises(ProfileError, match="unsupported voice profile"):
        runtime.synthesize("Hallo", "other-profile", "german")

    assert model.calls == []


def test_clone_runtime_serializes_model_inference(tmp_path: Path) -> None:
    profile = _profile(tmp_path)

    class ConcurrentModel(ExactCloneModel):
        def __init__(self, selected_profile: VoiceProfile) -> None:
            super().__init__(selected_profile)
            self.active = 0
            self.maximum_active = 0
            self.counter_lock = threading.Lock()

        def generate_voice_clone(self, **kwargs):
            with self.counter_lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            time.sleep(0.05)
            try:
                return self.warm_audio, self.sample_rate
            finally:
                with self.counter_lock:
                    self.active -= 1

    model = ConcurrentModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(runtime.synthesize, f"Text {index}", profile.profile_id, "de")
            for index in range(4)
        ]
        for future in futures:
            future.result(timeout=2)

    assert model.maximum_active == 1


def test_clone_runtime_holds_lock_for_complete_stream_lifetime(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    first_chunk = threading.Event()
    release_stream = threading.Event()

    class BlockingStreamModel(ExactCloneModel):
        def generate_voice_clone_streaming(self, **kwargs):
            first_chunk.set()
            yield np.array([1, 2], dtype=np.int16), self.sample_rate, {}
            release_stream.wait(timeout=2)

    model = BlockingStreamModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)
    stream = runtime.stream("stream", profile.profile_id, "de")

    with ThreadPoolExecutor(max_workers=2) as executor:
        next_future = executor.submit(next, stream)
        next_future.result(timeout=2)
        assert first_chunk.is_set()
        synthesize_future = executor.submit(
            runtime.synthesize,
            "parallel",
            profile.profile_id,
            "de",
        )
        time.sleep(0.05)
        assert synthesize_future.done() is False
        stream.close()
        release_stream.set()
        synthesize_future.result(timeout=2)
