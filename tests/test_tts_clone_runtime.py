import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from dgx.tts.api import encode_pcm_stream
from dgx.tts.clone_runtime import (
    CloneRuntime,
    SynthesisAdmissionTimeout,
    SynthesisCancelled,
)
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


def test_cancelled_queued_synthesis_never_starts_model_generation(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    active_started = threading.Event()
    release_active = threading.Event()
    cancel_queued = threading.Event()

    class BlockingSynthesisModel(ExactCloneModel):
        def __init__(self, selected_profile: VoiceProfile) -> None:
            super().__init__(selected_profile)
            self.generated_texts: list[str] = []

        def generate_voice_clone(self, **kwargs):
            self.generated_texts.append(kwargs["text"])
            if kwargs["text"] == "active":
                active_started.set()
                release_active.wait(timeout=2)
            return self.warm_audio, self.sample_rate

    model = BlockingSynthesisModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(runtime.synthesize, "active", profile.profile_id, "de")
        assert active_started.wait(timeout=1)
        queued = executor.submit(
            runtime.synthesize,
            "queued",
            profile.profile_id,
            "de",
            cancel_event=cancel_queued,
            lock_timeout=1.0,
        )
        cancel_queued.set()

        with pytest.raises(SynthesisCancelled):
            queued.result(timeout=1)
        release_active.set()
        active.result(timeout=1)

    assert model.generated_texts == ["active"]


def test_cancelled_active_synthesis_releases_model_for_following_request(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    active_started = threading.Event()
    cancel_active = threading.Event()

    class CooperativeSynthesisModel(ExactCloneModel):
        def __init__(self, selected_profile: VoiceProfile) -> None:
            super().__init__(selected_profile)
            self.generated_texts: list[str] = []

        def generate_voice_clone(self, *, cancel_event=None, **kwargs):
            self.generated_texts.append(kwargs["text"])
            if kwargs["text"] != "active":
                return self.warm_audio, self.sample_rate
            if cancel_event is None:
                return self.warm_audio, self.sample_rate
            active_started.set()
            cancel_event.wait(timeout=2)
            raise InterruptedError("generation cancelled")

    model = CooperativeSynthesisModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(
            runtime.synthesize,
            "active",
            profile.profile_id,
            "de",
            cancel_event=cancel_active,
        )
        assert active_started.wait(timeout=0.2)
        cancel_active.set()
        with pytest.raises(SynthesisCancelled):
            active.result(timeout=0.2)

        following = executor.submit(
            runtime.synthesize,
            "following",
            profile.profile_id,
            "de",
            lock_timeout=0.2,
        )
        following.result(timeout=0.2)

    assert model.generated_texts == ["active", "following"]


def test_synthesis_admission_timeout_recovers_for_following_request(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    active_started = threading.Event()
    release_active = threading.Event()

    class BlockingSynthesisModel(ExactCloneModel):
        def __init__(self, selected_profile: VoiceProfile) -> None:
            super().__init__(selected_profile)
            self.generated_texts: list[str] = []

        def generate_voice_clone(self, **kwargs):
            self.generated_texts.append(kwargs["text"])
            if kwargs["text"] == "active":
                active_started.set()
                release_active.wait(timeout=2)
            return self.warm_audio, self.sample_rate

    model = BlockingSynthesisModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(runtime.synthesize, "active", profile.profile_id, "de")
        assert active_started.wait(timeout=1)
        timed_out = executor.submit(
            runtime.synthesize,
            "timed-out",
            profile.profile_id,
            "de",
            lock_timeout=0.05,
        )
        with pytest.raises(SynthesisAdmissionTimeout):
            timed_out.result(timeout=1)
        release_active.set()
        active.result(timeout=1)

    runtime.synthesize("following", profile.profile_id, "de", lock_timeout=0.1)

    assert model.generated_texts == ["active", "following"]


def test_synthesize_signals_admission_before_generation(tmp_path: Path) -> None:
    # The request path stops enforcing the admission deadline once this event
    # is set, so it must be set only after the model lock is held and never
    # before generation starts.
    profile = _profile(tmp_path)
    admitted = threading.Event()
    admitted_at_generation: bool | None = None

    class SignalCheckingModel(ExactCloneModel):
        def generate_voice_clone(self, **kwargs):
            nonlocal admitted_at_generation
            admitted_at_generation = admitted.is_set()
            return self.warm_audio, self.sample_rate

    model = SignalCheckingModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    runtime.synthesize("hallo", profile.profile_id, "de", admitted=admitted)

    assert admitted_at_generation is True
    assert admitted.is_set()


def test_admission_timeout_leaves_admission_unsignalled(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    admitted = threading.Event()
    active_started = threading.Event()
    release_active = threading.Event()

    class BlockingSynthesisModel(ExactCloneModel):
        def generate_voice_clone(self, **kwargs):
            if kwargs["text"] == "active":
                active_started.set()
                release_active.wait(timeout=2)
            return self.warm_audio, self.sample_rate

    model = BlockingSynthesisModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(runtime.synthesize, "active", profile.profile_id, "de")
        assert active_started.wait(timeout=1)
        timed_out = executor.submit(
            runtime.synthesize,
            "timed-out",
            profile.profile_id,
            "de",
            lock_timeout=0.05,
            admitted=admitted,
        )
        with pytest.raises(SynthesisAdmissionTimeout):
            timed_out.result(timeout=1)
        assert not admitted.is_set()
        release_active.set()
        active.result(timeout=1)


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


async def test_cancelled_queued_stream_never_starts_model_generation(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)

    class CountingStreamModel(ExactCloneModel):
        def __init__(self, selected_profile: VoiceProfile) -> None:
            super().__init__(selected_profile)
            self.stream_calls = 0

        def generate_voice_clone_streaming(self, **kwargs):
            self.stream_calls += 1
            yield np.array([1, 2], dtype=np.int16), self.sample_rate, {}

    model = CountingStreamModel(profile)
    runtime = CloneRuntime(model, {profile.profile_id: profile}, profile.profile_id)
    active = runtime.stream("active", profile.profile_id, "de")
    next(active)
    queued = encode_pcm_stream(runtime.stream("queued", profile.profile_id, "de"))
    queued_request = asyncio.create_task(anext(queued))
    await asyncio.sleep(0.05)

    queued_request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(queued_request, timeout=1)
    active.close()

    assert model.stream_calls == 1
