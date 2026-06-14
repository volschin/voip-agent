# Smart Turn v3 In-Process ONNX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the turn detector from a DGX HTTP client to an in-process Smart Turn v3 ONNX model (12 ms CPU), deleting the DGX service while keeping the gating behaviour, FSM, barge-in, and the `classify(pcm) -> bool` interface unchanged. Stays opt-in / fail-closed.

**Architecture:** `TurnDetector` (same module, same async `classify`) downloads a revision-pinned v3.2 ONNX at construction via `hf_hub_download`, builds an `onnxruntime` session (CPU provider by default, configurable), and computes Whisper log-mel features with `transformers.WhisperFeatureExtractor`. Inference runs in `asyncio.to_thread`. It is constructed only when `turn_detection_enabled` is true; otherwise the detector is `None` and the legacy 800 ms path runs exactly as today. `AriClient._gate_turn_end` and all gating tests are untouched.

**Tech Stack:** Python 3.12+, onnxruntime, transformers (numpy-only WhisperFeatureExtractor, no torch), huggingface_hub, numpy, pydantic-settings, pytest-asyncio (`asyncio_mode=auto`), `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-06-14-smart-turn-v3-in-process-design.md`

---

## File Structure

- **Modify** `pyproject.toml` — add runtime deps (`onnxruntime`, `transformers`, `huggingface_hub`); register `realmodel` pytest marker.
- **Modify** `agent/config.py` — drop `turn_detector_url`, `turn_classify_timeout_ms`; add `turn_model_repo/filename/revision`, `turn_onnx_providers` + `turn_onnx_provider_list` property.
- **Rewrite** `agent/turn_detector.py` — `TurnDetector` (in-process ONNX) replacing the HTTP `TurnDetectorClient`.
- **Modify** `agent/main.py` — construct `TurnDetector` only when enabled; rename import.
- **Modify** `agent/ari.py` — rename type import/hint `TurnDetectorClient` → `TurnDetector` (no behaviour change).
- **Rewrite** `tests/test_turn_detector.py` — mock session + feature extractor (no model download in CI).
- **Modify** `tests/test_config.py` — update turn-detection field assertions.
- **Delete** `dgx/SMART_TURN.md`; **Modify** `README.md`, `CLAUDE.md` — in-process v3 wording + env vars.

Run all tests with `venv/bin/pytest -v`.

---

## Task 1: Dependencies + pytest marker

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add runtime deps**

In `pyproject.toml`, in `[project].dependencies`, after `"aiofiles>=23.2",` add:

```toml
    "onnxruntime>=1.18",
    "transformers>=4.44",
    "huggingface_hub>=0.24",
```

- [ ] **Step 2: Register the realmodel marker**

In `pyproject.toml`, replace the `[tool.pytest.ini_options]` block:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

with:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "realmodel: downloads the real Smart Turn v3 ONNX model (skipped in CI)",
]
```

- [ ] **Step 3: Install the new deps**

Run: `venv/bin/pip install -e ".[dev]"`
Expected: installs onnxruntime, transformers, huggingface_hub without error.

- [ ] **Step 4: Verify imports resolve**

Run: `venv/bin/python -c "import onnxruntime, transformers, huggingface_hub; print('OK')"`
Expected: prints `OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add onnxruntime/transformers/hf_hub deps + realmodel marker"
```

---

## Task 2: Config fields

**Files:**
- Modify: `agent/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Update the failing test**

In `tests/test_config.py`, replace the body of `test_turn_detection_defaults` with:

```python
def test_turn_detection_defaults(settings):
    assert settings.turn_detection_enabled is False
    assert settings.turn_complete_threshold == 0.5
    assert settings.turn_vad_silence_ms == 200
    assert settings.turn_model_repo == "pipecat-ai/smart-turn-v3"
    assert settings.turn_model_filename == "smart-turn-v3.2-cpu.onnx"
    assert settings.turn_model_revision == "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
    assert settings.turn_onnx_providers == "CPUExecutionProvider"
    assert settings.turn_onnx_provider_list == ["CPUExecutionProvider"]
    # Removed HTTP-only fields must be gone.
    assert not hasattr(settings, "turn_detector_url")
    assert not hasattr(settings, "turn_classify_timeout_ms")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_config.py::test_turn_detection_defaults -v`
Expected: FAIL (`turn_model_repo` missing / `turn_detector_url` still present).

- [ ] **Step 3: Replace the config block**

In `agent/config.py`, replace the existing turn-detection block:

```python
    # Turn detection (Smart Turn v2 on DGX). Fail closed: off by default so the
    # legacy single-buffer 800ms silence path is unchanged until explicitly
    # enabled AND German precision has been verified live.
    turn_detection_enabled: bool = False
    turn_detector_url: str = "http://dgx-spark:8004"
    turn_complete_threshold: float = 0.5  # prob >= this => caller's turn is complete
    turn_classify_timeout_ms: int = 150  # latency budget; exceed => degrade to flush
    turn_vad_silence_ms: int = 200  # endpoint-candidate floor for the turn-end VAD
```

with:

```python
    # Turn detection (Smart Turn v3, in-process ONNX). Fail closed: off by
    # default so the legacy single-buffer 800ms silence path is unchanged until
    # explicitly enabled AND German precision has been verified live. When
    # enabled the model is downloaded once (revision-pinned) and run on CPU.
    turn_detection_enabled: bool = False
    turn_complete_threshold: float = 0.5  # prob >= this => caller's turn is complete
    turn_vad_silence_ms: int = 200  # endpoint-candidate floor for the turn-end VAD
    turn_model_repo: str = "pipecat-ai/smart-turn-v3"
    turn_model_filename: str = "smart-turn-v3.2-cpu.onnx"
    turn_model_revision: str = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
    # Comma-separated onnxruntime execution providers. Default CPU; for the
    # NUC iGPU install onnxruntime-openvino and set
    # "OpenVINOExecutionProvider" + turn_model_filename=smart-turn-v3.2-gpu.onnx.
    turn_onnx_providers: str = "CPUExecutionProvider"

    @property
    def turn_onnx_provider_list(self) -> list[str]:
        return [p.strip() for p in self.turn_onnx_providers.split(",") if p.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_config.py::test_turn_detection_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat(config): switch turn-detection config to in-process v3 model"
```

---

## Task 3: Rewrite TurnDetector (in-process ONNX)

**Files:**
- Rewrite: `agent/turn_detector.py`
- Test: `tests/test_turn_detector.py`

- [ ] **Step 1: Rewrite the tests**

Replace the ENTIRE contents of `tests/test_turn_detector.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_turn_detector.py -v`
Expected: FAIL (`TurnDetector` doesn't exist; only `TurnDetectorClient` does).

- [ ] **Step 3: Rewrite the module**

Replace the ENTIRE contents of `agent/turn_detector.py` with:

```python
import asyncio

import numpy as np

_SAMPLE_RATE = 16000
_MAX_SAMPLES = 8 * _SAMPLE_RATE  # model accepts up to 8 s


class TurnDetector:
    """In-process Smart Turn v3 end-of-turn classifier.

    Downloads a revision-pinned ONNX model once at construction and runs it on
    CPU (or a configured execution provider). `classify` extracts Whisper
    log-mel features and runs inference off the event loop. Same async
    `classify(pcm) -> bool` interface as the prior HTTP client, so the ARI
    gating path is unchanged.

    Tests inject `session` + `feature_extractor` to avoid a model download.
    """

    def __init__(
        self,
        model_repo: str,
        model_filename: str,
        model_revision: str,
        providers: list[str],
        threshold: float = 0.5,
        session=None,
        feature_extractor=None,
    ) -> None:
        self._threshold = threshold
        if session is not None and feature_extractor is not None:
            self._session = session
            self._fx = feature_extractor
            return
        # Heavy deps imported lazily so importing this module (and running the
        # mocked tests) doesn't require onnxruntime/transformers, and so the
        # download cost is paid only when the feature is actually enabled.
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        path = hf_hub_download(
            repo_id=model_repo, filename=model_filename, revision=model_revision
        )
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1  # never starve the RTP pacing loop
        self._session = ort.InferenceSession(path, sess_options=so, providers=providers)
        self._fx = WhisperFeatureExtractor(chunk_length=8)

    def aclose(self) -> None:
        # No external resources to release; kept for call-site symmetry.
        pass

    async def classify(self, pcm_16k: np.ndarray) -> bool:
        """True = caller's turn is complete. Raises on inference failure."""
        return await asyncio.to_thread(self._classify_sync, pcm_16k)

    def _classify_sync(self, pcm_16k: np.ndarray) -> bool:
        audio = pcm_16k[-_MAX_SAMPLES:].astype(np.float32) / 32768.0
        inputs = self._fx(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_MAX_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        outputs = self._session.run(None, {"input_features": inputs.input_features})
        prob = float(np.asarray(outputs[0]).reshape(-1)[0])
        return prob > self._threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_turn_detector.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/turn_detector.py tests/test_turn_detector.py
git commit -m "feat(turn): rewrite TurnDetector as in-process Smart Turn v3 ONNX"
```

---

## Task 4: Wire main.py + rename ari.py type

**Files:**
- Modify: `agent/main.py`
- Modify: `agent/ari.py`

- [ ] **Step 1: Update main.py import**

In `agent/main.py`, change:

```python
from agent.turn_detector import TurnDetectorClient
```

to:

```python
from agent.turn_detector import TurnDetector
```

- [ ] **Step 2: Construct conditionally**

In `agent/main.py`, replace the block:

```python
    turn_detector = TurnDetectorClient(
        base_url=s.turn_detector_url,
        threshold=s.turn_complete_threshold,
        timeout_ms=s.turn_classify_timeout_ms,
        client=http_client,
    )
```

with:

```python
    # Build the detector (downloads the model) only when the feature is on;
    # otherwise pass None and AriClient runs the legacy silence path.
    turn_detector = (
        TurnDetector(
            model_repo=s.turn_model_repo,
            model_filename=s.turn_model_filename,
            model_revision=s.turn_model_revision,
            providers=s.turn_onnx_provider_list,
            threshold=s.turn_complete_threshold,
        )
        if s.turn_detection_enabled
        else None
    )
```

The `ari = AriClient(settings=s, pipeline=pipeline, turn_detector=turn_detector)` line stays unchanged.

- [ ] **Step 3: Rename the type in ari.py**

In `agent/ari.py`, change the import:

```python
from agent.turn_detector import TurnDetectorClient
```

to:

```python
from agent.turn_detector import TurnDetector
```

and the constructor hint:

```python
        turn_detector: TurnDetectorClient | None = None,
```

to:

```python
        turn_detector: TurnDetector | None = None,
```

- [ ] **Step 4: Import sanity check**

Run: `venv/bin/python -c "from agent.main import main; print('OK')"`
Expected: prints `OK`

- [ ] **Step 5: Run the ARI suite (interface unchanged → no regressions)**

Run: `venv/bin/pytest tests/test_ari.py -v`
Expected: all pass (gating tests use an `AsyncMock` detector; the `classify` contract held across the migration).

- [ ] **Step 6: Commit**

```bash
git add agent/main.py agent/ari.py
git commit -m "feat(main,ari): construct in-process TurnDetector only when enabled"
```

---

## Task 5: Docs

**Files:**
- Delete: `dgx/SMART_TURN.md`
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Delete the DGX service contract**

```bash
git rm dgx/SMART_TURN.md
```

- [ ] **Step 2: Update the README architecture paragraph**

In `README.md`, replace the existing "Turn detection (opt-in)" paragraph with:

```markdown
**Turn detection (opt-in):** by default the caller's turn ends on a fixed 800 ms silence — a long thinking pause is misread as end-of-turn and the agent talks over them. With `TURN_DETECTION_ENABLED=true` the listen path drops the VAD floor to ~200 ms and confirms each candidate with an **in-process** [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) ONNX model (Whisper-Tiny encoder, 8 MB, ~12 ms CPU, 23 languages incl. German): a complete turn flushes immediately, an incomplete one keeps listening (bounded by `max_speech_ms`). The model is downloaded once (revision-pinned) on first start when enabled; inference runs off the event loop. Classifier error or hard cap degrades to the legacy silence flush (fail toward responding). Barge-in is unchanged. Default off and fail-closed — German precision needs live verification first.
```

- [ ] **Step 3: Update the README env-var table**

In `README.md`, replace the five `TURN_*` rows with:

```markdown
| `TURN_DETECTION_ENABLED` | `false` | Enable Smart Turn v3 in-process end-of-turn gating (fail-closed; verify German live first) |
| `TURN_COMPLETE_THRESHOLD` | `0.5` | `prob` ≥ this ⇒ turn complete |
| `TURN_VAD_SILENCE_MS` | `200` | Lowered VAD silence floor for the turn-end candidate |
| `TURN_MODEL_REPO` | `pipecat-ai/smart-turn-v3` | HF repo for the ONNX model |
| `TURN_MODEL_FILENAME` | `smart-turn-v3.2-cpu.onnx` | Model file (use `-gpu.onnx` with an OpenVINO/CUDA provider) |
| `TURN_MODEL_REVISION` | `f766f81…` | Pinned HF revision |
| `TURN_ONNX_PROVIDERS` | `CPUExecutionProvider` | Comma-separated onnxruntime execution providers |
```

- [ ] **Step 4: Update .env.example**

In `.env.example`, replace the turn-detection block with:

```bash
# Turn detection (Smart Turn v3, in-process ONNX). Off by default (fail-closed); verify German live first.
TURN_DETECTION_ENABLED=false
TURN_COMPLETE_THRESHOLD=0.5
TURN_VAD_SILENCE_MS=200
TURN_MODEL_REPO=pipecat-ai/smart-turn-v3
TURN_MODEL_FILENAME=smart-turn-v3.2-cpu.onnx
TURN_MODEL_REVISION=f766f81d3cfdf7737ac64aad813d91bbfd56bf93
TURN_ONNX_PROVIDERS=CPUExecutionProvider
```

- [ ] **Step 5: Update CLAUDE.md**

In `CLAUDE.md`, in the "Turn detection (Smart Turn v2, opt-in)" architecture bullet, change the title to "Turn detection (Smart Turn v3, opt-in)" and replace the sentence

> `_gate_turn_end`. ... Contract pinned in `dgx/SMART_TURN.md`.

so the bullet ends with:

```markdown
`classify` runs **before** the turn lock; if state leaves LISTENING during the await, the verdict is discarded. The detector is **in-process**: an 8 MB Smart Turn v3 ONNX model (Whisper-Tiny encoder) downloaded once on enable (revision-pinned), run via `onnxruntime` in `asyncio.to_thread` with `inter_op_num_threads=1` so it never starves RTP pacing; Whisper log-mel features via `transformers.WhisperFeatureExtractor`.
```

Then update the "Smart Turn v2 fail-closed" key-design-decision bullet: change "v2" to "v3", drop the `dgx/SMART_TURN.md` reference, and note the model is auto-downloaded in-process (no DGX service).

- [ ] **Step 6: Commit**

```bash
git add README.md .env.example CLAUDE.md
git commit -m "docs: Smart Turn v3 in-process detector (drop DGX service)"
```

---

## Task 6: Full suite + lint gate

- [ ] **Step 1: Run the entire test suite**

Run: `venv/bin/pytest -v`
Expected: PASS (the `realmodel` integration test, if added later, is opt-in and not collected by default).

- [ ] **Step 2: Format + lint (CI gate)**

Run: `venv/bin/ruff format agent/ tests/` then `venv/bin/ruff check agent/ tests/`
Expected: clean. Commit any reformat:

```bash
git add -A
git commit -m "style: ruff format v3 turn-detection changes"
```

---

## Post-plan follow-up (outside the repo, do manually)

- Update the auto-memory `smart-turn-v2-live-gate-pending.md`: there is no DGX container to build anymore; the live gate is now "enable flag → model auto-downloads → verify German precision live." Rename/retitle to v3.

---

## Self-Review Notes (addressed)

- **Spec coverage:** in-process rewrite (Task 3), download-at-startup + revision pin (Tasks 2-3), transformers mel + lazy import (Task 3), configurable providers/CPU default (Task 2-4), conditional construction (Task 4), interface unchanged so ARI gating untouched (Task 4 verifies), deps (Task 1), docs + DGX deletion (Task 5), tests mock the boundary / no CI download (Task 3).
- **Type/name consistency:** `TurnDetector`, `classify`, `turn_onnx_provider_list`, `turn_model_repo/filename/revision`, `_MAX_SAMPLES`, `session`/`feature_extractor` injection seam used identically across tasks and tests.
- **Removed fields** (`turn_detector_url`, `turn_classify_timeout_ms`) are asserted absent in Task 2's test and have no remaining references after Task 4.
- **OpenVINO/iGPU** is config-only (provider string + gpu filename), not built — matches the spec's out-of-scope.
```
