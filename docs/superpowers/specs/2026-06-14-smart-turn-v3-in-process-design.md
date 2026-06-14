# Smart Turn v3 In-Process ONNX — Design

**Date:** 2026-06-14
**Status:** Approved for planning
**Supersedes:** `2026-06-13-smart-turn-v2-turn-detection-design.md` (the v2 design shipped a DGX HTTP service in PR #11; this migrates the detector to an in-process v3 ONNX model **before** the feature is ever enabled in production — `turn_detection_enabled` is still `False`).

## Problem

The shipped v2 design calls a **DGX GPU HTTP service** (`/v1/turn/classify`) because the wav2vec2-based Smart Turn v2 (~400 MB) was too heavy for the agent box. That forced a container build, a network hop, and a service to operate — all still pending (`smart-turn-v2-live-gate-pending`).

Smart Turn **v3** (released 2025-09, current v3.2) removes that reason:
- Whisper Tiny encoder + linear head, **8M params, 8 MB int8 ONNX** (~50× smaller than v2).
- **12 ms CPU inference** (60 ms on a low-end cloud box), no GPU required.
- **23 languages incl. German**, ~96 % German accuracy on pipecat's test set, better overall than v2.

At 12 ms CPU the detector fits **in-process** in the asyncio agent. That deletes the DGX container, the HTTP endpoint, and the deploy step.

## Goal

Migrate the turn detector from a DGX HTTP client to an in-process v3 ONNX model. Keep the existing gating behaviour, FSM, barge-in handling, and the `classify(pcm) -> bool` interface so `AriClient._gate_turn_end` and its tests are untouched. Stay opt-in and fail-closed.

Out of scope: changing gating logic / FSM / barge-in; enabling the feature; OpenVINO/iGPU execution (left as a config-only future toggle); a numpy reimplementation of mel features (future lean option).

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Detector location | **In-process** (was: DGX HTTP service) | v3 is 12 ms CPU; no GPU/service needed; deletes container + hop + deploy |
| Model version | **v3.2 cpu** (`smart-turn-v3.2-cpu.onnx`) | newest; CPU variant matches default execution provider |
| Model delivery | **Download at startup** via `huggingface_hub.hf_hub_download`, revision-pinned | repo stays small; HF cache makes it a one-time pull; only when enabled |
| Mel features | **`transformers.WhisperFeatureExtractor`** (numpy, no torch) | exact match to pipecat's reference `inference.py`; wrong mel silently kills accuracy |
| Execution provider | **`onnxruntime`, providers configurable, default `CPUExecutionProvider`** | CPU is fast enough; iGPU/OpenVINO is a one-setting change later, not built now |
| Interface | unchanged `async classify(pcm_16k: np.ndarray) -> bool` | `_gate_turn_end` + gating tests stay as-is |
| Rollout | still `turn_detection_enabled=False` (fail-closed) | German precision still wants a live spot-check |

## v3 inference contract (from pipecat `inference.py`)

```python
feature_extractor = WhisperFeatureExtractor(chunk_length=8)
inputs = feature_extractor(
    audio_array,                # float32/-int16 mono 16 kHz, last <=8 s of the turn
    sampling_rate=16000,
    return_tensors="np",
    padding="max_length",
    max_length=8 * 16000,
    truncation=True,
    do_normalize=True,
)
outputs = session.run(None, {"input_features": inputs.input_features})
probability = outputs[0][0].item()   # sigmoid prob; complete if > threshold (0.5)
```

Audio must be truncated to the **last 8 s** of the turn before extraction (model max). A frame VAD upstream is assumed — we already have `webrtcvad` driving candidates.

## Architecture

### Component: `TurnDetector` (`agent/turn_detector.py`, rewritten)

Replaces the HTTP `TurnDetectorClient`. Same module path and the same async `classify` signature, so `main.py` and `AriClient` change only at construction.

```python
class TurnDetector:
    def __init__(
        self,
        model_repo: str,
        model_filename: str,
        model_revision: str,
        providers: list[str],
        threshold: float = 0.5,
    ) -> None:
        # hf_hub_download(repo_id=model_repo, filename=model_filename,
        #                 revision=model_revision) -> local path (HF cache)
        # ort.InferenceSession(path, providers=providers) with
        #   sess_options: ORT_SEQUENTIAL, inter_op_num_threads=1
        # WhisperFeatureExtractor(chunk_length=8)
        ...

    async def classify(self, pcm_16k: np.ndarray) -> bool:
        # truncate to last 8 s, run feature-extract + session.run in a thread
        # (asyncio.to_thread) so the 12-60 ms inference never blocks the loop.
        # Returns prob > self._threshold. Raises on failure (caller degrades).

    def aclose(self) -> None:  # kept for symmetry; no resources to free
        ...
```

Construction does the download + session build once (synchronous, at startup). `classify` does only feature-extract + `session.run`, both off-loop via `asyncio.to_thread`. On any exception the method raises; `_gate_turn_end` already catches `Exception` and degrades to a silence flush (fail toward responding).

Truncation helper (last N seconds) lives in `agent/audio.py` next to the other audio utils, or inline in `TurnDetector` — keep it inline to avoid widening the audio module for one caller.

### Wiring: `agent/main.py`

```python
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
ari = AriClient(settings=s, pipeline=pipeline, turn_detector=turn_detector)
```

Constructing only when enabled means the model download + onnxruntime import cost is paid only by operators who turn the feature on. Flag off ⇒ `turn_detector=None` ⇒ `_turn_active()` is `False` ⇒ legacy 800 ms path, exactly as today.

### `AriClient` — unchanged behaviour

`_turn_active()`, `_gate_turn_end`, `_on_audio`, the two-buffer split, `_reset_vad`, and all gating tests are **untouched**. `_gate_turn_end` still calls `await self._turn_detector.classify(candidate)` and degrades on exception. The only ARI-side change is type hints if `TurnDetectorClient` was referenced by name (rename to `TurnDetector`).

### Config (`agent/config.py`)

Remove (HTTP-only, no longer meaningful in-process):
- `turn_detector_url`
- `turn_classify_timeout_ms`

Keep:
- `turn_detection_enabled: bool = False`
- `turn_complete_threshold: float = 0.5`
- `turn_vad_silence_ms: int = 200`

Add:
- `turn_model_repo: str = "pipecat-ai/smart-turn-v3"`
- `turn_model_filename: str = "smart-turn-v3.2-cpu.onnx"`
- `turn_model_revision: str = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"`  (pin; reproducible)
- `turn_onnx_providers: str = "CPUExecutionProvider"`  (comma-separated)
- property `turn_onnx_provider_list -> list[str]` (split/strip, like `trusted_caller_set`)

### Dependencies (`pyproject.toml`)

Add to the runtime deps: `onnxruntime`, `transformers`, `huggingface_hub`. (`numpy`/`scipy` already present.) `transformers.WhisperFeatureExtractor` is numpy-only and does **not** pull `torch` at runtime; document this so nobody adds torch. iGPU path (future): `onnxruntime-openvino` + `turn_onnx_providers=OpenVINOExecutionProvider` + `turn_model_filename=smart-turn-v3.2-gpu.onnx`.

### Docs

- Delete `dgx/SMART_TURN.md` (no DGX service anymore). Replace its intent with a short note in `README.md` and `CLAUDE.md`: detector is in-process v3 ONNX, model auto-downloaded, providers configurable.
- Update the README turn-detection paragraph + env-var table (drop URL/timeout rows, add model/provider rows).
- Update `CLAUDE.md` turn-detection invariant + the `smart-turn-v2-live-gate-pending` memory (no container to build; gate becomes: enable flag, model auto-downloads, verify German live).

## Error Handling / Edge Cases

- HF download fails at startup (enabled): construction raises → agent fails to start with a clear error (fail-closed; operator must fix network/revision). It does **not** silently disable the feature an operator explicitly enabled.
- `classify` inference error or empty/short audio: raises → `_gate_turn_end` degrades to flush. (Sub-`_MIN_CLASSIFY_SAMPLES` candidates are already short-circuited before `classify`.)
- Audio > 8 s: truncated to last 8 s inside `classify` (also bounded earlier by `max_speech_ms`).
- Event loop: inference runs in `asyncio.to_thread`; `inter_op_num_threads=1` + `ORT_SEQUENTIAL` keep it from oversubscribing the box.

## Testing

`onnxruntime` + `transformers` are **not** invoked in unit tests — mock the boundary (no model download in CI):
- `TurnDetector.classify` returns `True`/`False` around the threshold: inject a fake session whose `run` returns a scripted prob; assert truncation to 8 s and `prob > threshold` logic. (Construct with a stubbed session/extractor, or patch `_session.run`.)
- `classify` raises on a session error (degrade path is exercised by the existing `_gate_turn_end` ARI test).
- truncation: a >8 s input is reduced to 8 s of samples before extraction.
- config: `turn_onnx_provider_list` splits correctly; new defaults present; removed fields gone.
- ARI gating tests (`test_turn_gate_*`, `test_bargein_skips_turn_detector`, `test_reset_vad_*`) keep passing unchanged with an `AsyncMock` detector — proves the interface contract held across the migration.

A single **opt-in** integration test (skipped by default, marker `realmodel`) may download the model and classify a known WAV, to wire-verify features/output offline. Not run in CI.

## Risks / Caveats

- **German precision still needs a live spot-check** before `turn_detection_enabled=True`, but starts from pipecat's ~96 % rather than unknown.
- `transformers` is a heavy dependency for one class. Accepted for correctness; a numpy/scipy log-mel reimplementation is a documented future lean option if install weight on the NUC becomes a problem.
- Model pinned by revision `f766f81…`; bumping versions is a deliberate config change, not automatic.
- onnxruntime threading must stay capped (`inter_op_num_threads=1`) so a turn-end classify can't starve the RTP pacing loop.
