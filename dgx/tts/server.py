"""Offline, CUDA-only Qwen3-TTS Base server for private cloned profiles."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from dgx.tts.api import HealthMetadata, create_app
from dgx.tts.clone_runtime import CloneRuntime
from dgx.tts.profiles import load_profiles
from dgx.tts.runtime import require_gb10_cuda

MODEL_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
EXPECTED_MODEL_PATH = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/"
    f"snapshots/{MODEL_REVISION}"
)
MODEL_PATH = os.environ.get("QWEN_TTS_MODEL", EXPECTED_MODEL_PATH)
DEFAULT_PROFILE = os.environ.get(
    "QWEN_TTS_DEFAULT_PROFILE",
    "shared-female-de-v1",
)
PROFILE_DIR = Path("/run/voice-profiles")


def _build_app():
    if MODEL_PATH != EXPECTED_MODEL_PATH:
        raise RuntimeError("Qwen3-TTS model path must use the pinned Base revision")
    require_gb10_cuda(torch)
    profiles = load_profiles(PROFILE_DIR)
    if DEFAULT_PROFILE not in profiles:
        raise RuntimeError("default voice profile is unavailable")

    from faster_qwen3_tts import FasterQwen3TTS

    print(
        f"[load] faster-qwen3-tts Base revision={MODEL_REVISION} device=cuda",
        flush=True,
    )
    model = FasterQwen3TTS.from_pretrained(
        MODEL_PATH,
        device="cuda",
        dtype=torch.bfloat16,
    )
    runtime = CloneRuntime(model, profiles, DEFAULT_PROFILE)
    runtime.warm()
    device_name = torch.cuda.get_device_name(0)
    print(
        f"[load] profiles={','.join(sorted(profiles))} device={device_name} OK",
        flush=True,
    )
    return create_app(
        runtime,
        HealthMetadata(
            model_revision=MODEL_REVISION,
            default_profile=DEFAULT_PROFILE,
            profiles_loaded=tuple(sorted(profiles)),
            device=device_name,
        ),
    )


app = _build_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
