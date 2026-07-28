"""Deployment contract for the independently owned GX10 voice stack."""

from pathlib import Path

import pytest
import yaml

from dgx.tts.runtime import require_gb10_cuda

ROOT = Path(__file__).resolve().parents[1]
ASR_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"


def _compose() -> dict:
    return yaml.safe_load((ROOT / "dgx/docker-compose.yml").read_text(encoding="utf-8"))


def test_voice_services_keep_private_network_and_add_external_proxy_network() -> None:
    compose = _compose()

    assert compose["networks"]["shared_ai_voice"] == {
        "external": True,
        "name": "shared_ai_voice",
    }
    for name in ("qwen3-asr", "qwen3-tts"):
        service = compose["services"][name]
        assert service["networks"] == ["default", "shared_ai_voice"]
        assert "ports" not in service


def test_voice_services_request_gpu_without_legacy_runtime_name() -> None:
    compose = _compose()

    for name in ("qwen3-asr", "qwen3-tts"):
        service = compose["services"][name]
        assert "runtime" not in service
        assert service["deploy"]["resources"]["reservations"]["devices"] == [
            {
                "driver": "nvidia",
                "count": 1,
                "capabilities": ["gpu"],
            }
        ]


def test_asr_is_offline_pinned_and_health_checks_loaded_model() -> None:
    service = _compose()["services"]["qwen3-asr"]
    environment = set(service["environment"])
    command = [str(value) for value in service["command"]]
    health_command = " ".join(service["healthcheck"]["test"])

    assert "HF_HUB_OFFLINE=1" in environment
    assert "TRANSFORMERS_OFFLINE=1" in environment
    assert ASR_REVISION in " ".join(command)
    assert command[command.index("--served-model-name") + 1] == "qwen3-asr"
    assert "/v1/models" in health_command
    assert "qwen3-asr" in health_command


class FakeCuda:
    def __init__(self, *, available: bool, name: str = "NVIDIA GB10") -> None:
        self._available = available
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self._name


class FakeTorch:
    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda


def test_tts_runtime_accepts_only_nvidia_gb10_cuda() -> None:
    assert require_gb10_cuda(FakeTorch(FakeCuda(available=True))) == "cuda"

    with pytest.raises(RuntimeError, match="CUDA is required"):
        require_gb10_cuda(FakeTorch(FakeCuda(available=False)))
    with pytest.raises(RuntimeError, match="NVIDIA GB10"):
        require_gb10_cuda(FakeTorch(FakeCuda(available=True, name="NVIDIA RTX 4090")))


def test_tts_image_installs_runtime_gate_and_health_requires_loaded_model() -> None:
    dockerfile = (ROOT / "dgx/tts/Dockerfile").read_text(encoding="utf-8")
    health_command = " ".join(_compose()["services"]["qwen3-tts"]["healthcheck"]["test"])

    assert "COPY runtime.py /app/runtime.py" in dockerfile
    assert "model_loaded" in health_command
