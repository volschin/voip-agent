"""Deployment contract for the independently owned GX10 voice stack."""

from pathlib import Path

import pytest
import yaml

from dgx.tts.runtime import normalize_language, require_gb10_cuda

ROOT = Path(__file__).resolve().parents[1]
ASR_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
TTS_BASE_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"


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


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (None, None),
        ("de", "german"),
        ("en", "english"),
        ("german", "german"),
    ],
)
def test_tts_runtime_maps_openai_language_codes(language: str | None, expected: str | None) -> None:
    assert normalize_language(language) == expected


def test_tts_deployment_is_offline_base_only_with_private_read_only_profile() -> None:
    service = _compose()["services"]["qwen3-tts"]
    environment = set(service["environment"])
    volumes = set(service["volumes"])
    health_command = " ".join(service["healthcheck"]["test"])

    assert "HF_HUB_OFFLINE=1" in environment
    assert "TRANSFORMERS_OFFLINE=1" in environment
    assert any(TTS_BASE_REVISION in value and "1.7B-Base" in value for value in environment)
    assert not any("VoiceDesign" in value for value in environment)
    assert "QWEN_TTS_DEFAULT_PROFILE=shared-female-de-v1" in environment
    assert "${HOME}/.cache/huggingface:/root/.cache/huggingface:ro" in volumes
    assert "/home/volsch/voice-private/profiles:/run/voice-profiles:ro" in volumes
    assert TTS_BASE_REVISION in health_command
    assert "shared-female-de-v1" in health_command
    assert service["healthcheck"]["start_period"] == "300s"


def test_tts_image_copies_clone_code_but_no_private_profile_assets() -> None:
    dockerfile = (ROOT / "dgx/tts/Dockerfile").read_text(encoding="utf-8")
    copy_lines = {line.strip() for line in dockerfile.splitlines() if line.startswith("COPY ")}
    server = (ROOT / "dgx/tts/server.py").read_text(encoding="utf-8")
    build = _compose()["services"]["qwen3-tts"]["build"]

    assert build == {"context": ".", "dockerfile": "tts/Dockerfile"}
    assert "COPY tts/runtime.py /app/dgx/tts/runtime.py" in dockerfile
    assert "COPY tts/profiles.py /app/dgx/tts/profiles.py" in dockerfile
    assert "COPY tts/clone_runtime.py /app/dgx/tts/clone_runtime.py" in dockerfile
    assert "COPY tts/api.py /app/dgx/tts/api.py" in dockerfile
    assert not any(".wav" in line or "profiles/" in line for line in copy_lines)
    assert "generate_voice_design" not in server
    assert 'CMD ["python3", "-m", "dgx.tts.server"]' in dockerfile


def test_tts_image_pins_base_digest_and_runtime_dependencies() -> None:
    dockerfile = (ROOT / "dgx/tts/Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "dgx/tts/requirements-arm64.lock").read_text(encoding="utf-8")

    assert (
        "FROM ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3@"
        "sha256:6506ebcb79b1bd0d48f8afca127984791f32345333be1be0fef334eaa5a9e23a" in dockerfile
    )
    for dependency in (
        "faster-qwen3-tts==0.2.6",
        "qwen-tts==0.1.1",
        "fastapi==0.135.3",
        "uvicorn[standard]==0.44.0",
        "pydantic==2.12.5",
        "flash-attn==2.8.3",
        "av==17.0.1",
    ):
        assert dependency in requirements
    locked = [line for line in requirements.splitlines() if line and not line.startswith("#")]
    assert all("==" in line and " --hash=sha256:" in line for line in locked)
    assert "--require-hashes" in dockerfile
    assert "flash-attn install failed" not in dockerfile
