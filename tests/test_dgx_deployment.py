"""Deployment contract for the independently owned GX10 voice stack."""

import re
from pathlib import Path

import pytest
import yaml

from dgx.tts.runtime import normalize_language, require_gb10_cuda

ROOT = Path(__file__).resolve().parents[1]
ASR_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
TTS_BASE_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
LOCK_ENTRY = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(?:\[[^]]+\])?==\S+")
LOCK_HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}")


def _compose() -> dict:
    return yaml.safe_load((ROOT / "dgx/docker-compose.yml").read_text(encoding="utf-8"))


def _assert_hash_locked(requirements: str) -> None:
    lines = requirements.splitlines()
    entry_indexes = [index for index, line in enumerate(lines) if LOCK_ENTRY.match(line)]

    assert entry_indexes
    for index, next_index in zip(entry_indexes, [*entry_indexes[1:], len(lines)], strict=True):
        hashes = re.findall(r"--hash=\S+", "\n".join(lines[index:next_index]))
        assert hashes
        assert all(LOCK_HASH.fullmatch(value) for value in hashes)


def test_nuc_shared_ai_hosts_use_one_configurable_dgx_gateway() -> None:
    compose = yaml.safe_load((ROOT / "compose.yml").read_text(encoding="utf-8"))
    hosts = set(compose["services"]["voip-agent"]["extra_hosts"])
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert hosts == {
        "dgx-spark:${DGX_HOST_IP:-192.168.68.41}",
        "${AI_ORIGIN_HOST:-mate.olcon.de}:${DGX_HOST_IP:-192.168.68.41}",
    }
    assert "DGX_HOST_IP=192.168.68.41" in env_example
    assert "AI_ORIGIN=https://mate.olcon.de" in env_example
    assert "AI_ORIGIN_HOST=mate.olcon.de" in env_example


def test_ci_gates_python_314_and_both_runtime_image_imports() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    assert [str(version) for version in jobs["test"]["strategy"]["matrix"]["python-version"]] == [
        "3.12",
        "3.13",
        "3.14",
    ]
    commands = "\n".join(
        step.get("run", "") for step in jobs["container-smoke"]["steps"] if "run" in step
    )
    assert "docker build --file Dockerfile " in commands
    assert "docker build --file Dockerfile.pjsip-poc " in commands
    assert "import agent.main, pjsua2, webrtcvad" in commands
    assert "import agent.pjsip_poc, pjsua2, webrtcvad" in commands


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
    runtime_input = (ROOT / "dgx/tts/requirements-slim-arm64.in").read_text(encoding="utf-8")
    requirements = (ROOT / "dgx/tts/requirements-slim-arm64.lock").read_text(encoding="utf-8")
    tts_packages = (ROOT / "dgx/tts/tts-packages-arm64.lock").read_text(encoding="utf-8")
    flash_attn = (ROOT / "dgx/tts/flash-attn-arm64.lock").read_text(encoding="utf-8")

    assert (
        "ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.3.1-devel-ubuntu26.04@"
        "sha256:da3989b0ea8e8b4b241711edd5823bc1cc83d05a01882258bddad84d7394c37e" in dockerfile
    )
    assert (
        "ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.3.1-base-ubuntu26.04@"
        "sha256:f65b4f0b65bbf2e0a2520cebaec3120bf4ed110aecc3e7dcab3b11cb508a0484" in dockerfile
    )
    assert [line for line in dockerfile.splitlines() if line.startswith("FROM ")] == [
        "FROM ${CUDA_DEVEL_IMAGE} AS builder",
        "FROM ${CUDA_RUNTIME_IMAGE} AS runtime",
    ]
    for input_dependency, resolved_dependency in (
        ("accelerate==1.12.0", "accelerate==1.12.0"),
        ("einops==0.8.2", "einops==0.8.2"),
        ("fastapi==0.135.3", "fastapi==0.135.3"),
        ("huggingface-hub==0.36.2", "huggingface-hub==0.36.2"),
        ("librosa==0.11.0", "librosa==0.11.0"),
        ("onnxruntime==1.28.0", "onnxruntime==1.28.0"),
        ("pydantic==2.12.5", "pydantic==2.12.5"),
        ("soundfile==0.14.0", "soundfile==0.14.0"),
        ("sox==1.5.0", "sox==1.5.0"),
        ("torch==2.13.0+cu132", "torch==2.13.0+cu132"),
        ("transformers==4.57.3", "transformers==4.57.3"),
        ("uvicorn[standard]==0.44.0", "uvicorn==0.44.0"),
    ):
        assert input_dependency in runtime_input
        assert resolved_dependency in requirements

    assert (
        "faster-qwen3-tts==0.2.6 "
        "--hash=sha256:3881a41dc189f0a6e93fa047f376deffeb2fa84e888e7d570f79b3e2267765cc"
        in tts_packages
    )
    assert (
        "qwen-tts==0.1.1 "
        "--hash=sha256:11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d"
        in tts_packages
    )
    assert (
        "flash-attn==2.8.3 "
        "--hash=sha256:1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d"
        in flash_attn
    )
    assert "torchaudio" not in tts_packages
    assert "gradio" not in tts_packages
    for lock in (requirements, tts_packages, flash_attn):
        _assert_hash_locked(lock)

    assert (
        "COPY tts/requirements-slim-arm64.lock /tmp/requirements-slim-arm64.lock\n"
        "RUN pip install --no-cache-dir --require-hashes \\\n"
        "      -r /tmp/requirements-slim-arm64.lock" in dockerfile
    )
    assert (
        "COPY tts/tts-packages-arm64.lock /tmp/tts-packages-arm64.lock\n"
        "RUN pip install --no-cache-dir --require-hashes --no-deps \\\n"
        "      -r /tmp/tts-packages-arm64.lock" in dockerfile
    )
    assert (
        "COPY tts/flash-attn-arm64.lock /tmp/flash-attn-arm64.lock\n"
        "RUN pip install --no-cache-dir --require-hashes --no-deps --no-build-isolation \\\n"
        "      -r /tmp/flash-attn-arm64.lock" in dockerfile
    )
    assert "--require-hashes" in dockerfile
    assert "flash-attn install failed" not in dockerfile


def test_tts_lock_contract_rejects_tampered_hashes() -> None:
    requirements = (ROOT / "dgx/tts/requirements-slim-arm64.lock").read_text(encoding="utf-8")
    tts_packages = (ROOT / "dgx/tts/tts-packages-arm64.lock").read_text(encoding="utf-8")
    tampered_runtime = requirements.replace(
        "3e2091cd341423207e2f084a6654b1efcd250dc326f2a37d6dde446e07cabb11",
        "not-a-full-sha256",
    )
    tampered_qwen = tts_packages.replace(
        "3881a41dc189f0a6e93fa047f376deffeb2fa84e888e7d570f79b3e2267765cc",
        "not-a-full-sha256",
    )

    for tampered in (tampered_runtime, tampered_qwen):
        with pytest.raises(AssertionError):
            _assert_hash_locked(tampered)
