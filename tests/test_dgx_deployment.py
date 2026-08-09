"""Deployment contract for the independently owned GX10 voice stack."""

import re
from pathlib import Path

import pytest
import yaml

from dgx.tts.runtime import normalize_language, require_gb10_cuda

ROOT = Path(__file__).resolve().parents[1]
ASR_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
EUGR_MIDPOINT_COMMIT = "b51af15a280d28c2ad9096b3ef581524eddbd0e7"
VLLM_MIDPOINT_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
FLASHINFER_MIDPOINT_COMMIT = "d768c14e7cf5dd5df45a8a1de78ae815879f108a"
NCCL_MIDPOINT_COMMIT = "6da422082f910a8dd230f7e42e26ece4dc37bccc"
MIDPOINT_DEPENDENCY_CUTOFF = "2026-06-18T23:59:59Z"
MIDPOINT_CUDA_IMAGE_DIGEST = (
    "sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
)
QWEN3_ASR_ADAPTER_SHA256 = "e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d"
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
        package_lines = lines[index:next_index]
        package_block = "\n".join(line.split("#", maxsplit=1)[0] for line in package_lines)
        hashes = re.findall(r"--hash=\S*", package_block)
        assert hashes
        assert all(LOCK_HASH.fullmatch(value) for value in hashes)


def _normalized_lock_headers(requirements: str) -> set[str]:
    return {
        re.sub(r"[-_.]+", "-", match.group(0).split("==", maxsplit=1)[0]).lower()
        + "=="
        + match.group(0).split("==", maxsplit=1)[1]
        for line in requirements.splitlines()
        if (match := LOCK_ENTRY.match(line))
    }


def _normalized_package_name(header: str) -> str:
    package_name = header.split("==", maxsplit=1)[0].split("[", maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", package_name).lower()


def _logical_docker_instructions(dockerfile: str) -> list[str]:
    return re.sub(r"\\[ \t]*\n", " ", dockerfile).splitlines()


def _has_forbidden_torchaudio_pip_install(dockerfile: str) -> bool:
    return any(
        re.search(r"\bpip\s+install\b.*\btorchaudio\b", instruction)
        for instruction in _logical_docker_instructions(dockerfile)
    )


_ASR_FORBIDDEN_RUNTIME_PACKAGES = (
    "torch",
    "vllm",
    "triton",
    "flashinfer",
    "flash-attn",
    "flash_attn",
    "cffi",
)
_ASR_AUDIO_LOCK_PATH = "/tmp/requirements-audio-arm64.lock"
_PIP_INSTALL_COMMAND = re.compile(
    r"\b(?:python(?:3(?:\.\d+)?)?\s+-m\s+)?(?:\S*/)?pip(?:3(?:\.\d+)?)?\b"
    r"(?:\s+(?!install\b)\S+)*\s+install\b",
    flags=re.IGNORECASE,
)
_REQUIREMENTS_FILE = re.compile(r"(?:^|\s)(?:-r|--requirement)(?:\s+|=)(\S+)")
_ASR_FORBIDDEN_COMPILER_ADDITION = re.compile(
    r"(?<![A-Za-z0-9_.+-])(?:gcc(?:-\d+)?|g\+\+(?:-\d+)?|make|build-essential|cmake|"
    r"ninja(?:-build)?|nvcc|(?:nvidia-)?cuda-toolkit(?:-[A-Za-z0-9.]+)*|"
    r"cuda-(?:nvcc|compiler)(?:-[A-Za-z0-9.]+)*)(?![A-Za-z0-9_.+-])",
    flags=re.IGNORECASE,
)


def _asr_python_package_install_commands(dockerfile: str) -> list[str]:
    commands = []
    for instruction in _logical_docker_instructions(dockerfile):
        if not re.match(r"RUN\s+", instruction, flags=re.IGNORECASE):
            continue
        for match in _PIP_INSTALL_COMMAND.finditer(instruction):
            command_end = re.search(r"\s+(?:&&|\|\||;)\s+", instruction[match.end() :])
            end = match.end() + command_end.start() if command_end else len(instruction)
            commands.append(instruction[match.start() : end])
    return commands


def _has_asr_invalid_python_package_install(dockerfile: str) -> bool:
    """Return whether ASR has another Python-package path or replaces inherited runtime packages."""
    installs = _asr_python_package_install_commands(dockerfile)
    if len(installs) != 1:
        return True

    install = installs[0]
    if _REQUIREMENTS_FILE.findall(install) != [_ASR_AUDIO_LOCK_PATH]:
        return True
    if "--require-hashes" not in install or "--no-deps" not in install:
        return True

    return any(
        re.search(
            rf"(?<![A-Za-z0-9_.-]){re.escape(package)}"
            r"(?![A-Za-z0-9_.-])",
            install,
        )
        for package in _ASR_FORBIDDEN_RUNTIME_PACKAGES
    )


def _has_asr_forbidden_compiler_addition(dockerfile: str) -> bool:
    """Return whether a Dockerfile adds a compiler or CUDA build toolkit."""
    return any(
        _ASR_FORBIDDEN_COMPILER_ADDITION.search(instruction)
        for instruction in _logical_docker_instructions(dockerfile)
        if re.match(r"RUN\s+", instruction, flags=re.IGNORECASE)
    )


def _runtime_stage_instructions(dockerfile: str) -> list[str]:
    instructions = _logical_docker_instructions(dockerfile)
    runtime_start = next(
        index
        for index, instruction in enumerate(instructions)
        if re.fullmatch(r"FROM\s+.+\s+AS\s+runtime", instruction, flags=re.IGNORECASE)
    )
    runtime_end = next(
        (
            index
            for index, instruction in enumerate(
                instructions[runtime_start + 1 :], runtime_start + 1
            )
            if re.match(r"FROM\s+", instruction, flags=re.IGNORECASE)
        ),
        len(instructions),
    )
    return [
        instruction
        for instruction in instructions[runtime_start + 1 : runtime_end]
        if re.match(r"(?:RUN|COPY|ADD|ENV|CMD|ENTRYPOINT)\s+", instruction, flags=re.IGNORECASE)
    ]


_RUNTIME_TOOLING_ABSENCE_GATE = re.compile(
    r"^RUN\s+!\s+command\s+-v\s+nvcc\s+&&\s+!\s+command\s+-v\s+make\s+&&\s+!\s+"
    r"dpkg-query\s+-W\s+-f='\$\{db:Status-Status\}'\s+build-essential\s+2>/dev/null\s+"
    r"\|\s+grep\s+-qx\s+installed$"
)


def _runtime_stage_has_tooling_absence_gate(dockerfile: str) -> bool:
    return any(
        _RUNTIME_TOOLING_ABSENCE_GATE.fullmatch(instruction)
        for instruction in _runtime_stage_instructions(dockerfile)
    )


def _runtime_stage_uses_forbidden_tool(dockerfile: str) -> bool:
    return any(
        re.search(rf"(?<![A-Za-z0-9_.-]){tool}(?![A-Za-z0-9_.-])", instruction)
        for instruction in _runtime_stage_instructions(dockerfile)
        if not _RUNTIME_TOOLING_ABSENCE_GATE.fullmatch(instruction)
        for tool in ("nvcc", "make", "build-essential")
    )


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


def test_asr_audio_lock_rejects_decoder_drift_and_implicit_dependency_resolution() -> None:
    """Catch an audio decoder update or resolver expansion that changes the validated base image."""
    requirements_input = (ROOT / "dgx/asr/requirements-audio-arm64.in").read_text(encoding="utf-8")
    requirements_lock = (ROOT / "dgx/asr/requirements-audio-arm64.lock").read_text(encoding="utf-8")
    expected_headers = {"soundfile==0.13.1", "av==17.0.1"}

    assert requirements_input.splitlines() == ["soundfile==0.13.1", "av==17.0.1"], (
        "The ASR audio input must remain the two validated direct decoder distributions."
    )
    lock_headers = [
        re.sub(r"[-_.]+", "-", match.group(0).split("==", maxsplit=1)[0]).lower()
        + "=="
        + match.group(0).split("==", maxsplit=1)[1]
        for line in requirements_lock.splitlines()
        if (match := LOCK_ENTRY.match(line))
    ]
    assert len(lock_headers) == 2 and set(lock_headers) == expected_headers, (
        "The generated ASR audio lock must not introduce a transitive or replacement package."
    )

    lines = requirements_lock.splitlines()
    entry_indexes = [index for index, line in enumerate(lines) if LOCK_ENTRY.match(line)]
    for index, next_index in zip(entry_indexes, [*entry_indexes[1:], len(lines)], strict=True):
        header = lines[index].split(maxsplit=1)[0]
        package_block = "\n".join(
            line.split("#", maxsplit=1)[0] for line in lines[index:next_index]
        )
        hashes = re.findall(r"--hash=\S+", package_block)

        assert hashes, f"{header} must have a hash so a future build cannot resolve a new wheel."
        assert all(LOCK_HASH.fullmatch(value) for value in hashes), (
            f"{header} must use complete SHA-256 hashes so the selected wheel is reproducible."
        )


@pytest.mark.parametrize(
    "dockerfile",
    (
        "RUN pip3 \\\n    install vllm==0.26.1rc1",
        "RUN python3 -m pip \\\n    --quiet install triton==3.6.0",
        "RUN pip install --require-hashes --no-deps -r /tmp/requirements-audio-arm64.lock "
        "&& pip3.12 install vllm==0.26.1rc1",
        "RUN pip install --require-hashes -r /tmp/runtime-replacement.lock",
    ),
)
def test_asr_spark_image_install_guard_rejects_alternate_python_package_paths(
    dockerfile: str,
) -> None:
    """Catch an alternate pip path that can replace the inherited runtime."""
    assert _has_asr_invalid_python_package_install(dockerfile), (
        "Every non-audio Python package installation path must be rejected."
    )


def test_asr_spark_image_install_guard_allows_runtime_metadata_validation() -> None:
    """Allow validating inherited packages after the sole locked audio installation."""
    dockerfile = """RUN python3 -m pip install --no-cache-dir --require-hashes --no-deps \\
    -r /tmp/requirements-audio-arm64.lock \\
 && python3 -c "import cffi, vllm; print(cffi.__version__, vllm.__version__)"
"""

    assert not _has_asr_invalid_python_package_install(dockerfile), (
        "Metadata checks after the locked audio install must not be mistaken for "
        "package replacement."
    )


@pytest.mark.parametrize(
    "dockerfile",
    (
        "RUN apt-get update && apt-get install -y gcc g++ make build-essential cmake ninja",
        "RUN apt-get install -y cuda-toolkit-13-0",
        "RUN apt-get install -y cuda-nvcc-13-0",
        "RUN apt-get install -y cuda-compiler-13-0",
        "RUN apt-get install -y \\\n    nvidia-cuda-toolkit && /usr/local/cuda/bin/nvcc --version",
    ),
)
def test_asr_spark_image_rejects_compiler_and_cuda_toolkit_additions(dockerfile: str) -> None:
    """Catch build tooling that invalidates the fixed no-compiler Spark image contract."""
    assert _has_asr_forbidden_compiler_addition(dockerfile), (
        "The minimal ASR derivative must reject compiler and CUDA-toolkit additions."
    )


def test_asr_spark_image_has_no_compiler_or_cuda_toolkit_additions() -> None:
    """Catch the real ASR recipe gaining build tooling after its fixed base image is adopted."""
    dockerfile = (ROOT / "dgx/asr/Dockerfile").read_text(encoding="utf-8")

    assert not _has_asr_forbidden_compiler_addition(dockerfile), (
        "The ASR Dockerfile must not add a compiler or CUDA toolkit to the fixed Spark base."
    )


def test_asr_spark_image_rejects_core_stack_reinstallation() -> None:
    """Catch a pip install that replaces the Torch/vLLM runtime proven on the Spark base image."""
    dockerfile = (ROOT / "dgx/asr/Dockerfile").read_text(encoding="utf-8")
    logical_instructions = _logical_docker_instructions(dockerfile)
    assert [
        instruction
        for instruction in logical_instructions
        if re.match(r"(?:ARG|FROM)\s+", instruction)
    ] == [
        "ARG SPARK_BASE=dgx-spark-vllm:midpoint-v023",
        "FROM ${SPARK_BASE}",
    ], "The ASR derivative must inherit exactly the repository-built midpoint Spark-vLLM base."
    assert not _has_asr_invalid_python_package_install(dockerfile), (
        "The ASR image may install only the direct audio lock and must not replace "
        "inherited runtime dependencies."
    )
    assert not _has_asr_forbidden_compiler_addition(dockerfile), (
        "The ASR derivative must not add a compiler or CUDA build toolkit."
    )


def test_asr_midpoint_build_pins_complete_historical_stack() -> None:
    script = (ROOT / "dgx/asr/build-midpoint-base.sh").read_text(encoding="utf-8")
    patch = (ROOT / "dgx/asr/eugr-midpoint.patch").read_text(encoding="utf-8")

    for value in (
        "b51af15a280d28c2ad9096b3ef581524eddbd0e7",
        "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "d768c14e7cf5dd5df45a8a1de78ae815879f108a",
        "6da422082f910a8dd230f7e42e26ece4dc37bccc",
        "2026-06-18T23:59:59Z",
        "sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6",
        "sha256:450d11555d20ac8ebbbc13ebf17589c2bd42869171a90179ce7098b4a5e64c6a",
    ):
        assert value in script or value in patch
    assert "transformers==5.12.1" in patch
    assert "VLLM_PRS" not in script
    assert "FLASHINFER_PRS" not in script


def test_asr_midpoint_build_asserts_arm64_manifest_and_label_provenance() -> None:
    script = (ROOT / "dgx/asr/build-midpoint-base.sh").read_text(encoding="utf-8")

    assert "docker buildx imagetools inspect --raw" in script
    assert "docker buildx imagetools inspect --format '{{json .Image}}'" in script
    assert "CUDA_ARM64_MANIFEST" in script
    assert "RootFS.Layers" in script
    assert "source_layers" in script
    assert 'source_image["config"].get("Labels")' in script
    assert "image_labels == source_labels" in script
    assert 'test "$labels" = null' not in script


def test_asr_midpoint_build_normalizes_exact_vllm_distribution_version() -> None:
    patch = (ROOT / "dgx/asr/eugr-midpoint.patch").read_text(encoding="utf-8")

    assert "VLLM_VERSION_OVERRIDE=0.23.0 uv build" in patch, (
        "The pinned post-release vLLM commit otherwise creates a date-dependent dev wheel."
    )


def test_asr_midpoint_build_uses_exact_refs_as_deterministic_cache_keys() -> None:
    script = (ROOT / "dgx/asr/build-midpoint-base.sh").read_text(encoding="utf-8")
    patch = (ROOT / "dgx/asr/eugr-midpoint.patch").read_text(encoding="utf-8")
    added_lines = {
        line[1:].strip()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    }

    assert 'FI_CMD+=("--build-arg" "CACHEBUST_FLASHINFER=$FLASHINFER_REF")' in added_lines
    assert 'VLLM_CMD+=("--build-arg" "CACHEBUST_VLLM=$VLLM_REF")' in added_lines
    assert not any("CACHEBUST_" in line and "date +%s" in line for line in added_lines)
    assert "expected_upstream_changes=$'Dockerfile\\nbuild-and-copy.sh'" in script


def test_asr_midpoint_build_exempts_only_exact_pytorch_stack_from_cutoff() -> None:
    patch = (ROOT / "dgx/asr/eugr-midpoint.patch").read_text(encoding="utf-8")
    exact_install = (
        "env -u UV_EXCLUDE_NEWER uv pip install torch==2.11.0 torchvision==0.26.0 "
        "torchaudio==2.11.0 triton==3.6.0 "
        "--index-url https://download.pytorch.org/whl/cu130"
    )

    assert patch.count(exact_install) == 2, (
        "The PyTorch index omits upload dates, so only its exact builder and runner stack "
        "may override the historical package cutoff."
    )


def test_asr_midpoint_runtime_asserts_adapter_and_versions() -> None:
    dockerfile = (ROOT / "dgx/asr/Dockerfile").read_text(encoding="utf-8")
    assert "ARG SPARK_BASE=dgx-spark-vllm:midpoint-v023" in dockerfile
    assert "0.23.0" in dockerfile
    assert "2.11.0+cu130" in dockerfile
    assert "5.12.1" in dockerfile
    assert "0.6.12" in dockerfile
    assert "e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d" in dockerfile


def test_asr_spark_image_build_preserves_existing_service_contract() -> None:
    """Catch a mutable Compose image or changed running ASR service."""
    service = _compose()["services"]["qwen3-asr"]
    environment = set(service["environment"])
    command = [str(value) for value in service["command"]]
    health_command = " ".join(service["healthcheck"]["test"])

    assert service["build"] == {"context": ".", "dockerfile": "asr/Dockerfile"}, (
        "Compose must build the repository-owned ASR derivative rather than "
        "pulling an external image."
    )
    assert "image" not in service, (
        "A mutable external ASR image tag would bypass the locked derivative."
    )
    assert service["container_name"] == "qwen3-asr", (
        "The stable ASR service identity must not change."
    )
    assert service["deploy"]["resources"]["reservations"]["devices"] == [
        {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}
    ], "The ASR service must retain its GPU reservation."
    assert service["networks"] == ["default", "shared_ai_voice"], (
        "The ASR service must retain both its internal and proxy networks."
    )
    assert service["shm_size"] == "4gb", "The ASR image must retain its shared-memory allocation."
    assert service["restart"] == "unless-stopped", (
        "The ASR service must retain its recovery policy."
    )
    assert service["labels"] == ["autoheal=true"], (
        "The ASR service must remain eligible for autoheal."
    )
    assert {"HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"} <= environment, (
        "The ASR service must continue using the preloaded model cache offline."
    )
    assert ASR_REVISION in " ".join(command), "The ASR model snapshot revision must remain pinned."
    assert command[command.index("--served-model-name") + 1] == "qwen3-asr", (
        "The ASR OpenAI model identifier must stay stable."
    )
    assert "/v1/models" in health_command and "qwen3-asr" in health_command, (
        "The ASR health check must continue validating the loaded served model."
    )


def test_asr_spark_image_is_unpatched() -> None:
    """Ensure the image inherits vLLM's additive transcription response unchanged."""
    dockerfile = (ROOT / "dgx/asr/Dockerfile").read_text(encoding="utf-8")

    assert "patch_vllm_transcription_contract.py" not in dockerfile
    assert "TranscriptionResponse" not in dockerfile
    assert not (ROOT / "dgx/asr/patch_vllm_transcription_contract.py").exists()


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
    runtime_apt_packages = (ROOT / "dgx/tts/apt-runtime-packages-arm64.lock").read_text(
        encoding="utf-8"
    )

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
    runtime_input_headers = _normalized_lock_headers(runtime_input)
    runtime_lock_headers = _normalized_lock_headers(requirements)
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
        assert input_dependency in runtime_input_headers
        assert resolved_dependency in runtime_lock_headers

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
    excluded_runtime_packages = {
        "vllm",
        "ray",
        "flashinfer",
        "gradio",
        "hf-gradio",
        "torchaudio",
    }
    assert not excluded_runtime_packages & {
        _normalized_package_name(header) for header in runtime_lock_headers
    }
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
    assert runtime_apt_packages.splitlines() == [
        "python3=3.14.3-0ubuntu2",
        "gcc-15=15.2.0-16ubuntu1",
        "libc6-dev=2.43-2ubuntu2.3",
        "libsndfile1=1.2.2-4",
        "libgomp1=16-20260322-1ubuntu1",
        "sox=14.7.0.9+ds1-1",
        "libsox-fmt-base=14.7.0.9+ds1-1",
    ]
    assert "COPY --from=builder /usr/include/python3.14 /usr/include/python3.14" in dockerfile
    assert "COPY --from=builder /usr/include/aarch64-linux-gnu/python3.14" in dockerfile
    assert "CC=gcc-15" in dockerfile
    assert not _runtime_stage_uses_forbidden_tool(dockerfile)
    assert _runtime_stage_has_tooling_absence_gate(dockerfile)


def test_tts_runtime_stage_rejects_forbidden_tooling_instructions() -> None:
    builder_only = """FROM base AS builder
RUN apt-get install -y build-essential make
FROM base AS runtime
RUN echo runtime-ready
"""
    assert not _runtime_stage_uses_forbidden_tool(builder_only)

    later_debug_stage = """FROM base AS builder
FROM base AS runtime
RUN echo runtime-ready
FROM base AS debug
RUN make docs
"""
    assert not _runtime_stage_uses_forbidden_tool(later_debug_stage)

    forbidden_runtime_fixtures = (
        """FROM base AS builder
FROM base AS runtime
RUN apt-get install -y build-essential
""",
        """FROM base AS builder
FROM base AS runtime
RUN /usr/bin/make all
""",
        """FROM base AS builder
FROM base AS runtime
COPY --from=builder /usr/local/cuda/bin/nvcc /usr/local/bin/nvcc
""",
    )
    assert all(
        _runtime_stage_uses_forbidden_tool(fixture) for fixture in forbidden_runtime_fixtures
    )


def test_tts_image_extracts_pure_kaldi_compat_without_installing_torchaudio() -> None:
    """Catch a CUDA 13.2 Torch image silently gaining an ABI-mismatched TorchAudio wheel."""
    dockerfile = (ROOT / "dgx/tts/Dockerfile").read_text(encoding="utf-8")
    compat_lock = (ROOT / "dgx/tts/torchaudio-kaldi-compat-arm64.lock").read_text(encoding="utf-8")

    assert (
        "torchaudio==2.9.1 "
        "--hash=sha256:9c0d004f784c49078017f8217fdc901df0eb9724e50fb269b3a6c99b1d4eae75"
        in compat_lock
    )
    assert (
        "COPY tts/torchaudio-kaldi-compat-arm64.lock "
        "/tmp/torchaudio-kaldi-compat-arm64.lock" in dockerfile
    )
    assert "pip download --no-cache-dir --require-hashes --no-deps" in dockerfile
    assert "torchaudio/compliance/kaldi.py" in dockerfile
    assert "torchaudio-2.9.1.dist-info/LICENSE" in dockerfile
    assert "kaldi_compat.py" in dockerfile
    assert "BSD-2-Clause" in dockerfile
    assert "SPDX-License-Identifier: BSD-2-Clause\\\\n# Full license:" in dockerfile
    assert "only fbank is supported" in dockerfile
    assert "/usr/share/licenses/torchaudio-kaldi-compat/LICENSE" in dockerfile
    assert "COPY --from=builder /opt/tts-licenses /usr/share/licenses" in dockerfile
    assert "importlib.util.find_spec(" in dockerfile
    assert "import qwen_tts; print" not in dockerfile
    assert "importlib.util.find_spec('torchaudio') is None" not in dockerfile

    logical_dockerfile = "\n".join(_logical_docker_instructions(dockerfile))
    assert re.search(
        r"\bpip\s+download\b.*torchaudio-kaldi-compat-arm64\.lock",
        logical_dockerfile,
    )
    assert not _has_forbidden_torchaudio_pip_install(dockerfile)

    forbidden_install_fixtures = (
        "RUN pip install torchaudio==2.9.1",
        "RUN pip \\" + "\n    install torchaudio==2.9.1",
        "RUN pip \\  " + "\n    install torchaudio==2.9.1",
        "RUN pip \\\t" + "\n    install torchaudio==2.9.1",
    )
    assert all(
        _has_forbidden_torchaudio_pip_install(fixture) for fixture in forbidden_install_fixtures
    )
    assert not _has_forbidden_torchaudio_pip_install("RUN pip download torchaudio==2.9.1")


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
    tampered_empty_runtime_hash = requirements.replace(
        "    --hash=sha256:70988c352feb481887077d2ab845125024b2a137a5090d6d7a32b57d03a45df6",
        "    --hash=",
        1,
    )
    comment_only_multiline_hash = "demo==1.0\n # --hash=sha256:" + "0" * 64
    comment_only_inline_hash = "demo==1.0 # --hash=sha256:" + "0" * 64

    for tampered in (
        tampered_runtime,
        tampered_qwen,
        tampered_empty_runtime_hash,
        comment_only_multiline_hash,
        comment_only_inline_hash,
    ):
        with pytest.raises(AssertionError):
            _assert_hash_locked(tampered)

    assert _normalized_package_name("vllm[foo]==1.0") == "vllm"
