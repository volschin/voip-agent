#!/usr/bin/env python3
"""Fail-closed verification for the isolated vLLM 0.24 wheel candidate."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_BASE_IMAGE_ID = "sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930"
EXPECTED_BASE_ADAPTER_SHA256 = "e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d"
EXPECTED_CANDIDATE_ADAPTER_SHA256 = (
    "639d3691fae9195ed38e17306a29b04bc60025e1119d0090443ec7d935eceffd"
)
CHANGED_DISTRIBUTIONS = frozenset({"vllm", "humming-kernels"})
EXPECTED_RELEASES = {
    "vllm": {
        "name": "vllm",
        "version": "0.24.0",
        "requires_python": "<3.15,>=3.10",
        "filename": "vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl",
        "url": "https://files.pythonhosted.org/packages/9e/80/51a071305b4eed0f6f512dc1c1c6957cbb14ccce38db1be90ffcff2a2844/vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl",
        "size": 271361241,
        "sha256": "700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5",
    },
    "humming-kernels": {
        "name": "humming-kernels",
        "version": "0.1.6",
        "requires_python": ">=3.10",
        "filename": "humming_kernels-0.1.6-py3-none-any.whl",
        "url": "https://files.pythonhosted.org/packages/25/85/490681b9ba24531da91d0bae801d2b26850e5a80bbd02c2efc500756e36b/humming_kernels-0.1.6-py3-none-any.whl",
        "size": 178759,
        "sha256": "e64c0883fca930074bf920f4ba47cbf3acd244d7352f6c74c8d2182439770d8f",
    },
}
EXPECTED_CHANGED_VERSIONS = {
    "base": {"vllm": ("0.23.0",), "humming-kernels": ("0.1.4",)},
    "candidate": {"vllm": ("0.24.0",), "humming-kernels": ("0.1.6",)},
}
_RUNTIME_INVENTORY_CODE = r"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

vllm_distribution = metadata.distribution("vllm")
adapter = Path(vllm_distribution.locate_file("vllm/model_executor/models/qwen3_asr.py"))
if not adapter.is_file():
    raise FileNotFoundError("vLLM Qwen3-ASR adapter is missing")
distributions = sorted(
    (
        {"name": distribution.metadata.get("Name", ""), "version": distribution.version}
        for distribution in metadata.distributions()
    ),
    key=lambda value: (value["name"].lower(), value["version"]),
)
print(json.dumps({
    "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
    "distributions": distributions,
}, sort_keys=True))
"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_map(inventory: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    entries = inventory.get("distributions")
    _require(isinstance(entries, list), "distribution inventory must be a list")
    distributions: dict[str, list[str]] = collections.defaultdict(list)
    for entry in entries:
        _require(isinstance(entry, dict), "distribution entry must be an object")
        name = entry.get("name")
        version = entry.get("version")
        _require(type(name) is str and type(version) is str, "invalid distribution entry")
        distributions[_normalized_name(name)].append(version)
    return {name: tuple(sorted(versions)) for name, versions in distributions.items()}


def verify_images(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    _require(base.get("image_id") == EXPECTED_BASE_IMAGE_ID, "unexpected baseline image ID")
    _require(candidate.get("image_id") != base.get("image_id"), "candidate did not add an image")
    _require(
        base.get("architecture") == candidate.get("architecture") == "arm64",
        "candidate architecture changed",
    )
    _require(base.get("os") == candidate.get("os") == "linux", "operating system changed")
    base_layers = base.get("rootfs_layers")
    candidate_layers = candidate.get("rootfs_layers")
    _require(
        isinstance(base_layers, list) and len(base_layers) == 21, "baseline layer count changed"
    )
    _require(isinstance(candidate_layers, list), "candidate layers must be a list")
    _require(candidate_layers[:21] == base_layers, "candidate rootfs prefix changed")
    _require(len(candidate_layers) == 22, "candidate must append exactly one rootfs layer")
    _require(candidate.get("config") == base.get("config"), "candidate image configuration changed")
    _require(
        base.get("adapter_sha256") == EXPECTED_BASE_ADAPTER_SHA256,
        "baseline adapter changed",
    )
    _require(
        candidate.get("adapter_sha256") == EXPECTED_CANDIDATE_ADAPTER_SHA256,
        "candidate adapter is not the official vLLM 0.24 adapter",
    )

    base_distributions = _distribution_map(base)
    candidate_distributions = _distribution_map(candidate)
    base_changed = {name: base_distributions.pop(name, ()) for name in CHANGED_DISTRIBUTIONS}
    candidate_changed = {
        name: candidate_distributions.pop(name, ()) for name in CHANGED_DISTRIBUTIONS
    }
    _require(
        base_changed == EXPECTED_CHANGED_VERSIONS["base"],
        "baseline changed-package inventory is unexpected",
    )
    _require(
        candidate_changed == EXPECTED_CHANGED_VERSIONS["candidate"],
        "candidate changed-package inventory is unexpected",
    )
    _require(candidate_distributions == base_distributions, "an unrelated distribution changed")


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    _require(path.is_file(), f"missing file: {path.name}")
    _require(path.stat().st_size == expected_size, f"unexpected size: {path.name}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    _require(digest == expected_sha256, f"unexpected SHA-256: {path.name}")


def verify_release(package: str, path: Path) -> None:
    _require(package in EXPECTED_RELEASES, "unsupported release package")
    expected = EXPECTED_RELEASES[package]
    release = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(release, dict), "release metadata must be an object")
    info = release.get("info")
    urls = release.get("urls")
    _require(isinstance(info, dict), "release info must be an object")
    _require(isinstance(urls, list), "release URLs must be a list")
    _require(info.get("name") == expected["name"], "unexpected package name")
    _require(info.get("version") == expected["version"], "unexpected package version")
    _require(
        info.get("requires_python") == expected["requires_python"],
        "unexpected release Python requirement",
    )
    selected = [
        item
        for item in urls
        if isinstance(item, dict) and item.get("filename") == expected["filename"]
    ]
    _require(len(selected) == 1, "expected wheel must occur exactly once")
    asset = selected[0]
    _require(
        type(asset.get("url")) is str and asset["url"] == expected["url"], "unexpected wheel URL"
    )
    _require(
        type(asset.get("size")) is int and asset["size"] == expected["size"],
        "unexpected wheel size",
    )
    digests = asset.get("digests")
    _require(isinstance(digests, dict), "wheel digests must be an object")
    _require(digests.get("sha256") == expected["sha256"], "unexpected wheel SHA-256")
    _require(
        asset.get("requires_python") == expected["requires_python"],
        "unexpected wheel Python requirement",
    )
    _require(asset.get("yanked") is False, "wheel must not be yanked")


def _isolated_docker_run(image: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--entrypoint",
            "python3",
            image,
            *command,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def capture_image(image: str, output: Path) -> None:
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", image], check=True, capture_output=True, text=True
    )
    inspect_values = json.loads(inspect_result.stdout)
    _require(
        isinstance(inspect_values, list) and len(inspect_values) == 1,
        "docker inspect must return exactly one image",
    )
    runtime_value = json.loads(_isolated_docker_run(image, ["-c", _RUNTIME_INVENTORY_CODE]).stdout)
    _isolated_docker_run(image, ["-m", "pip", "check"])
    inspect_value = inspect_values[0]
    inventory = {
        "image_id": inspect_value["Id"],
        "architecture": inspect_value["Architecture"],
        "os": inspect_value["Os"],
        "rootfs_layers": inspect_value["RootFS"]["Layers"],
        "config": inspect_value["Config"],
        "adapter_sha256": runtime_value["adapter_sha256"],
        "distributions": runtime_value["distributions"],
    }
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_inventory(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"inventory must be an object: {path.name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    images_parser = subparsers.add_parser("verify-images")
    images_parser.add_argument("base", type=Path)
    images_parser.add_argument("candidate", type=Path)
    file_parser = subparsers.add_parser("verify-file")
    file_parser.add_argument("path", type=Path)
    file_parser.add_argument("expected_size", type=int)
    file_parser.add_argument("expected_sha256")
    release_parser = subparsers.add_parser("verify-release")
    release_parser.add_argument("package", choices=sorted(EXPECTED_RELEASES))
    release_parser.add_argument("path", type=Path)
    capture_parser = subparsers.add_parser("capture-image")
    capture_parser.add_argument("image")
    capture_parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.command == "verify-images":
        verify_images(_load_inventory(args.base), _load_inventory(args.candidate))
    elif args.command == "verify-file":
        verify_file(args.path, args.expected_size, args.expected_sha256)
    elif args.command == "verify-release":
        verify_release(args.package, args.path)
    elif args.command == "capture-image":
        capture_image(args.image, args.output)


if __name__ == "__main__":
    main()
