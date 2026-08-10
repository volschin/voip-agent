#!/usr/bin/env python3
"""Fail-closed verification for the isolated FlashInfer wheel candidate."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

FLASHINFER_DISTRIBUTIONS = frozenset(
    {"flashinfer-cubin", "flashinfer-jit-cache", "flashinfer-python"}
)
EXPECTED_BASE_IMAGE_ID = "sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930"
EXPECTED_ADAPTER_SHA256 = "e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d"
EXPECTED_RELEASE_ID = 367461871
EXPECTED_RELEASE_ASSETS = frozenset(
    {
        (507452716, "flashinfer_python-0.6.18-py3-none-any.whl", 17122160),
        (
            507452715,
            "flashinfer_jit_cache-0.6.18-cp39-abi3-manylinux_2_28_aarch64.whl",
            252992614,
        ),
        (507452717, "flashinfer_cubin-0.6.18-py3-none-any.whl", 1239178852),
    }
)
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
    distributions: dict[str, list[str]] = collections.defaultdict(list)
    for entry in inventory["distributions"]:
        name = _normalized_name(entry["name"])
        distributions[name].append(entry["version"])
    return {name: tuple(sorted(versions)) for name, versions in distributions.items()}


def verify_images(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    _require(base["image_id"] == EXPECTED_BASE_IMAGE_ID, "unexpected baseline image ID")
    _require(candidate["image_id"] != base["image_id"], "candidate did not add an image")
    _require(
        base["architecture"] == candidate["architecture"] == "arm64",
        "candidate architecture changed",
    )
    _require(base["os"] == candidate["os"] == "linux", "candidate operating system changed")
    _require(
        candidate["rootfs_layers"][: len(base["rootfs_layers"])] == base["rootfs_layers"],
        "candidate does not inherit the exact baseline rootfs",
    )
    _require(
        len(candidate["rootfs_layers"]) == len(base["rootfs_layers"]) + 1,
        "candidate must append exactly one rootfs layer",
    )
    _require(candidate["config"] == base["config"], "candidate image configuration changed")
    _require(base["adapter_sha256"] == EXPECTED_ADAPTER_SHA256, "baseline adapter changed")
    _require(
        candidate["adapter_sha256"] == EXPECTED_ADAPTER_SHA256,
        "candidate adapter changed",
    )

    base_distributions = _distribution_map(base)
    candidate_distributions = _distribution_map(candidate)
    base_flashinfer = {name: base_distributions.pop(name, ()) for name in FLASHINFER_DISTRIBUTIONS}
    candidate_flashinfer = {
        name: candidate_distributions.pop(name, ()) for name in FLASHINFER_DISTRIBUTIONS
    }
    _require(
        base_flashinfer == dict.fromkeys(FLASHINFER_DISTRIBUTIONS, ("0.6.12",)),
        "baseline FlashInfer inventory changed",
    )
    _require(
        candidate_flashinfer == dict.fromkeys(FLASHINFER_DISTRIBUTIONS, ("0.6.18",)),
        "candidate FlashInfer inventory is not exactly 0.6.18",
    )
    _require(
        candidate_distributions == base_distributions,
        "a non-FlashInfer distribution changed",
    )


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    _require(path.stat().st_size == expected_size, f"unexpected size: {path.name}")
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    _require(actual_sha256 == expected_sha256, f"unexpected SHA-256: {path.name}")


def verify_release(path: Path) -> None:
    release = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(release, dict), "release metadata must be an object")
    _require(release.get("id") == EXPECTED_RELEASE_ID, "unexpected release ID")
    assets = release.get("assets")
    _require(isinstance(assets, list), "release assets must be a list")
    actual_assets = frozenset(
        (asset["id"], asset["name"], asset["size"]) for asset in assets if isinstance(asset, dict)
    )
    _require(actual_assets == EXPECTED_RELEASE_ASSETS, "release asset metadata changed")


def capture_image(image: str, output: Path) -> None:
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    inspect_values = json.loads(inspect_result.stdout)
    _require(
        isinstance(inspect_values, list) and len(inspect_values) == 1,
        "docker inspect must return exactly one image",
    )
    inspect_value = inspect_values[0]
    runtime_result = subprocess.run(
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
            "-c",
            _RUNTIME_INVENTORY_CODE,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime_value = json.loads(runtime_result.stdout)
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
    verify_parser = subparsers.add_parser("verify-images")
    verify_parser.add_argument("base", type=Path)
    verify_parser.add_argument("candidate", type=Path)
    file_parser = subparsers.add_parser("verify-file")
    file_parser.add_argument("path", type=Path)
    file_parser.add_argument("expected_size", type=int)
    file_parser.add_argument("expected_sha256")
    release_parser = subparsers.add_parser("verify-release")
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
        verify_release(args.path)
    elif args.command == "capture-image":
        capture_image(args.image, args.output)


if __name__ == "__main__":
    main()
