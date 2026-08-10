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
EXPECTED_ADAPTER_SHA256 = "e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d"
_RUNTIME_INVENTORY_CODE = r"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

vllm_distribution = metadata.distribution("vllm")
adapter = Path(vllm_distribution.locate_file("vllm/model_executor/models/qwen3_asr.py"))
assert adapter.is_file()
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


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_map(inventory: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    distributions: dict[str, list[str]] = collections.defaultdict(list)
    for entry in inventory["distributions"]:
        name = _normalized_name(entry["name"])
        distributions[name].append(entry["version"])
    return {name: tuple(sorted(versions)) for name, versions in distributions.items()}


def verify_images(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    assert base["architecture"] == candidate["architecture"] == "arm64"
    assert base["os"] == candidate["os"] == "linux"
    assert candidate["rootfs_layers"][: len(base["rootfs_layers"])] == base["rootfs_layers"]
    assert len(candidate["rootfs_layers"]) > len(base["rootfs_layers"])
    assert candidate["config"] == base["config"]
    assert base["adapter_sha256"] == EXPECTED_ADAPTER_SHA256
    assert candidate["adapter_sha256"] == EXPECTED_ADAPTER_SHA256

    base_distributions = _distribution_map(base)
    candidate_distributions = _distribution_map(candidate)
    assert {
        name: base_distributions.pop(name) for name in FLASHINFER_DISTRIBUTIONS
    } == dict.fromkeys(FLASHINFER_DISTRIBUTIONS, ("0.6.12",))
    assert {
        name: candidate_distributions.pop(name) for name in FLASHINFER_DISTRIBUTIONS
    } == dict.fromkeys(FLASHINFER_DISTRIBUTIONS, ("0.6.18",))
    assert candidate_distributions == base_distributions


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    assert path.stat().st_size == expected_size, f"unexpected size: {path.name}"
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual_sha256 == expected_sha256, f"unexpected SHA-256: {path.name}"


def capture_image(image: str, output: Path) -> None:
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    inspect_values = json.loads(inspect_result.stdout)
    assert isinstance(inspect_values, list) and len(inspect_values) == 1
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
    assert isinstance(value, dict), f"inventory must be an object: {path.name}"
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
    capture_parser = subparsers.add_parser("capture-image")
    capture_parser.add_argument("image")
    capture_parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.command == "verify-images":
        verify_images(_load_inventory(args.base), _load_inventory(args.candidate))
    elif args.command == "verify-file":
        verify_file(args.path, args.expected_size, args.expected_sha256)
    elif args.command == "capture-image":
        capture_image(args.image, args.output)


if __name__ == "__main__":
    main()
