#!/usr/bin/env bash
set -euo pipefail

EUGR_COMMIT=b51af15a280d28c2ad9096b3ef581524eddbd0e7
VLLM_COMMIT=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
FLASHINFER_COMMIT=d768c14e7cf5dd5df45a8a1de78ae815879f108a
NCCL_COMMIT=6da422082f910a8dd230f7e42e26ece4dc37bccc
DEPENDENCY_CUTOFF=2026-06-18T23:59:59Z
CUDA_IMAGE_DIGEST=sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
CUDA_ARM64_MANIFEST=sha256:450d11555d20ac8ebbbc13ebf17589c2bd42869171a90179ce7098b4a5e64c6a
BASE_TAG=dgx-spark-vllm:midpoint-v023
FINAL_TAG=dgx-qwen3-asr:spark-midpoint-v023-test

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(mktemp -d)
trap 'rm -rf -- "$source_root"' EXIT

assert_cuda_arm64_manifest() {
  docker buildx imagetools inspect --raw "nvidia/cuda:13.0.2-devel-ubuntu24.04@$CUDA_IMAGE_DIGEST" | \
    python3 -c '
import json
import sys

expected_digest = sys.argv[1]
manifest = json.load(sys.stdin)
matches = [
    entry
    for entry in manifest.get("manifests", [])
    if entry.get("digest") == expected_digest
    and entry.get("platform", {}).get("architecture") == "arm64"
    and entry.get("platform", {}).get("os") == "linux"
]
assert len(matches) == 1, matches
' "$CUDA_ARM64_MANIFEST"
}

assert_image_inventory() {
  local image_tag=$1
  local image_id architecture labels environment source_image image_layers

  image_id=$(docker image inspect --format '{{.Id}}' "$image_tag")
  architecture=$(docker image inspect --format '{{.Architecture}}' "$image_tag")
  labels=$(docker image inspect --format '{{json .Config.Labels}}' "$image_tag")
  environment=$(docker image inspect --format '{{json .Config.Env}}' "$image_tag")
  test -n "$image_id"
  test "$architecture" = arm64
  assert_cuda_arm64_manifest
  source_image=$(docker buildx imagetools inspect --format '{{json .Image}}' \
    "nvidia/cuda:13.0.2-devel-ubuntu24.04@$CUDA_ARM64_MANIFEST")
  image_layers=$(docker image inspect --format '{{json .RootFS.Layers}}' "$image_tag")
  python3 - "$image_tag" "$source_image" "$labels" "$image_layers" <<'PY'
import json
import sys

image_tag, source_image_json, image_labels_json, image_layers_json = sys.argv[1:]
source_image = json.loads(source_image_json)
source_labels = source_image["config"].get("Labels")
image_labels = json.loads(image_labels_json)
assert image_labels == source_labels, (image_tag, image_labels, source_labels)
source_layers = source_image["rootfs"]["diff_ids"]
image_layers = json.loads(image_layers_json)
assert source_layers
assert image_layers[: len(source_layers)] == source_layers, image_tag
PY
  case "$environment" in
    *"UV_EXCLUDE_NEWER=$DEPENDENCY_CUTOFF"*) ;;
    *) printf '%s is missing UV_EXCLUDE_NEWER=%s\n' "$image_tag" "$DEPENDENCY_CUTOFF" >&2; exit 1 ;;
  esac
  printf '%s image: id=%s architecture=%s labels=%s\n' \
    "$image_tag" "$image_id" "$architecture" "$labels"
}

assert_historical_inventory() {
  local image_tag=$1

  assert_image_inventory "$image_tag"
  python3 - "$image_tag" 3< <(docker image save "$image_tag") <<'PY'
import hashlib
import os
import tarfile
import sys

image_tag = sys.argv[1]
expected_versions = {
    "vllm": "0.23.0",
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "torchaudio": "2.11.0+cu130",
    "transformers": "5.12.1",
    "flashinfer-python": "0.6.12",
}
expected_adapter_sha256 = "e233961d38d0a396db34cf2f7d83c6dc1c33aa55768ba894eee6de097120342d"
expected_metadata = (
    "build_script_commit: b51af15a280d28c2ad9096b3ef581524eddbd0e7",
    "vllm_commit: 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
    "flashinfer_commit: d768c14e7cf5dd5df45a8a1de78ae815879f108a",
    "base_image: nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6",
)

versions = {}
adapter_sha256 = None
build_metadata = None
with os.fdopen(3, "rb") as image_stream:
    with tarfile.open(fileobj=image_stream, mode="r|") as image_archive:
        for image_member in image_archive:
            if not image_member.isfile() or not image_member.name.endswith("/layer.tar"):
                continue
            layer_stream = image_archive.extractfile(image_member)
            assert layer_stream is not None
            with tarfile.open(fileobj=layer_stream, mode="r|") as layer:
                for member in layer:
                    if not member.isfile():
                        continue
                    path = member.name.lstrip("./")
                    if path.endswith("vllm/model_executor/models/qwen3_asr.py"):
                        content = layer.extractfile(member)
                        assert content is not None
                        adapter_sha256 = hashlib.file_digest(content, "sha256").hexdigest()
                    elif path.endswith(".dist-info/METADATA"):
                        content = layer.extractfile(member)
                        assert content is not None
                        headers = {}
                        for line in content.read().decode("utf-8").splitlines():
                            if ": " in line:
                                key, value = line.split(": ", 1)
                                if key in {"Name", "Version"}:
                                    headers[key] = value
                        name = headers.get("Name", "").lower()
                        if name in expected_versions:
                            versions[name] = headers.get("Version")
                    elif path == "workspace/build-metadata.yaml":
                        content = layer.extractfile(member)
                        assert content is not None
                        build_metadata = content.read().decode("utf-8")

assert adapter_sha256 == expected_adapter_sha256, (image_tag, adapter_sha256)
assert versions == expected_versions, (image_tag, versions)
assert build_metadata is not None, (image_tag, "missing build metadata")
assert all(value in build_metadata for value in expected_metadata), (image_tag, build_metadata)
print(f"{image_tag} runtime: " + " ".join(f"{name}={version}" for name, version in sorted(versions.items())))
print(f"{image_tag} adapter: vllm/model_executor/models/qwen3_asr.py sha256={adapter_sha256}")
print(f"{image_tag} build metadata: verified")
PY
  printf '%s build metadata: eugr=%s vllm=%s flashinfer=%s nccl=%s cutoff=%s cuda=%s arm64-manifest=%s\n' \
    "$image_tag" "$EUGR_COMMIT" "$VLLM_COMMIT" "$FLASHINFER_COMMIT" "$NCCL_COMMIT" \
    "$DEPENDENCY_CUTOFF" "$CUDA_IMAGE_DIGEST" "$CUDA_ARM64_MANIFEST"
  printf '%s historical eugr source adjustments: flashinfer_cache.patch; FlashInfer license metadata normalization; AutoGPTQ symmetric-MoE workaround; MiniMax QK RMSNorm workaround; build-requirement substitutions\n' \
    "$image_tag"
  printf '%s repository-added source patches: none\n' "$image_tag"
}

assert_cuda_arm64_manifest
git clone --filter=blob:none https://github.com/eugr/spark-vllm-docker.git "$source_root/eugr"
git -C "$source_root/eugr" checkout --detach "$EUGR_COMMIT"
git -C "$source_root/eugr" apply --check --unidiff-zero "$script_dir/eugr-midpoint.patch"
git -C "$source_root/eugr" apply --unidiff-zero "$script_dir/eugr-midpoint.patch"
test "$(git -C "$source_root/eugr" diff --name-only)" = Dockerfile

(
  cd "$source_root/eugr"
  ./build-and-copy.sh --tag "$BASE_TAG" --gpu-arch 12.1a --build-jobs 4 \
    --tf5 --rebuild-vllm --vllm-ref "$VLLM_COMMIT" \
    --rebuild-flashinfer --flashinfer-ref "$FLASHINFER_COMMIT"
)
assert_historical_inventory "$BASE_TAG"

docker build --pull=false --build-arg "SPARK_BASE=$BASE_TAG" \
  --tag "$FINAL_TAG" --file "$script_dir/Dockerfile" "$script_dir/.."
assert_historical_inventory "$FINAL_TAG"
