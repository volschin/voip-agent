#!/usr/bin/env bash
set -euo pipefail

QUALIFIED_BASE_TAG=dgx-qwen3-asr:vllm023-615e858c
QUALIFIED_BASE_IMAGE_ID=sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930
CANDIDATE_TAG=dgx-qwen3-asr:vllm024-pypi-test
ASSETS=(
  "vllm|0.24.0|vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl|271361241|700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5|https://files.pythonhosted.org/packages/9e/80/51a071305b4eed0f6f512dc1c1c6957cbb14ccce38db1be90ffcff2a2844/vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl|<3.15,>=3.10"
  "humming-kernels|0.1.6|humming_kernels-0.1.6-py3-none-any.whl|178759|e64c0883fca930074bf920f4ba47cbf3acd244d7352f6c74c8d2182439770d8f|https://files.pythonhosted.org/packages/25/85/490681b9ba24531da91d0bae801d2b26850e5a80bbd02c2efc500756e36b/humming_kernels-0.1.6-py3-none-any.whl|>=3.10"
)

usage() {
  printf 'Usage: %s\nBuild the isolated vLLM 0.24 wheel candidate.\n' "${0##*/}"
}

if test "$#" -gt 1; then
  printf 'unexpected arguments\n' >&2
  usage >&2
  exit 2
fi
case ${1-} in
  "") ;;
  --help|-h) usage; exit 0 ;;
  *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac

base_image_id=$(docker image inspect --format '{{.Id}}' "$QUALIFIED_BASE_TAG")
if test "$base_image_id" != "$QUALIFIED_BASE_IMAGE_ID"; then
  printf 'qualified base image mismatch\n' >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
umask 077
wheel_dir=$(mktemp -d)
inventory_dir=$(mktemp -d)
temporary_base_tag="dgx-qwen3-asr:vllm024-base-$$-$RANDOM"
cleanup() {
  docker image rm "$temporary_base_tag" >/dev/null 2>&1 || true
  rm -rf -- "$wheel_dir" "$inventory_dir"
}
trap cleanup EXIT

docker image tag "$QUALIFIED_BASE_IMAGE_ID" "$temporary_base_tag"
temporary_base_image_id=$(docker image inspect --format '{{.Id}}' "$temporary_base_tag")
if test "$temporary_base_image_id" != "$QUALIFIED_BASE_IMAGE_ID"; then
  printf 'temporary base tag mismatch\n' >&2
  exit 1
fi

for asset in "${ASSETS[@]}"; do
  IFS='|' read -r package version filename expected_size expected_sha256 url requires_python \
    <<<"$asset"
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$inventory_dir/$package-release.json" \
    "https://pypi.org/pypi/$package/$version/json"
  python3 "$script_dir/verify_vllm024_wheel_candidate.py" verify-release \
    "$package" "$inventory_dir/$package-release.json"
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$wheel_dir/$filename" "$url"
  python3 "$script_dir/verify_vllm024_wheel_candidate.py" verify-file \
    "$wheel_dir/$filename" "$expected_size" "$expected_sha256"
  if test -z "$requires_python"; then
    printf 'missing Python requirement for %s\n' "$package" >&2
    exit 1
  fi
done

DOCKER_BUILDKIT=1 docker build --network=none --pull=false \
  --build-arg "QUALIFIED_ASR_BASE=$temporary_base_tag" \
  --iidfile "$inventory_dir/candidate.iid" \
  --file "$script_dir/Dockerfile.vllm024-wheel-candidate" \
  "$wheel_dir"
candidate_image_id=$(tr -d '\n' <"$inventory_dir/candidate.iid")
if test -z "$candidate_image_id"; then
  printf 'candidate image ID is missing\n' >&2
  exit 1
fi
temporary_base_image_id=$(docker image inspect --format '{{.Id}}' "$temporary_base_tag")
if test "$temporary_base_image_id" != "$QUALIFIED_BASE_IMAGE_ID"; then
  printf 'temporary base tag changed during build\n' >&2
  exit 1
fi

python3 "$script_dir/verify_vllm024_wheel_candidate.py" capture-image \
  "$QUALIFIED_BASE_IMAGE_ID" "$inventory_dir/base.json"
python3 "$script_dir/verify_vllm024_wheel_candidate.py" capture-image \
  "$candidate_image_id" "$inventory_dir/candidate.json"
python3 "$script_dir/verify_vllm024_wheel_candidate.py" verify-images \
  "$inventory_dir/base.json" "$inventory_dir/candidate.json"

current_base_image_id=$(docker image inspect --format '{{.Id}}' "$QUALIFIED_BASE_TAG")
if test "$current_base_image_id" != "$QUALIFIED_BASE_IMAGE_ID"; then
  printf 'qualified base tag changed during build\n' >&2
  exit 1
fi
docker image tag "$candidate_image_id" "$CANDIDATE_TAG"
promoted_image_id=$(docker image inspect --format '{{.Id}}' "$CANDIDATE_TAG")
if test "$promoted_image_id" != "$candidate_image_id"; then
  printf 'candidate tag promotion failed\n' >&2
  exit 1
fi
printf 'candidate verified: tag=%s image=%s\n' "$CANDIDATE_TAG" "$candidate_image_id"
