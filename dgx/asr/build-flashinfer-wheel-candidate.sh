#!/usr/bin/env bash
set -euo pipefail

QUALIFIED_BASE_TAG=dgx-qwen3-asr:vllm023-615e858c
QUALIFIED_BASE_IMAGE_ID=sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930
CANDIDATE_TAG=dgx-qwen3-asr:vllm023-flashinfer0618-test
RELEASE_ID=367461871
ASSETS=(
  "flashinfer_python-0.6.18-py3-none-any.whl|507452716|17122160|a722af5bbabd9156a6f75cec948822f0e966f8a83fffd913ead1fac851da7754"
  "flashinfer_jit_cache-0.6.18-cp39-abi3-manylinux_2_28_aarch64.whl|507452715|252992614|6e1eadb95eeaff33eb9393f73d6ad45d1c2bce001370d4388f76950043d7f0f1"
  "flashinfer_cubin-0.6.18-py3-none-any.whl|507452717|1239178852|fe03b57b9fa233a23efc3d29fa7a1fd48ebb9b7e8eda529187d505f2b1493315"
)

usage() {
  printf 'Usage: %s\nBuild the isolated FlashInfer 0.6.18 wheel candidate.\n' "${0##*/}"
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
cleanup() {
  rm -rf -- "$wheel_dir" "$inventory_dir"
}
trap cleanup EXIT

curl --fail --location --retry 3 --retry-all-errors \
  --header 'Accept: application/vnd.github+json' \
  --header 'X-GitHub-Api-Version: 2022-11-28' \
  --output "$inventory_dir/release.json" \
  "https://api.github.com/repos/eugr/spark-vllm-docker/releases/$RELEASE_ID"
python3 "$script_dir/verify_flashinfer_wheel_candidate.py" verify-release \
  "$inventory_dir/release.json"

for asset in "${ASSETS[@]}"; do
  IFS='|' read -r filename asset_id expected_size expected_sha256 <<<"$asset"
  curl --fail --location --retry 3 --retry-all-errors \
    --header 'Accept: application/octet-stream' \
    --header 'X-GitHub-Api-Version: 2022-11-28' \
    --output "$wheel_dir/$filename" \
    "https://api.github.com/repos/eugr/spark-vllm-docker/releases/assets/$asset_id"
  python3 "$script_dir/verify_flashinfer_wheel_candidate.py" verify-file \
    "$wheel_dir/$filename" "$expected_size" "$expected_sha256"
done

docker build --pull=false \
  --build-arg "QUALIFIED_ASR_BASE=$QUALIFIED_BASE_IMAGE_ID" \
  --iidfile "$inventory_dir/candidate.iid" \
  --file "$script_dir/Dockerfile.flashinfer-wheel-candidate" \
  "$wheel_dir"
candidate_image_id=$(tr -d '\n' <"$inventory_dir/candidate.iid")
if test -z "$candidate_image_id"; then
  printf 'candidate image ID is missing\n' >&2
  exit 1
fi

python3 "$script_dir/verify_flashinfer_wheel_candidate.py" capture-image \
  "$QUALIFIED_BASE_IMAGE_ID" "$inventory_dir/base.json"
python3 "$script_dir/verify_flashinfer_wheel_candidate.py" capture-image \
  "$candidate_image_id" "$inventory_dir/candidate.json"
python3 "$script_dir/verify_flashinfer_wheel_candidate.py" verify-images \
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
printf 'candidate verified: release=%s tag=%s image=%s\n' \
  "$RELEASE_ID" "$CANDIDATE_TAG" "$candidate_image_id"
