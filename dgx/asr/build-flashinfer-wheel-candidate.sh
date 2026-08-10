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

case ${1-} in
  "") ;;
  --help|-h) usage; exit 0 ;;
  *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac
test "$#" -le 1

base_image_id=$(docker image inspect --format '{{.Id}}' "$QUALIFIED_BASE_TAG")
if test "$base_image_id" != "$QUALIFIED_BASE_IMAGE_ID"; then
  printf 'qualified base image mismatch\n' >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
wheel_dir=$(mktemp -d)
inventory_dir=$(mktemp -d)
cleanup() {
  rm -rf -- "$wheel_dir" "$inventory_dir"
}
trap cleanup EXIT

umask 077
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
  --build-arg "QUALIFIED_ASR_BASE=$QUALIFIED_BASE_TAG" \
  --tag "$CANDIDATE_TAG" \
  --file "$script_dir/Dockerfile.flashinfer-wheel-candidate" \
  "$wheel_dir"

python3 "$script_dir/verify_flashinfer_wheel_candidate.py" capture-image \
  "$QUALIFIED_BASE_TAG" "$inventory_dir/base.json"
python3 "$script_dir/verify_flashinfer_wheel_candidate.py" capture-image \
  "$CANDIDATE_TAG" "$inventory_dir/candidate.json"
python3 "$script_dir/verify_flashinfer_wheel_candidate.py" verify-images \
  "$inventory_dir/base.json" "$inventory_dir/candidate.json"

candidate_image_id=$(docker image inspect --format '{{.Id}}' "$CANDIDATE_TAG")
printf 'candidate verified: release=%s tag=%s image=%s\n' \
  "$RELEASE_ID" "$CANDIDATE_TAG" "$candidate_image_id"
