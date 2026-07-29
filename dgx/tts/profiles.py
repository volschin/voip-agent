"""Fail-closed loading of private, read-only TTS voice profiles."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROFILE_ID = "shared-female-de-v1"
USAGE_SCOPE = "private-user-assistant-only"
SOURCE_TYPES = {
    "licensed-human-reference-private",
    "synthetic-qwen-voice-design",
}
ROLLOUT_ALIASES = {
    "Eine warme, natürliche deutsche Stimme mit angemessenem Sprechtempo.",
    (
        "Eine warme, natürliche, erwachsene weibliche deutsche Stimme mit klarer "
        "Aussprache, lebendiger Intonation und ruhigem Sprechtempo."
    ),
}
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_CANDIDATE_RE = re.compile(r"candidate-[a-z][a-z0-9-]*")
_TOP_LEVEL_FIELDS = {"schema_version", "usage_scope", "profiles"}
_PROFILE_FIELDS = {
    "id",
    "audio",
    "reference_text",
    "language",
    "source_type",
    "source_revision",
    "source_sha256",
    "sha256",
    "selected_at",
    "evaluation_score",
    "selected_candidate_id",
    "design_instruction",
}


class ProfileError(ValueError):
    """Bounded validation failure for a private voice-profile bundle."""


@dataclass(frozen=True)
class VoiceProfile:
    profile_id: str
    audio_path: Path
    reference_text: str
    language: str
    source_type: str
    source_revision: str
    source_sha256: str
    sha256: str
    selected_at: str
    evaluation_score: float
    selected_candidate_id: str
    design_instruction: str | None


def load_profiles(
    profile_dir: Path,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> dict[str, VoiceProfile]:
    """Load and validate the private profile bundle or fail startup."""
    if profile_dir.is_symlink() or not profile_dir.is_dir():
        raise ProfileError("profile directory is unavailable")
    resolved_dir = profile_dir.resolve()
    if stat.S_IMODE(resolved_dir.stat().st_mode) & 0o077:
        raise ProfileError("profile directory permissions must be private")
    _require_read_only_mount(resolved_dir, mountinfo_path)

    manifest_path = resolved_dir / "profiles.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ProfileError("profile manifest is unavailable")
    _require_private_file(manifest_path)
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except ProfileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProfileError("profile manifest is invalid") from error

    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ProfileError("profile manifest fields are invalid")
    if payload["schema_version"] != 1:
        raise ProfileError("profile schema version is unsupported")
    if payload["usage_scope"] != USAGE_SCOPE:
        raise ProfileError("profile usage scope is invalid")
    entries = payload["profiles"]
    if not isinstance(entries, list) or not entries:
        raise ProfileError("profiles must be a non-empty list")

    profiles: dict[str, VoiceProfile] = {}
    for entry in entries:
        profile = _load_profile(entry, resolved_dir)
        if profile.profile_id in profiles:
            raise ProfileError("duplicate profile id")
        profiles[profile.profile_id] = profile
    return profiles


def resolve_profile(
    value: str | None,
    profiles: Mapping[str, VoiceProfile],
    default_id: str,
) -> VoiceProfile:
    """Resolve a canonical profile or one of the bounded rollout aliases."""
    selector = default_id if value is None or value in ROLLOUT_ALIASES else value
    if not isinstance(selector, str) or not selector.strip() or selector not in profiles:
        raise ProfileError("unsupported voice profile")
    return profiles[selector]


def _load_profile(entry: Any, profile_dir: Path) -> VoiceProfile:
    if not isinstance(entry, dict) or set(entry) != _PROFILE_FIELDS:
        raise ProfileError("profile fields are invalid")

    profile_id = _nonblank_string(entry["id"], "id")
    audio_name = _nonblank_string(entry["audio"], "audio")
    audio_part = Path(audio_name)
    if audio_part.is_absolute() or len(audio_part.parts) != 1 or audio_part.name != audio_name:
        raise ProfileError("profile audio path is invalid")
    audio_path = profile_dir / audio_name
    if audio_path.is_symlink():
        raise ProfileError("profile audio must not be a symlink")
    if not audio_path.is_file() or audio_path.resolve().parent != profile_dir:
        raise ProfileError("profile audio path is invalid")
    _require_private_file(audio_path)

    reference_text = _nonblank_string(entry["reference_text"], "reference_text")
    if entry["language"] != "german":
        raise ProfileError("profile language must be german")
    source_type = entry["source_type"]
    if source_type not in SOURCE_TYPES:
        raise ProfileError("profile source_type is invalid")
    source_revision = _nonblank_string(entry["source_revision"], "source_revision")
    source_sha256 = _hash(entry["source_sha256"], "source_sha256")
    audio_sha256 = _hash(entry["sha256"], "sha256")
    if hashlib.sha256(audio_path.read_bytes()).hexdigest() != audio_sha256:
        raise ProfileError("profile audio hash mismatch")

    selected_at = _utc_timestamp(entry["selected_at"])
    score = entry["evaluation_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
        raise ProfileError("profile evaluation_score is invalid")
    candidate_id = entry["selected_candidate_id"]
    if not isinstance(candidate_id, str) or _CANDIDATE_RE.fullmatch(candidate_id) is None:
        raise ProfileError("profile selected_candidate_id is invalid")

    instruction = entry["design_instruction"]
    if source_type == "licensed-human-reference-private":
        if instruction is not None:
            raise ProfileError("profile design_instruction must be null for licensed source")
    elif not isinstance(instruction, str) or not instruction.strip():
        raise ProfileError("profile design_instruction is required for synthetic source")

    _validate_wav(audio_path)
    return VoiceProfile(
        profile_id=profile_id,
        audio_path=audio_path.resolve(),
        reference_text=reference_text,
        language="german",
        source_type=source_type,
        source_revision=source_revision,
        source_sha256=source_sha256,
        sha256=audio_sha256,
        selected_at=selected_at,
        evaluation_score=float(score),
        selected_candidate_id=candidate_id,
        design_instruction=instruction,
    )


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ProfileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonblank_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"profile {name} is invalid")
    return value.strip()


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ProfileError(f"profile {name} is invalid")
    return value


def _utc_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProfileError("profile selected_at is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProfileError("profile selected_at is invalid") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ProfileError("profile selected_at is invalid")
    return value


def _require_private_file(path: Path) -> None:
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ProfileError("profile file permissions must be private")


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1:
                raise ProfileError("profile audio must be mono")
            if audio.getframerate() != 24_000:
                raise ProfileError("profile audio must be 24 kHz")
            if audio.getsampwidth() != 2:
                raise ProfileError("profile audio must be 16-bit PCM")
            if audio.getcomptype() != "NONE":
                raise ProfileError("profile audio must be uncompressed PCM")
            if audio.getnframes() <= 0:
                raise ProfileError("profile audio must not be empty")
    except ProfileError:
        raise
    except (EOFError, OSError, wave.Error) as error:
        raise ProfileError("profile audio must be a valid WAV") from error


def _require_read_only_mount(profile_dir: Path, mountinfo_path: Path) -> None:
    try:
        lines = mountinfo_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProfileError("profile read-only mount cannot be verified") from error

    candidates: list[tuple[int, set[str]]] = []
    directory = str(profile_dir)
    for line in lines:
        before_separator = line.split(" - ", 1)[0].split()
        if len(before_separator) < 6:
            continue
        mount_point = _unescape_mountinfo(before_separator[4])
        if directory == mount_point or directory.startswith(mount_point.rstrip("/") + "/"):
            candidates.append((len(mount_point), set(before_separator[5].split(","))))
    if not candidates or "ro" not in max(candidates, key=lambda item: item[0])[1]:
        raise ProfileError("profile directory must be a read-only mount")


def _unescape_mountinfo(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )
