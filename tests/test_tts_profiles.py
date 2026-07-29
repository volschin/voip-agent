import hashlib
import json
import wave
from pathlib import Path

import pytest

from dgx.tts.profiles import ProfileError, load_profiles, resolve_profile

PROFILE_ID = "shared-female-de-v1"
REFERENCE_TEXT = "Guten Tag, hier ist Ihre digitale Assistentin."


def _write_wav(path: Path, *, channels: int = 1, rate: int = 24_000, width: int = 2) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(b"\x01\x00" * rate)
    path.chmod(0o600)


def _profile(audio_hash: str) -> dict:
    return {
        "id": PROFILE_ID,
        "audio": f"{PROFILE_ID}.wav",
        "reference_text": REFERENCE_TEXT,
        "language": "german",
        "source_type": "licensed-human-reference-private",
        "source_revision": "private-source-v1",
        "source_sha256": "1" * 64,
        "sha256": audio_hash,
        "selected_at": "2026-07-29T12:00:00Z",
        "evaluation_score": 91.25,
        "selected_candidate_id": "candidate-a",
        "design_instruction": None,
    }


def _bundle(tmp_path: Path) -> tuple[Path, Path, dict]:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir(mode=0o700)
    audio = profile_dir / f"{PROFILE_ID}.wav"
    _write_wav(audio)
    audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "usage_scope": "private-user-assistant-only",
        "profiles": [_profile(audio_hash)],
    }
    manifest = profile_dir / "profiles.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.chmod(0o600)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"36 25 0:32 / {profile_dir} ro,relatime - ext4 /dev/test ro\n",
        encoding="utf-8",
    )
    return profile_dir, mountinfo, payload


def test_load_profiles_accepts_private_read_only_pcm_bundle(tmp_path: Path) -> None:
    profile_dir, mountinfo, _ = _bundle(tmp_path)

    profiles = load_profiles(profile_dir, mountinfo_path=mountinfo)

    profile = profiles[PROFILE_ID]
    assert profile.audio_path == (profile_dir / f"{PROFILE_ID}.wav").resolve()
    assert profile.reference_text == REFERENCE_TEXT
    assert profile.language == "german"
    assert profile.evaluation_score == 91.25
    assert profile.selected_candidate_id == "candidate-a"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("usage_scope", "public", "usage scope"),
        ("schema_version", 2, "schema version"),
    ],
)
def test_load_profiles_rejects_wrong_bundle_contract(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    profile_dir, mountinfo, payload = _bundle(tmp_path)
    payload[field] = value
    (profile_dir / "profiles.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileError, match=message):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


def test_load_profiles_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    profile_dir, mountinfo, payload = _bundle(tmp_path)
    encoded = json.dumps(payload)
    encoded = encoded.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    (profile_dir / "profiles.json").write_text(encoded, encoding="utf-8")

    with pytest.raises(ProfileError, match="duplicate JSON key"):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


def test_load_profiles_rejects_duplicate_profile_ids(tmp_path: Path) -> None:
    profile_dir, mountinfo, payload = _bundle(tmp_path)
    payload["profiles"].append(dict(payload["profiles"][0]))
    (profile_dir / "profiles.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileError, match="duplicate profile id"):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_text", " ", "reference_text"),
        ("language", "de", "language"),
        ("sha256", "A" * 64, "sha256"),
        ("source_sha256", "0" * 63, "source_sha256"),
        ("selected_at", "yesterday", "selected_at"),
        ("evaluation_score", 101, "evaluation_score"),
        ("selected_candidate_id", "../candidate-a", "selected_candidate_id"),
        ("source_type", "downloaded-public", "source_type"),
        ("design_instruction", "must be private", "design_instruction"),
    ],
)
def test_load_profiles_rejects_invalid_profile_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    profile_dir, mountinfo, payload = _bundle(tmp_path)
    payload["profiles"][0][field] = value
    (profile_dir / "profiles.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileError, match=message):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


@pytest.mark.parametrize("audio_value", ["../voice.wav", "/tmp/voice.wav", "nested/voice.wav"])
def test_load_profiles_rejects_audio_path_escape(tmp_path: Path, audio_value: str) -> None:
    profile_dir, mountinfo, payload = _bundle(tmp_path)
    payload["profiles"][0]["audio"] = audio_value
    (profile_dir / "profiles.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileError, match="audio path"):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


def test_load_profiles_rejects_symlink_audio(tmp_path: Path) -> None:
    profile_dir, mountinfo, payload = _bundle(tmp_path)
    audio = profile_dir / f"{PROFILE_ID}.wav"
    target = tmp_path / "outside.wav"
    audio.rename(target)
    audio.symlink_to(target)

    with pytest.raises(ProfileError, match="symlink"):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


def test_load_profiles_rejects_hash_mismatch(tmp_path: Path) -> None:
    profile_dir, mountinfo, _ = _bundle(tmp_path)
    audio = profile_dir / f"{PROFILE_ID}.wav"
    audio.write_bytes(audio.read_bytes() + b"\x00")

    with pytest.raises(ProfileError, match="hash mismatch"):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


@pytest.mark.parametrize(
    ("channels", "rate", "width", "message"),
    [
        (2, 24_000, 2, "mono"),
        (1, 16_000, 2, "24 kHz"),
        (1, 24_000, 1, "16-bit"),
    ],
)
def test_load_profiles_rejects_wrong_wav_contract(
    tmp_path: Path, channels: int, rate: int, width: int, message: str
) -> None:
    profile_dir, mountinfo, payload = _bundle(tmp_path)
    audio = profile_dir / f"{PROFILE_ID}.wav"
    _write_wav(audio, channels=channels, rate=rate, width=width)
    payload["profiles"][0]["sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
    (profile_dir / "profiles.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileError, match=message):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


def test_load_profiles_requires_private_directory_permissions(tmp_path: Path) -> None:
    profile_dir, mountinfo, _ = _bundle(tmp_path)
    profile_dir.chmod(0o755)

    with pytest.raises(ProfileError, match="permissions"):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


def test_load_profiles_requires_read_only_mount(tmp_path: Path) -> None:
    profile_dir, mountinfo, _ = _bundle(tmp_path)
    mountinfo.write_text(
        f"36 25 0:32 / {profile_dir} rw,relatime - ext4 /dev/test rw\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileError, match="read-only"):
        load_profiles(profile_dir, mountinfo_path=mountinfo)


def test_resolve_profile_accepts_default_id_and_rollout_aliases(tmp_path: Path) -> None:
    profile_dir, mountinfo, _ = _bundle(tmp_path)
    profiles = load_profiles(profile_dir, mountinfo_path=mountinfo)
    aliases = [
        None,
        PROFILE_ID,
        "Eine warme, natürliche deutsche Stimme mit angemessenem Sprechtempo.",
        (
            "Eine warme, natürliche, erwachsene weibliche deutsche Stimme mit klarer "
            "Aussprache, lebendiger Intonation und ruhigem Sprechtempo."
        ),
    ]

    assert all(
        resolve_profile(value, profiles, PROFILE_ID).profile_id == PROFILE_ID for value in aliases
    )


@pytest.mark.parametrize("value", ["", " ", "unknown-profile"])
def test_resolve_profile_rejects_blank_or_unknown_selector(tmp_path: Path, value: str) -> None:
    profile_dir, mountinfo, _ = _bundle(tmp_path)
    profiles = load_profiles(profile_dir, mountinfo_path=mountinfo)

    with pytest.raises(ProfileError, match="unsupported voice profile"):
        resolve_profile(value, profiles, PROFILE_ID)
