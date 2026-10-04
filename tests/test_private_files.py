from pathlib import Path

import pytest

from agent.private_files import validate_private_file


def projection(root: Path, mode=0o440):
    generation = root / "..2026_10_04"
    root.chmod(0o755)
    generation.mkdir(mode=0o755)
    file = generation / "key"
    file.write_text("test-secret")
    file.chmod(mode)
    (root / "..data").symlink_to(generation.name)
    key = root / "key"
    key.symlink_to("..data/key")
    return key


def test_native_projection_accepts_current_group_and_rotation(tmp_path):
    key = projection(tmp_path)
    assert (
        validate_private_file(str(key), label="key", forbid_group_other_read=True).read_text()
        == "test-secret"
    )
    next_generation = tmp_path / "..next"
    next_generation.mkdir(mode=0o755)
    (next_generation / "key").write_text("rotated-secret")
    (next_generation / "key").chmod(0o440)
    (tmp_path / "..data").unlink()
    (tmp_path / "..data").symlink_to("..next")
    assert (
        validate_private_file(str(key), label="key", forbid_group_other_read=True).read_text()
        == "rotated-secret"
    )


@pytest.mark.parametrize("mode", [0o444, 0o460, 0o442])
def test_projection_rejects_unsafe_modes(tmp_path, mode):
    with pytest.raises(ValueError, match="permissions"):
        validate_private_file(
            str(projection(tmp_path, mode)), label="key", forbid_group_other_read=True
        )


def test_projection_rejects_escape(tmp_path):
    root = tmp_path / "mount"
    root.mkdir()
    outside = tmp_path / "..outside"
    outside.mkdir()
    (outside / "key").write_text("secret")
    (outside / "key").chmod(0o440)
    (root / "..data").symlink_to(outside)
    (root / "key").symlink_to("..data/key")
    with pytest.raises(ValueError, match="unsafe"):
        validate_private_file(str(root / "key"), label="key", forbid_group_other_read=True)


def test_projection_rejects_writable_mount(tmp_path):
    key = projection(tmp_path)
    tmp_path.chmod(0o777)
    with pytest.raises(ValueError, match="unsafe"):
        validate_private_file(str(key), label="key", forbid_group_other_read=True)
