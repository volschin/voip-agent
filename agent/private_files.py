"""Fail-closed validation for mounted credential and trust files."""

from __future__ import annotations

import stat
from pathlib import Path


def validate_private_file(
    path: str,
    *,
    label: str,
    forbid_group_other_read: bool,
) -> Path:
    """Require a regular, non-symlink file with the requested private mode."""

    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ValueError(f"{label} file is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} file is unsafe")

    forbidden = stat.S_IWGRP | stat.S_IWOTH
    if forbid_group_other_read:
        forbidden |= stat.S_IRGRP | stat.S_IROTH | stat.S_IXGRP | stat.S_IXOTH
    if metadata.st_mode & forbidden or not metadata.st_mode & stat.S_IRUSR:
        raise ValueError(f"{label} file has unsafe permissions")
    return candidate
