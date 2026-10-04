"""Fail-closed validation for mounted credential and trust files."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def validate_private_file(
    path: str,
    *,
    label: str,
    forbid_group_other_read: bool,
) -> Path:
    """Validate owner-only files or protected kubelet Secret projections."""

    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ValueError(f"{label} file is unavailable") from error
    projected = stat.S_ISLNK(metadata.st_mode)
    if projected:
        root = candidate.parent
        try:
            data = root / "..data"
            generation = os.readlink(data)
            if (
                os.readlink(candidate) != f"..data/{candidate.name}"
                or not generation.startswith("..")
                or "/" in generation
                or generation in {"..", "..data"}
                or root.is_symlink()
                or (root / generation).is_symlink()
            ):
                raise ValueError(f"{label} file is unsafe")
            resolved = candidate.resolve(strict=True)
            if resolved.parent != (root / generation).resolve(strict=True):
                raise ValueError(f"{label} file is unsafe")
            for directory in (root, resolved.parent):
                info = directory.stat()
                if info.st_mode & 0o022 or info.st_uid not in {0, os.geteuid()}:
                    raise ValueError(f"{label} file is unsafe")
            candidate = resolved
            metadata = candidate.lstat()
            if metadata.st_uid not in {0, os.geteuid()}:
                raise ValueError(f"{label} file is unsafe")
        except (OSError, RuntimeError) as error:
            raise ValueError(f"{label} file is unsafe") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} file is unsafe")

    forbidden = stat.S_IWGRP | stat.S_IWOTH
    if forbid_group_other_read:
        forbidden |= stat.S_IROTH | stat.S_IXGRP | stat.S_IXOTH
        if not projected or metadata.st_gid != os.getegid():
            forbidden |= stat.S_IRGRP
    if metadata.st_mode & forbidden or not metadata.st_mode & stat.S_IRUSR:
        raise ValueError(f"{label} file has unsafe permissions")
    return candidate
