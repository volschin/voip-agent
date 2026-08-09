#!/usr/bin/env python3
"""Restore the validated Qwen3-Omni audio cu_seqlens construction."""

from __future__ import annotations

import argparse
import ast
import os
import tempfile
from pathlib import Path

ASYNC_CU_SEQLENS = """        cu_seqlens = async_tensor_h2d(
            cu_chunk_lens, dtype=torch.int32, device=aftercnn_lens.device
        ).cumsum(-1, dtype=torch.int32)
"""
DIRECT_CU_SEQLENS = (
    "        cu_seqlens = torch.tensor("
    "cu_chunk_lens, device=aftercnn_lens.device).cumsum(\n"
    "            -1, dtype=torch.int32\n"
    "        )\n"
)


def patch_module(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(ASYNC_CU_SEQLENS) != 1:
        raise ValueError("expected exactly one validated async cu_seqlens operation")

    patched = source.replace(ASYNC_CU_SEQLENS, DIRECT_CU_SEQLENS)
    ast.parse(patched, filename=str(path))
    compile(patched, str(path), "exec")

    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(patched)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(path.stat().st_mode)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path)
    args = parser.parse_args()
    patch_module(args.module)
    print("patched Qwen3OmniMoeAudioEncoder.forward cu_seqlens operation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
