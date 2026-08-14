"""중간 실패 파일이 정상 캐시로 남지 않게 하는 원자적 저장 함수."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_text(
    path: str | Path, content: str, encoding: str = "utf-8"
) -> Path:
    return atomic_write_bytes(path, content.encode(encoding))


def atomic_write_csv(data: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        data.to_csv(tmp_path, index=False)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return destination

