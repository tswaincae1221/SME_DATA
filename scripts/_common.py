"""CLI 스크립트 공통 초기화."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_local_env(path: Path) -> None:
    """python-dotenv가 없을 때도 단순 KEY=VALUE 형식은 읽는다."""
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
        return
    except ImportError:
        pass
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env(ROOT / ".env")


def inclusive_dates(start: str, end: str) -> pd.DatetimeIndex:
    dates = pd.date_range(start, end, freq="D")
    if len(dates) == 0:
        raise ValueError("start는 end보다 늦을 수 없습니다.")
    return dates


def require_api_key() -> str:
    key = os.getenv("KMA_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "KMA_API_KEY가 없습니다. .env.example을 .env로 복사한 뒤 재발급한 키를 입력하세요."
        )
    return key
