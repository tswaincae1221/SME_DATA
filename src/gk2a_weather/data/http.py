"""KMA API 호출에 공통으로 쓰는 재시도 로직."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

class DownloadError(RuntimeError):
    """API 응답을 정상 데이터로 사용할 수 없을 때 발생한다."""


def get_with_retries(
    url: str,
    *,
    params: Mapping[str, object],
    timeout: float,
    max_retries: int,
    backoff_seconds: float,
    session: Any = None,
) -> Any:
    """인증키를 로그에 남기지 않고 GET 요청을 재시도한다."""
    try:
        import requests
    except ImportError as exc:
        raise DownloadError(
            "requests가 설치되어 있지 않습니다. pip install -r requirements.txt를 실행하세요."
        ) from exc
    client = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    raise DownloadError(
        f"API 요청이 {max_retries}회 모두 실패했습니다. "
        "인터넷 연결, 활용 신청 상태, 요청 시각을 확인하세요."
    ) from last_error


def looks_like_error_document(content: bytes, content_type: str = "") -> bool:
    """HTML/XML 오류 문서를 위성 바이너리로 잘못 저장하지 않게 검사한다."""
    prefix = content[:512].lstrip().lower()
    mime = content_type.lower()
    return (
        b"<html" in prefix
        or b"<!doctype html" in prefix
        or b"<error" in prefix
        or "text/html" in mime
    )
