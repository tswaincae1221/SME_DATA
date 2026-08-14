"""GK-2A 16채널 원본 파일 수집."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gk2a_weather.constants import GK2A_CHANNELS
from gk2a_weather.data.http import (
    DownloadError,
    get_with_retries,
    looks_like_error_document,
)
from gk2a_weather.utils.io import atomic_write_bytes


LOGGER = logging.getLogger(__name__)
GK2A_URL = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{channel}/{area}/data"
KST = timezone(timedelta(hours=9))


class SatelliteTimeBasisError(ValueError):
    """위성 API 시간 기준이 확인되지 않았을 때 발생한다."""


def canonical_gk2a_path(
    root: str | Path, day: pd.Timestamp, channel: str, requested: str
) -> Path:
    """팀 다운로더와 동일한 YYYY/MM/DD 구조의 LE1B NetCDF 경로."""
    return (
        Path(root)
        / day.strftime("%Y")
        / day.strftime("%m")
        / day.strftime("%d")
        / f"{channel.upper()}_{requested}.nc"
    )


def find_gk2a_file(
    root: str | Path, day: pd.Timestamp, channel: str, requested: str
) -> Path | None:
    """신규 .nc 구조와 기존 .bin 캐시를 모두 읽되 LE1B 채널만 찾는다."""
    root_path = Path(root)
    channel = channel.upper()
    candidates = (
        canonical_gk2a_path(root_path, day, channel, requested),
        root_path / day.strftime("%Y%m%d") / f"{channel}_{requested}.nc",
        root_path / day.strftime("%Y%m%d") / f"{channel}_{requested}.bin",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    day_root = root_path / day.strftime("%Y") / day.strftime("%m") / day.strftime("%d")
    if day_root.is_dir():
        patterns = (
            f"*{channel.lower()}*{requested}*.nc",
            f"*{channel.upper()}*{requested}*.nc",
        )
        matches = sorted({path for pattern in patterns for path in day_root.glob(pattern)})
        if len(matches) > 1:
            raise ValueError(
                f"{day.date()} {channel} 파일이 여러 개입니다: "
                + ", ".join(path.name for path in matches[:5])
            )
        if matches:
            return matches[0]
    return None


def api_timestamp(timestamp_kst: datetime, time_basis: str) -> str:
    """14시 KST를 설정된 위성 API 시간 기준으로 변환한다."""
    basis = time_basis.lower()
    if basis == "review_required":
        raise SatelliteTimeBasisError(
            "configs/data.yaml의 satellite.api_time_basis가 review_required입니다. "
            "대회 baseline_notebook에서 위성 API 요청 시각이 KST인지 UTC인지 확인한 뒤 "
            "kst 또는 utc로 바꿔주세요."
        )
    if basis not in {"kst", "utc"}:
        raise SatelliteTimeBasisError("api_time_basis는 kst, utc 중 하나여야 합니다.")

    aware = timestamp_kst.replace(tzinfo=KST)
    converted = aware if basis == "kst" else aware.astimezone(timezone.utc)
    return converted.strftime("%Y%m%d%H%M")


def download_gk2a_channel(
    timestamp_kst: datetime,
    channel: str,
    *,
    api_key: str,
    output_path: str | Path,
    area: str = "KO",
    api_time_basis: str,
    timeout_seconds: float = 90,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    minimum_file_bytes: int = 10_000,
    session: Any = None,
) -> Path:
    channel = channel.upper()
    if channel not in GK2A_CHANNELS:
        raise ValueError(f"지원하지 않는 GK-2A 채널입니다: {channel}")
    if area.upper() != "KO":
        raise ValueError("이 대회에서는 위성 영역을 KO로 사용해야 합니다.")
    if not api_key.strip():
        raise ValueError("KMA_API_KEY가 비어 있습니다.")

    requested = api_timestamp(timestamp_kst, api_time_basis)
    response = get_with_retries(
        GK2A_URL.format(channel=channel, area=area.upper()),
        params={"date": requested, "authKey": api_key},
        timeout=timeout_seconds,
        max_retries=max_retries,
        backoff_seconds=retry_backoff_seconds,
        session=session,
    )
    content = response.content
    content_type = response.headers.get("content-type", "")
    if looks_like_error_document(content, content_type):
        raise DownloadError(
            f"{channel} {requested} 응답이 데이터가 아닌 오류 문서입니다."
        )
    if len(content) < minimum_file_bytes:
        raise DownloadError(
            f"{channel} {requested} 파일이 너무 작습니다: {len(content)} bytes"
        )
    return atomic_write_bytes(output_path, content)


def collect_gk2a_range(
    dates: Iterable[date | datetime | pd.Timestamp],
    *,
    api_key: str,
    raw_dir: str | Path,
    channels: Iterable[str] = GK2A_CHANNELS,
    observation_hour_kst: int = 14,
    area: str = "KO",
    api_time_basis: str,
    timeout_seconds: float = 90,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    request_interval_seconds: float = 0.2,
    minimum_file_bytes: int = 10_000,
    force: bool = False,
) -> list[dict[str, str]]:
    """날짜×채널을 수집하고 실패 목록을 반환한다."""
    root = Path(raw_dir)
    failures: list[dict[str, str]] = []

    try:
        import requests
    except ImportError as exc:
        raise DownloadError(
            "requests가 설치되어 있지 않습니다. pip install -r requirements.txt를 실행하세요."
        ) from exc

    with requests.Session() as session:
        for value in dates:
            day = pd.Timestamp(value)
            timestamp = day.replace(
                hour=observation_hour_kst, minute=0, second=0, microsecond=0
            ).to_pydatetime()
            requested = api_timestamp(timestamp, api_time_basis)

            for channel_value in channels:
                channel = channel_value.upper()
                output_path = canonical_gk2a_path(root, day, channel, requested)
                if output_path.exists() and output_path.stat().st_size >= minimum_file_bytes and not force:
                    LOGGER.info("GK-2A %s %s 캐시 사용", requested, channel)
                    continue
                try:
                    download_gk2a_channel(
                        timestamp,
                        channel,
                        api_key=api_key,
                        output_path=output_path,
                        area=area,
                        api_time_basis=api_time_basis,
                        timeout_seconds=timeout_seconds,
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                        minimum_file_bytes=minimum_file_bytes,
                        session=session,
                    )
                    LOGGER.info("GK-2A %s %s 완료", requested, channel)
                except (DownloadError, OSError, ValueError) as exc:
                    LOGGER.error("GK-2A %s %s 실패: %s", requested, channel, exc)
                    failures.append(
                        {"timestamp": requested, "channel": channel, "error": str(exc)}
                    )
                if request_interval_seconds > 0:
                    time.sleep(request_interval_seconds)
    return failures
