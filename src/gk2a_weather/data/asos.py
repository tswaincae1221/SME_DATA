"""ASOS 시간자료 수집·파싱·캐시."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gk2a_weather.constants import ASOS_MISSING
from gk2a_weather.data.http import DownloadError, get_with_retries
from gk2a_weather.utils.io import atomic_write_csv, atomic_write_text


LOGGER = logging.getLogger(__name__)
ASOS_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"


class AsosParseError(ValueError):
    """ASOS 응답이 예상 형식과 다를 때 발생한다."""


def parse_asos_response(text: str) -> pd.DataFrame:
    """공백 구분 ASOS 응답에서 TM, STN_ID, TA, HM을 추출한다.

    공식 필드 순서 기준으로 TA는 12번째, HM은 14번째 항목이다.
    주석과 도움말 행은 건너뛴다.
    """
    if not text.rstrip().endswith("#7777END"):
        raise AsosParseError(
            "ASOS 응답 종료표시 #7777END가 없어 전송 중 잘린 원문입니다."
        )

    records: list[dict[str, object]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 14:
            continue
        if not fields[0].isdigit() or not fields[1].lstrip("-").isdigit():
            continue
        try:
            timestamp = pd.to_datetime(fields[0], format="%Y%m%d%H%M")
            stn_id = int(fields[1])
            ta = float(fields[11])
            hm = float(fields[13])
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "date": timestamp.strftime("%Y-%m-%d"),
                "timestamp_kst": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "STN_ID": stn_id,
                "TA": np.nan if ta == ASOS_MISSING["TA"] else ta,
                "HM": np.nan if hm == ASOS_MISSING["HM"] else hm,
            }
        )

    if not records:
        preview = " ".join(text.strip().split())[:160]
        raise AsosParseError(
            "ASOS 데이터 행을 찾지 못했습니다. API 승인·키·요청 시각을 확인하세요. "
            f"응답 앞부분: {preview!r}"
        )

    frame = pd.DataFrame.from_records(records)
    frame = frame.drop_duplicates(["timestamp_kst", "STN_ID"], keep="last")
    return frame.sort_values(["timestamp_kst", "STN_ID"]).reset_index(drop=True)


def fetch_asos(
    timestamp_kst: datetime,
    *,
    api_key: str,
    timeout_seconds: float = 30,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    session: Any = None,
) -> tuple[str, pd.DataFrame]:
    if not api_key.strip():
        raise ValueError("KMA_API_KEY가 비어 있습니다.")
    response = get_with_retries(
        ASOS_URL,
        params={
            "tm": timestamp_kst.strftime("%Y%m%d%H%M"),
            "stn": 0,
            "help": 0,
            "authKey": api_key,
        },
        timeout=timeout_seconds,
        max_retries=max_retries,
        backoff_seconds=retry_backoff_seconds,
        session=session,
    )
    # 기상청 응답은 환경에 따라 명시된 encoding이 다를 수 있으나 숫자 필드는 ASCII다.
    text = response.content.decode(response.encoding or "utf-8", errors="replace")
    return text, parse_asos_response(text)


def collect_asos_range(
    dates: Iterable[date | datetime | pd.Timestamp],
    *,
    api_key: str,
    raw_dir: str | Path,
    parsed_dir: str | Path,
    station_ids: Iterable[int] | None = None,
    observation_hour_kst: int = 14,
    timeout_seconds: float = 30,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    request_interval_seconds: float = 0.2,
    force: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """날짜별 파일을 저장하며 중간부터 재시작 가능한 수집을 수행한다."""
    raw_root = Path(raw_dir)
    parsed_root = Path(parsed_dir)
    allowed = set(int(value) for value in station_ids) if station_ids else None
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

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
            key = timestamp.strftime("%Y%m%d%H%M")
            raw_path = raw_root / f"asos_{key}.txt"
            parsed_path = parsed_root / f"asos_{key}.csv"

            try:
                if parsed_path.exists() and not force:
                    frame = pd.read_csv(parsed_path)
                else:
                    if raw_path.exists() and not force:
                        text = raw_path.read_text(encoding="utf-8", errors="replace")
                        frame = parse_asos_response(text)
                    else:
                        text, frame = fetch_asos(
                            timestamp,
                            api_key=api_key,
                            timeout_seconds=timeout_seconds,
                            max_retries=max_retries,
                            retry_backoff_seconds=retry_backoff_seconds,
                            session=session,
                        )
                        atomic_write_text(raw_path, text)

                    if allowed is not None:
                        frame = frame[frame["STN_ID"].isin(allowed)].copy()
                    atomic_write_csv(frame, parsed_path)

                if allowed is not None:
                    frame = frame[frame["STN_ID"].isin(allowed)].copy()
                frames.append(frame)
                LOGGER.info("ASOS %s 완료 (%d개 지점)", key, len(frame))
            except (DownloadError, AsosParseError, OSError, ValueError) as exc:
                LOGGER.error("ASOS %s 실패: %s", key, exc)
                failures.append(key)

            if request_interval_seconds > 0:
                time.sleep(request_interval_seconds)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, failures
