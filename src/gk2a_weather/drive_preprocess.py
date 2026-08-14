"""Google Drive의 GK-2A LE1B/KO NetCDF를 관측소 단위 표로 변환한다.

이 모듈은 Google Drive API를 호출하지 않는다. Colab에서 Drive를 마운트한 뒤
일반 디렉터리처럼 읽고 쓰므로, 인증키를 코드나 저장소에 남기지 않는다.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from gk2a_weather.constants import GK2A_CHANNELS
from gk2a_weather.features.satellite import (
    add_channel_differences,
    extract_channel_features,
    load_satellite_array,
)


KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc
TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(20\d{10})(?!\d)")
CHANNEL_PATTERN = re.compile(
    r"(?<![A-Z0-9])(" + "|".join(GK2A_CHANNELS) + r")(?![A-Z0-9])",
    re.IGNORECASE,
)
STATISTICS = ("mean", "std", "min", "max", "p10", "p50", "p90", "valid_ratio")


@dataclass(frozen=True)
class NcFileRecord:
    """파일명에서 읽은 채널과 시각 정보."""

    path: Path
    channel: str
    timestamp_source: datetime
    timestamp_utc: datetime
    timestamp_kst: datetime


def parse_nc_filename(
    path: str | Path,
    *,
    source_timezone: str = "utc",
) -> NcFileRecord | None:
    """KMA 원본명과 ``CHANNEL_YYYYMMDDHHMM.nc`` 이름을 모두 해석한다."""
    nc_path = Path(path)
    if nc_path.suffix.lower() != ".nc":
        return None

    channel_match = CHANNEL_PATTERN.search(nc_path.stem.upper())
    timestamps = TIMESTAMP_PATTERN.findall(nc_path.stem)
    if channel_match is None or not timestamps:
        return None

    channel = channel_match.group(1).upper()
    naive = datetime.strptime(timestamps[-1], "%Y%m%d%H%M")
    basis = source_timezone.lower()
    if basis == "utc":
        source = naive.replace(tzinfo=UTC)
        timestamp_utc = source
        timestamp_kst = source.astimezone(KST)
    elif basis == "kst":
        source = naive.replace(tzinfo=KST)
        timestamp_kst = source
        timestamp_utc = source.astimezone(UTC)
    else:
        raise ValueError("source_timezone은 utc 또는 kst여야 합니다.")

    return NcFileRecord(
        path=nc_path,
        channel=channel,
        timestamp_source=source,
        timestamp_utc=timestamp_utc,
        timestamp_kst=timestamp_kst,
    )


def discover_nc_files(
    input_dir: str | Path,
    *,
    source_timezone: str = "utc",
) -> tuple[list[NcFileRecord], list[Path]]:
    """하위 폴더 전체를 검색하고 이름을 해석하지 못한 NC도 함께 반환한다."""
    root = Path(input_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Drive 입력 폴더가 없습니다: {root}")

    records: list[NcFileRecord] = []
    unparsed: list[Path] = []
    candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".nc"
    )
    for path in candidates:
        record = parse_nc_filename(path, source_timezone=source_timezone)
        if record is None:
            unparsed.append(path)
        else:
            records.append(record)
    return records, unparsed


def record_is_in_scope(
    record: NcFileRecord,
    *,
    start_date: str,
    end_date: str,
    months: Iterable[int],
    target_hour_kst: int,
) -> bool:
    day = record.timestamp_kst.date()
    return (
        pd.Timestamp(start_date).date() <= day <= pd.Timestamp(end_date).date()
        and record.timestamp_kst.month in set(int(value) for value in months)
        and record.timestamp_kst.hour == int(target_hour_kst)
        and record.timestamp_kst.minute == 0
    )


def _preference_score(record: NcFileRecord) -> tuple[int, int, int, str]:
    """중복 파일에서는 KO 표기와 단순 표준 파일명을 우선한다."""
    name = record.path.name.lower()
    ko_hint = int(bool(re.search(r"(^|[_-])ko(?:[_\-.]|\d)", name)))
    fd_hint = int(bool(re.search(r"(^|[_-])fd(?:[_\-.]|\d)", name)))
    simple_name = int(
        name.startswith(
            f"{record.channel.lower()}_{record.timestamp_source.strftime('%Y%m%d%H%M')}"
        )
    )
    try:
        size = int(record.path.stat().st_size)
    except OSError:
        size = -1
    return ko_hint - fd_hint, simple_name, size, str(record.path)


def group_nc_records(
    records: Iterable[NcFileRecord],
) -> dict[datetime, dict[str, list[NcFileRecord]]]:
    """KST 시각과 채널별로 파일을 묶는다."""
    groups: dict[datetime, dict[str, list[NcFileRecord]]] = {}
    for record in records:
        timestamp = record.timestamp_kst.replace(second=0, microsecond=0)
        groups.setdefault(timestamp, {}).setdefault(record.channel, []).append(record)
    return groups


def select_group_files(
    group: dict[str, list[NcFileRecord]],
) -> tuple[dict[str, NcFileRecord], dict[str, list[str]]]:
    """채널별 중복 중 하나를 선택하고 선택되지 않은 경로를 기록한다."""
    selected: dict[str, NcFileRecord] = {}
    duplicates: dict[str, list[str]] = {}
    for channel, candidates in group.items():
        ordered = sorted(candidates, key=_preference_score, reverse=True)
        selected[channel] = ordered[0]
        if len(ordered) > 1:
            duplicates[channel] = [str(record.path) for record in ordered[1:]]
    return selected, duplicates


def channel_feature_columns(channel: str, radii_km: Iterable[float]) -> list[str]:
    columns: list[str] = []
    suffixes: list[str] = []
    for radius in radii_km:
        radius = float(radius)
        suffix = "center" if radius == 0 else f"r{radius:g}km"
        suffixes.append(suffix)
        columns.extend(f"{channel}_{suffix}_{stat}" for stat in STATISTICS)
    if suffixes:
        columns.append(f"{channel}_center_minus_{suffixes[-1]}")
    columns.append(f"{channel}_missing")
    return columns


def _date_station_frame(stations: pd.DataFrame, timestamp_kst: datetime) -> pd.DataFrame:
    day = pd.Timestamp(timestamp_kst.date())
    angle = 2.0 * math.pi * (int(day.dayofyear) - 1) / 365.25
    frame = stations.copy()
    frame.insert(0, "ID", day.strftime("%Y%m%d") + "_" + frame["STN_ID"].astype(str))
    frame.insert(0, "date", day.strftime("%Y-%m-%d"))
    frame["timestamp_kst"] = timestamp_kst.isoformat()
    frame["timestamp_utc"] = timestamp_kst.astimezone(UTC).isoformat()
    frame["year"] = int(day.year)
    frame["month"] = int(day.month)
    frame["day"] = int(day.day)
    frame["day_of_year"] = int(day.dayofyear)
    frame["day_sin"] = math.sin(angle)
    frame["day_cos"] = math.cos(angle)
    return frame


def _stage_path(record: NcFileRecord, stage_dir: Path) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    suffix = record.path.suffix.lower()
    return stage_dir / (
        f"{record.channel}_{record.timestamp_utc.strftime('%Y%m%d%H%M')}{suffix}"
    )


def process_timestamp(
    *,
    timestamp_kst: datetime,
    selected_files: dict[str, NcFileRecord],
    stations: pd.DataFrame,
    radii_km: tuple[float, ...],
    channels: tuple[str, ...] = tuple(GK2A_CHANNELS),
    stage_dir: str | Path | None = None,
) -> pd.DataFrame:
    """한 시각의 최대 16개 NC를 읽어 96개 관측소 행으로 변환한다."""
    if not radii_km or float(radii_km[0]) != 0.0:
        raise ValueError("radii_km의 첫 값은 중심 픽셀을 뜻하는 0이어야 합니다.")

    combined = _date_station_frame(stations, timestamp_kst)
    temporary_root = Path(stage_dir) if stage_dir else None
    missing_feature_values: dict[str, float | int] = {}

    for channel in channels:
        record = selected_files.get(channel)
        if record is None:
            for column in channel_feature_columns(channel, radii_km):
                missing_feature_values[column] = (
                    1 if column.endswith("_missing") else np.nan
                )
            continue

        read_path = record.path
        staged_path: Path | None = None
        try:
            if temporary_root is not None:
                staged_path = _stage_path(record, temporary_root)
                shutil.copy2(record.path, staged_path)
                read_path = staged_path
            array = load_satellite_array(read_path, channel)
            features = extract_channel_features(
                array,
                stations,
                channel=channel,
                radii_km=radii_km,
            )
            combined = combined.merge(features, on="STN_ID", how="left", validate="one_to_one")
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)

    if missing_feature_values:
        missing_frame = pd.DataFrame(
            {
                column: np.full(len(combined), value)
                for column, value in missing_feature_values.items()
            },
            index=combined.index,
        )
        combined = pd.concat([combined, missing_frame], axis=1)

    combined = add_channel_differences(combined)
    combined["n_channels_available"] = len(selected_files)

    metadata = [
        "date",
        "ID",
        "timestamp_kst",
        "timestamp_utc",
        "STN_ID",
        "latitude",
        "longitude",
        "altitude",
        "year",
        "month",
        "day",
        "day_of_year",
        "day_sin",
        "day_cos",
        "n_channels_available",
    ]
    feature_columns = [column for column in combined.columns if column not in metadata]
    combined = combined[metadata + feature_columns]

    for column in feature_columns:
        if column.endswith("_missing"):
            combined[column] = combined[column].astype("int8")
        elif pd.api.types.is_numeric_dtype(combined[column]):
            combined[column] = combined[column].astype("float32")
    combined["STN_ID"] = combined["STN_ID"].astype("int16")
    combined["year"] = combined["year"].astype("int16")
    for column in ("month", "day", "n_channels_available"):
        combined[column] = combined[column].astype("int8")
    combined["day_of_year"] = combined["day_of_year"].astype("int16")
    return combined


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_csv(frame: pd.DataFrame, path: str | Path, *, compression=None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_json(payload: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def combine_daily_parquets(
    daily_paths: Iterable[str | Path],
    *,
    output_path: str | Path,
    write_csv_gzip: bool = False,
) -> pd.DataFrame:
    paths = sorted({Path(path) for path in daily_paths})
    if not paths:
        raise ValueError("병합할 날짜별 Parquet 파일이 없습니다.")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame = frame.sort_values(["date", "STN_ID"]).reset_index(drop=True)
    if frame.duplicated(["date", "STN_ID"]).any():
        raise ValueError("병합 결과에 date/STN_ID 중복 행이 있습니다.")
    atomic_write_parquet(frame, output_path)
    if write_csv_gzip:
        csv_path = Path(output_path).with_suffix(".csv.gz")
        atomic_write_csv(frame, csv_path, compression="gzip")
    return frame
