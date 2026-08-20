"""Standalone incremental GK-2A short-term table builder.

This script intentionally does not depend on the older phased short-term pipeline.
It scans whatever raw files currently exist, stores one small station-value cache per
NC file, writes NaN for unavailable channel/time slots, and can later download only
the missing files before rebuilding the tables from the updated cache.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = (
    "VI004", "VI005", "VI006", "VI008", "NR013", "NR016", "SW038", "WV063",
    "WV069", "WV073", "IR087", "IR096", "IR105", "IR112", "IR123", "IR133",
)
GK2A_URL = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{channel}/KO/data"
KST = timezone(timedelta(hours=9))
UTC = timezone.utc
FILE_RE = re.compile(
    r"(?P<channel>VI004|VI005|VI006|VI008|NR013|NR016|SW038|WV063|WV069|WV073|"
    r"IR087|IR096|IR105|IR112|IR123|IR133)[^0-9]*(?P<timestamp>[0-9]{12})",
    re.IGNORECASE,
)

CHANNEL_RESOLUTION_KM = {
    "VI006": 0.5,
    "VI004": 1.0,
    "VI005": 1.0,
    "VI008": 1.0,
    "NR013": 2.0,
    "NR016": 2.0,
    "SW038": 2.0,
    "WV063": 2.0,
    "WV069": 2.0,
    "WV073": 2.0,
    "IR087": 2.0,
    "IR096": 2.0,
    "IR105": 2.0,
    "IR112": 2.0,
    "IR123": 2.0,
    "IR133": 2.0,
}
GRID_SPECS = {
    0.5: (3600, 3600, 500.0, -899750.0, 899750.0),
    1.0: (1800, 1800, 1000.0, -899500.0, 899500.0),
    2.0: (900, 900, 2000.0, -899000.0, 899000.0),
}
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_bytes(content: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_npz(destination: Path, **arrays: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_stations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(
        columns={
            "STN": "STN_ID", "stn": "STN_ID", "stn_id": "STN_ID",
            "latitude": "LAT", "lat": "LAT", "longitude": "LON", "lon": "LON",
            "altitude": "ALT", "alt": "ALT", "HT": "ALT",
        }
    )
    required = ["STN_ID", "LAT", "LON", "ALT"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"station_list.csv 필수 컬럼 누락: {missing}")
    result = frame[required].copy()
    result["STN_ID"] = pd.to_numeric(result["STN_ID"], errors="raise").astype(int)
    for column in ("LAT", "LON", "ALT"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    if result["STN_ID"].duplicated().any() or result.isna().any().any():
        raise ValueError("station_list.csv에 중복 지점 또는 결측값이 있습니다.")
    return result.sort_values("STN_ID").reset_index(drop=True)


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = map(int, value.split(":"))
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"잘못된 시각: {value}")
    return hour, minute


def expected_inventory(args: argparse.Namespace, raw_root: Path) -> pd.DataFrame:
    if args.step_minutes <= 0:
        raise ValueError("--step-minutes는 1 이상이어야 합니다.")
    start_hour, start_minute = parse_hhmm(args.start_time)
    end_hour, end_minute = parse_hhmm(args.end_time)
    if (start_hour, start_minute) > (end_hour, end_minute):
        raise ValueError("--start-time은 --end-time보다 늦을 수 없습니다.")
    rows: list[dict[str, object]] = []
    for year in sorted(dict.fromkeys(args.years)):
        start_day = pd.Timestamp(f"{year}-{args.start_mmdd}")
        end_day = pd.Timestamp(f"{year}-{args.end_mmdd}")
        for day in pd.date_range(start_day, end_day, freq="D"):
            current = day.replace(hour=start_hour, minute=start_minute)
            end = day.replace(hour=end_hour, minute=end_minute)
            while current <= end:
                aware_kst = current.to_pydatetime().replace(tzinfo=KST)
                timestamp_utc = aware_kst.astimezone(UTC).strftime("%Y%m%d%H%M")
                for channel in CHANNELS:
                    expected_path = (
                        raw_root / f"{year:04d}" / day.strftime("%m") / day.strftime("%d")
                        / f"{channel}_{timestamp_utc}.nc"
                    )
                    rows.append(
                        {
                            "Year": year,
                            "Date": int(day.strftime("%Y%m%d")),
                            "TimeKST": int(current.strftime("%H%M")),
                            "TimestampUTC": timestamp_utc,
                            "Channel": channel,
                            "ExpectedPath": str(expected_path),
                        }
                    )
                current += pd.Timedelta(minutes=args.step_minutes)
    return pd.DataFrame(rows)


def scan_raw_files(raw_root: Path, minimum_file_bytes: int) -> tuple[dict[tuple[str, str], Path], list[str]]:
    selected: dict[tuple[str, str], Path] = {}
    duplicates: list[str] = []
    if not raw_root.exists():
        return selected, duplicates
    for path in raw_root.rglob("*.nc"):
        match = FILE_RE.search(path.stem)
        if not match:
            continue
        key = (match.group("channel").upper(), match.group("timestamp"))
        if path.stat().st_size < minimum_file_bytes:
            continue
        previous = selected.get(key)
        if previous is None:
            selected[key] = path
        else:
            winner = max((previous, path), key=lambda item: (item.stat().st_size, str(item)))
            loser = path if winner == previous else previous
            selected[key] = winner
            duplicates.append(f"{key[0]} {key[1]}: selected={winner}, ignored={loser}")
    return selected, duplicates


def attach_sources(
    expected: pd.DataFrame,
    raw_root: Path,
    minimum_file_bytes: int,
) -> tuple[pd.DataFrame, list[str]]:
    available, duplicates = scan_raw_files(raw_root, minimum_file_bytes)
    inventory = expected.copy()
    source_paths: list[str] = []
    source_sizes: list[int] = []
    statuses: list[str] = []
    for row in inventory.itertuples(index=False):
        path = available.get((row.Channel, row.TimestampUTC))
        if path is None:
            source_paths.append("")
            source_sizes.append(0)
            statuses.append("missing")
        else:
            source_paths.append(str(path))
            source_sizes.append(path.stat().st_size)
            statuses.append("available")
    inventory["SourcePath"] = source_paths
    inventory["SourceSize"] = source_sizes
    inventory["Status"] = statuses
    return inventory, duplicates


def lcc_forward(longitude: float, latitude: float) -> tuple[float, float]:
    eccentricity = math.sqrt(WGS84_F * (2.0 - WGS84_F))

    def m(phi: float) -> float:
        return math.cos(phi) / math.sqrt(1.0 - eccentricity**2 * math.sin(phi) ** 2)

    def t(phi: float) -> float:
        ratio = (1.0 - eccentricity * math.sin(phi)) / (1.0 + eccentricity * math.sin(phi))
        return math.tan(math.pi / 4.0 - phi / 2.0) / ratio ** (eccentricity / 2.0)

    phi1, phi2, phi0 = map(math.radians, (30.0, 60.0, 38.0))
    lambda0 = math.radians(126.0)
    phi, lam = math.radians(latitude), math.radians(longitude)
    n_value = (math.log(m(phi1)) - math.log(m(phi2))) / (math.log(t(phi1)) - math.log(t(phi2)))
    f_value = m(phi1) / (n_value * t(phi1) ** n_value)
    rho0 = WGS84_A * f_value * t(phi0) ** n_value
    rho = WGS84_A * f_value * t(phi) ** n_value
    theta = n_value * (lam - lambda0)
    return rho * math.sin(theta), rho0 - rho * math.cos(theta)


def station_pixels(stations: pd.DataFrame, channel: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    resolution = CHANNEL_RESOLUTION_KM[channel]
    height, width, pixel_size, upper_left_easting, upper_left_northing = GRID_SPECS[resolution]
    if tuple(shape) != (height, width):
        raise ValueError(f"{channel} shape={shape}, expected={(height, width)} (LE1B/KO 확인 필요)")
    rows, cols = [], []
    for station in stations.itertuples(index=False):
        x_coord, y_coord = lcc_forward(float(station.LON), float(station.LAT))
        rows.append(int(round((upper_left_northing - y_coord) / pixel_size)))
        cols.append(int(round((x_coord - upper_left_easting) / pixel_size)))
    row_array = np.asarray(rows, dtype=int)
    col_array = np.asarray(cols, dtype=int)
    if not ((row_array >= 0).all() and (row_array < height).all() and (col_array >= 0).all() and (col_array < width).all()):
        raise ValueError(f"{channel}: 공식 관측소 좌표가 KO 격자 밖에 있습니다.")
    return row_array, col_array


def load_nc_array(path: Path, channel: str) -> np.ndarray:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray와 h5netcdf/netCDF4가 필요합니다.") from exc
    with xr.open_dataset(path, mask_and_scale=True, decode_times=False) as dataset:
        preferred = (channel, channel.lower(), "image_pixel_values", "image")
        variable_name = next(
            (name for name in preferred if name in dataset.data_vars and dataset[name].ndim >= 2),
            None,
        )
        if variable_name is None:
            candidates = [
                (name, variable.size)
                for name, variable in dataset.data_vars.items()
                if variable.ndim == 2 and np.issubdtype(variable.dtype, np.number)
            ]
            if not candidates:
                raise ValueError(f"2차원 영상 변수가 없습니다: {list(dataset.data_vars)}")
            variable_name = max(candidates, key=lambda item: item[1])[0]
        array = dataset[variable_name].squeeze().to_numpy()
    if array.ndim != 2:
        raise ValueError(f"2차원 영상이 아닙니다: shape={array.shape}")
    return np.asarray(array, dtype=float)


def cache_path(cache_root: Path, channel: str, timestamp_utc: str) -> Path:
    return cache_root / channel / f"{timestamp_utc}.npz"


def cache_matches(path: Path, source: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as cached:
            return (
                int(cached["source_size"][0]) == source.stat().st_size
                and int(cached["source_mtime_ns"][0]) == source.stat().st_mtime_ns
            )
    except Exception:
        return False


def extract_to_cache(source: Path, destination: Path, channel: str, stations: pd.DataFrame) -> None:
    array = load_nc_array(source, channel)
    rows, cols = station_pixels(stations, channel, array.shape)
    values = array[rows, cols].astype(float)
    stat = source.stat()
    atomic_npz(
        destination,
        stn_id=stations["STN_ID"].to_numpy(dtype=int),
        values=values,
        source_size=np.asarray([stat.st_size], dtype=np.int64),
        source_mtime_ns=np.asarray([stat.st_mtime_ns], dtype=np.int64),
    )


def update_feature_cache(
    inventory: pd.DataFrame,
    cache_root: Path,
    stations: pd.DataFrame,
    rebuild_cache: bool,
    max_process_files: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    process_count = 0
    available = inventory[inventory["Status"] == "available"]
    for item in available.itertuples(index=False):
        source = Path(item.SourcePath)
        destination = cache_path(cache_root, item.Channel, item.TimestampUTC)
        if not rebuild_cache and cache_matches(destination, source):
            rows.append({"Channel": item.Channel, "TimestampUTC": item.TimestampUTC, "Status": "cached", "Error": ""})
            continue
        if max_process_files and process_count >= max_process_files:
            rows.append({"Channel": item.Channel, "TimestampUTC": item.TimestampUTC, "Status": "deferred", "Error": "max_process_files"})
            continue
        try:
            extract_to_cache(source, destination, item.Channel, stations)
            status, error = "processed", ""
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
        rows.append({"Channel": item.Channel, "TimestampUTC": item.TimestampUTC, "Status": status, "Error": error})
        process_count += 1
        if process_count % 25 == 0:
            print(f"[CACHE] 새로 처리한 NC: {process_count}", flush=True)
    return pd.DataFrame(rows, columns=["Channel", "TimestampUTC", "Status", "Error"])


def read_label_file(path: Path, stations: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["STN_ID", "TA", "HM"])
    frame = pd.read_csv(path).rename(columns={"STN": "STN_ID", "stn": "STN_ID"})
    required = ["STN_ID", "TA", "HM"]
    if any(column not in frame for column in required):
        return pd.DataFrame(columns=required)
    result = frame[required].copy()
    result["STN_ID"] = pd.to_numeric(result["STN_ID"], errors="coerce")
    result["TA"] = pd.to_numeric(result["TA"], errors="coerce")
    result["HM"] = pd.to_numeric(result["HM"], errors="coerce")
    result = result.dropna(subset=["STN_ID"]).drop_duplicates("STN_ID", keep="last")
    result["STN_ID"] = result["STN_ID"].astype(int)
    return result[result["STN_ID"].isin(stations["STN_ID"])]


def build_labels(expected: pd.DataFrame, data_root: Path, stations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_parts: list[pd.DataFrame] = []
    missing_rows: list[dict[str, object]] = []
    for year, date_value in expected[["Year", "Date"]].drop_duplicates().itertuples(index=False):
        path = data_root / "asos" / "parsed" / f"asos_{int(date_value)}1400.csv"
        label = read_label_file(path, stations)
        base = stations[["STN_ID"]].copy()
        base.insert(0, "Date", int(date_value))
        base.insert(0, "Year", int(year))
        base = base.merge(label, on="STN_ID", how="left", validate="1:1")
        label_parts.append(base)
        if not len(label):
            missing_rows.append({"Year": year, "Date": date_value, "ExpectedPath": str(path), "Reason": "label file missing or invalid"})
    return pd.concat(label_parts, ignore_index=True), pd.DataFrame(missing_rows)


def load_cached_values(path: Path, station_ids: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            cached_ids = cached["stn_id"].astype(int)
            values = cached["values"].astype(float)
        if not np.array_equal(cached_ids, station_ids):
            return None
        return values
    except Exception:
        return None


def build_tables(
    inventory: pd.DataFrame,
    data_root: Path,
    cache_root: Path,
    stations: pd.DataFrame,
    output_dir: Path,
    valid_cache_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Path]:
    labels, missing_labels = build_labels(inventory, data_root, stations)
    station_ids = stations["STN_ID"].to_numpy(dtype=int)
    parts: list[pd.DataFrame] = []
    slots = inventory[["Year", "Date", "TimeKST", "TimestampUTC"]].drop_duplicates()
    available_keys = {
        (row.Channel, row.TimestampUTC)
        for row in inventory[inventory["Status"] == "available"].itertuples(index=False)
    }
    if valid_cache_keys is not None:
        available_keys &= valid_cache_keys
    for slot in slots.itertuples(index=False):
        frame = stations.copy()
        frame.insert(0, "TimestampUTC", slot.TimestampUTC)
        frame.insert(0, "TimeKST", int(slot.TimeKST))
        frame.insert(0, "Date", int(slot.Date))
        frame.insert(0, "Year", int(slot.Year))
        for channel in CHANNELS:
            values = None
            if (channel, slot.TimestampUTC) in available_keys:
                values = load_cached_values(cache_path(cache_root, channel, slot.TimestampUTC), station_ids)
            frame[channel] = values if values is not None else np.nan
        frame["available_channel_count"] = frame[list(CHANNELS)].notna().sum(axis=1).astype(int)
        frame["TA"] = np.nan
        frame["HM"] = np.nan
        if int(slot.TimeKST) == 1400:
            frame = frame.drop(columns=["TA", "HM"]).merge(
                labels[["Year", "Date", "STN_ID", "TA", "HM"]],
                on=["Year", "Date", "STN_ID"], how="left", validate="1:1",
            )
        parts.append(frame)
    long_df = pd.concat(parts, ignore_index=True).sort_values(["Date", "STN_ID", "TimeKST"])

    pivot = long_df.pivot(index=["Year", "Date", "STN_ID"], columns="TimeKST", values=list(CHANNELS))
    pivot.columns = [f"{channel}_{int(hhmm):04d}" for channel, hhmm in pivot.columns]
    wide_df = pivot.reset_index().merge(stations, on="STN_ID", how="left", validate="m:1")
    wide_df = wide_df.merge(labels, on=["Year", "Date", "STN_ID"], how="left", validate="1:1")

    year_tag = f"{int(inventory['Year'].min())}to{int(inventory['Year'].max())}"
    paths = {
        "long": output_dir / f"shortterm_long_{year_tag}.csv",
        "wide": output_dir / f"shortterm_wide_{year_tag}.csv",
        "labels": output_dir / f"shortterm_labels_1400_{year_tag}.csv",
        "missing_nc": output_dir / f"missing_nc_inventory_{year_tag}.csv",
        "missing_labels": output_dir / f"missing_label_inventory_{year_tag}.csv",
        "summary": output_dir / f"incremental_summary_{year_tag}.csv",
    }
    missing_nc = inventory[inventory["Status"] != "available"].copy()
    summary = pd.DataFrame(
        [
            {
                "expected_nc": len(inventory),
                "available_nc": int((inventory["Status"] == "available").sum()),
                "missing_nc": len(missing_nc),
                "expected_label_files": inventory[["Year", "Date"]].drop_duplicates().shape[0],
                "missing_label_files": len(missing_labels),
                "long_rows": len(long_df),
                "wide_rows": len(wide_df),
                "label_rows": len(labels),
                "long_nan_cells": int(long_df[list(CHANNELS)].isna().sum().sum()),
            }
        ]
    )
    for frame, key in (
        (long_df, "long"), (wide_df, "wide"), (labels, "labels"),
        (missing_nc, "missing_nc"), (missing_labels, "missing_labels"), (summary, "summary"),
    ):
        atomic_csv(frame, paths[key])
    return paths


def looks_like_error_document(content: bytes, content_type: str) -> bool:
    prefix = content[:256].lstrip().lower()
    return (
        "text/html" in content_type.lower()
        or "application/json" in content_type.lower()
        or prefix.startswith((b"<html", b"<!doctype", b"{", b"["))
    )


def download_one(
    *,
    channel: str,
    timestamp_utc: str,
    destination: Path,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    minimum_file_bytes: int,
) -> tuple[bool, str]:
    import requests

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                GK2A_URL.format(channel=channel),
                params={"date": timestamp_utc, "authKey": api_key},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            content = response.content
            if looks_like_error_document(content, response.headers.get("content-type", "")):
                raise RuntimeError("API 오류 문서 응답")
            if len(content) < minimum_file_bytes:
                raise RuntimeError(f"응답 크기 부족: {len(content)} bytes")
            atomic_bytes(content, destination)
            return True, ""
        except Exception as exc:
            last_error = f"attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
    return False, last_error


def retry_missing(
    inventory: pd.DataFrame,
    api_key: str,
    max_downloads: int,
    request_interval: float,
    timeout_seconds: float,
    max_retries: int,
    minimum_file_bytes: int,
    output_dir: Path,
) -> pd.DataFrame:
    missing = inventory[inventory["Status"] != "available"].copy()
    if max_downloads > 0:
        missing = missing.head(max_downloads)
    rows: list[dict[str, object]] = []
    for number, item in enumerate(missing.itertuples(index=False), start=1):
        destination = Path(item.ExpectedPath)
        success, error = download_one(
            channel=item.Channel,
            timestamp_utc=item.TimestampUTC,
            destination=destination,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            minimum_file_bytes=minimum_file_bytes,
        )
        rows.append(
            {
                "Year": item.Year, "Date": item.Date, "TimeKST": item.TimeKST,
                "TimestampUTC": item.TimestampUTC, "Channel": item.Channel,
                "Success": success, "Destination": str(destination), "Error": error,
            }
        )
        print(f"[RETRY {number}/{len(missing)}] {item.Channel} {item.TimestampUTC}: {'OK' if success else 'FAIL'}", flush=True)
        if request_interval > 0:
            time.sleep(request_interval)
    result = pd.DataFrame(rows)
    atomic_csv(result, output_dir / "latest_retry_results.csv")
    return result


def sync_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_root / "incremental_12to14_tables"
    raw_root = data_root / "raw_gk2a"
    cache_root = output_dir / "feature_cache"
    stations = load_stations(Path(args.station_list).expanduser().resolve())
    expected = expected_inventory(args, raw_root)
    inventory, duplicates = attach_sources(expected, raw_root, args.minimum_file_bytes)
    print(
        f"[INVENTORY] available={(inventory['Status'] == 'available').sum()}/{len(inventory)}, "
        f"missing={(inventory['Status'] != 'available').sum()}",
        flush=True,
    )
    atomic_csv(inventory, output_dir / "source_inventory.csv")
    atomic_csv(pd.DataFrame({"duplicate": duplicates}), output_dir / "duplicate_sources.csv")
    cache_status = update_feature_cache(
        inventory, cache_root, stations, args.rebuild_cache, args.max_process_files
    )
    atomic_csv(cache_status, output_dir / "latest_cache_status.csv")
    error_count = int((cache_status["Status"] == "error").sum()) if len(cache_status) else 0
    deferred_count = int((cache_status["Status"] == "deferred").sum()) if len(cache_status) else 0
    print(f"[CACHE STATUS] errors={error_count}, deferred={deferred_count}", flush=True)
    valid_cache_keys = {
        (row.Channel, row.TimestampUTC)
        for row in cache_status[cache_status["Status"].isin(["cached", "processed"])].itertuples(index=False)
    }
    paths = build_tables(
        inventory, data_root, cache_root, stations, output_dir,
        valid_cache_keys=valid_cache_keys,
    )
    print("[SYNC COMPLETE]", flush=True)
    for name, path in paths.items():
        print(f"  {name}: {path}", flush=True)
    return paths


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and incrementally refresh short-term GK-2A station tables")
    parser.add_argument("--phase", choices=["inventory", "sync", "retry", "retry-and-sync"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--station-list", default="data/metadata/station_list.csv")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2026)))
    parser.add_argument("--start-mmdd", default="08-24")
    parser.add_argument("--end-mmdd", default="08-30")
    parser.add_argument("--start-time", default="12:00")
    parser.add_argument("--end-time", default="14:00")
    parser.add_argument("--step-minutes", type=int, default=10)
    parser.add_argument("--minimum-file-bytes", type=int, default=10_000)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-process-files", type=int, default=0)
    parser.add_argument("--max-downloads", type=int, default=200)
    parser.add_argument("--request-interval", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--api-key-env", default="KMA_API_KEY")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_root / "incremental_12to14_tables"
    raw_root = data_root / "raw_gk2a"
    expected = expected_inventory(args, raw_root)
    inventory, duplicates = attach_sources(expected, raw_root, args.minimum_file_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(inventory, output_dir / "source_inventory.csv")
    atomic_csv(pd.DataFrame({"duplicate": duplicates}), output_dir / "duplicate_sources.csv")

    if args.phase == "inventory":
        print(inventory["Status"].value_counts(dropna=False).to_string())
        return
    if args.phase in {"retry", "retry-and-sync"}:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"환경변수 {args.api_key_env}에 KMA API 키를 설정하세요.")
        retry_missing(
            inventory, api_key, args.max_downloads, args.request_interval,
            args.timeout_seconds, args.max_retries, args.minimum_file_bytes, output_dir,
        )
        if args.phase == "retry":
            return
    sync_pipeline(args)


if __name__ == "__main__":
    main()
