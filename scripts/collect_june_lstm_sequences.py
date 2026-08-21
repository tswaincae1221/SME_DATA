#!/usr/bin/env python3
"""Collect a compact June GK-2A sequence table for the LSTM feasibility test.

The collector is deliberately resumable.  Each successful channel/time image is
reduced immediately to the 96 official stations and cached as a small CSV.  The
raw NetCDF is then removed by default.  A rerun skips valid caches and retries
only missing/failed files.

Competition contract
--------------------
* Only ``GK2A/LE1B/{channel}/KO/data`` is requested.
* ASOS TA/HM are copied from the historical master only as 14:00 labels.
* Official latitude/longitude/altitude come from ``station_list.csv``.
* No ASOS value is used as an input feature.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import submission_predictor as satellite  # noqa: E402


DEFAULT_CHANNELS = [
    "IR087", "IR096", "IR105", "IR112", "IR123", "SW038", "WV069", "WV073",
]
DEFAULT_TIMES = ["12:00", "12:30", "13:00", "13:30", "14:00"]
KEYS = ["Date", "STN_ID"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("KMA_API_KEY"))
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--station-list", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2020, 2026)))
    parser.add_argument("--start-mmdd", default="06-24")
    parser.add_argument("--end-mmdd", default="06-30")
    parser.add_argument("--times", nargs="+", default=DEFAULT_TIMES)
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-interval-seconds", type=float, default=0.25)
    parser.add_argument(
        "--download-1400", action="store_true",
        help="Download 14:00 too. Default: reuse the existing 14:00 master CSV.",
    )
    parser.add_argument(
        "--keep-nc", action="store_true",
        help="Keep raw NetCDF after station extraction (uses considerably more Drive space).",
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="Do not call the API; rebuild tables from current caches/master only.",
    )
    return parser.parse_args()


def hhmm(value: str | int) -> int:
    text = str(value).strip().replace(":", "")
    if len(text) not in (3, 4) or not text.isdigit():
        raise ValueError(f"invalid KST time: {value!r}")
    number = int(text)
    hour, minute = divmod(number, 100)
    if hour > 23 or minute > 59:
        raise ValueError(f"invalid KST time: {value!r}")
    return hour * 100 + minute


def date_range(year: int, start_mmdd: str, end_mmdd: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(f"{year}-{start_mmdd}")
    end = pd.Timestamp(f"{year}-{end_mmdd}")
    if end < start:
        raise ValueError("--end-mmdd must not precede --start-mmdd within a year")
    return list(pd.date_range(start, end, freq="D"))


def api_timestamp(date: pd.Timestamp, time_kst: int) -> str:
    hour, minute = divmod(int(time_kst), 100)
    local = pd.Timestamp(date).normalize() + pd.Timedelta(hours=hour, minutes=minute)
    local = local.tz_localize("Asia/Seoul")
    return local.tz_convert("UTC").strftime("%Y%m%d%H%M")


def cache_path(root: Path, date_value: int, time_kst: int, channel: str) -> Path:
    return root / "station_cache" / str(date_value) / f"{channel}_{time_kst:04d}.csv"


def valid_cache(path: Path, stations: pd.DataFrame, channel: str) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
        expected = ["STN_ID", channel]
        if frame.columns.tolist() != expected or len(frame) != len(stations):
            return False
        got = pd.to_numeric(frame.STN_ID, errors="raise").astype(int).to_numpy()
        return np.array_equal(got, stations.STN_ID.to_numpy(dtype=int))
    except Exception:
        return False


def write_station_cache(
    path: Path, stations: pd.DataFrame, channel: str, values: np.ndarray,
) -> None:
    if len(values) != len(stations):
        raise ValueError(f"{channel}: expected {len(stations)} station values, got {len(values)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pd.DataFrame({"STN_ID": stations.STN_ID.astype(int), channel: values}).to_csv(
        temporary, index=False
    )
    os.replace(temporary, path)


def master_1400_cache(
    master: pd.DataFrame,
    stations: pd.DataFrame,
    date_value: int,
    channel: str,
) -> tuple[np.ndarray, int]:
    rows = master.loc[master.Date.eq(date_value), ["STN_ID", channel]].copy()
    rows = rows.drop_duplicates("STN_ID", keep="last")
    aligned = stations[["STN_ID"]].merge(rows, on="STN_ID", how="left", validate="1:1")
    return aligned[channel].to_numpy(dtype=float), int(aligned[channel].notna().sum())


def sanitise_error(error: Exception, api_key: str | None) -> str:
    text = f"{type(error).__name__}: {error}"
    if api_key:
        text = text.replace(api_key, "***")
    return text


def build_long_table(
    root: Path,
    dates: list[pd.Timestamp],
    times: list[int],
    channels: list[str],
    stations: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    labels = master[["Date", "STN_ID", "TA", "HM"]].drop_duplicates(KEYS, keep="last")
    for date in dates:
        date_value = int(date.strftime("%Y%m%d"))
        for time_kst in times:
            slot = stations.copy()
            slot.insert(0, "TimestampUTC", api_timestamp(date, time_kst))
            slot.insert(0, "TimeKST", time_kst)
            slot.insert(0, "Date", date_value)
            for channel in channels:
                path = cache_path(root, date_value, time_kst, channel)
                if valid_cache(path, stations, channel):
                    values = pd.read_csv(path)[channel].to_numpy(dtype=float)
                else:
                    values = np.full(len(stations), np.nan, dtype=float)
                slot[channel] = values
            slot = slot.merge(labels, on=KEYS, how="left", validate="1:1")
            # TA/HM are 14:00 labels only, never a sequence input.
            if time_kst != 1400:
                slot[["TA", "HM"]] = np.nan
            records.append(slot)
    return pd.concat(records, ignore_index=True).sort_values(
        ["Date", "STN_ID", "TimeKST"]
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    years = sorted(set(args.years))
    channels = list(dict.fromkeys(args.channels))
    times = sorted(set(hhmm(value) for value in args.times))
    unknown = sorted(set(channels) - set(satellite.CHANNELS))
    if unknown:
        raise ValueError(f"unsupported GK-2A channels: {unknown}")
    if 1400 not in times:
        raise ValueError("14:00 must be present because labels are defined at 14:00 KST")
    if not args.build_only and not args.api_key:
        raise ValueError("KMA API key is required unless --build-only is used")

    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stations = satellite.validate_stations(pd.read_csv(args.station_list))
    master = pd.read_csv(args.master_csv)
    required_master = ["Date", "STN_ID", "TA", "HM", *channels]
    missing = [column for column in required_master if column not in master]
    if missing:
        raise ValueError(f"master CSV missing columns: {missing}")
    master["Date"] = pd.to_numeric(master.Date, errors="raise").round().astype("int64")
    master["STN_ID"] = pd.to_numeric(master.STN_ID, errors="raise").round().astype("int64")
    if master.duplicated(KEYS).any():
        raise ValueError("master CSV contains duplicate Date/STN_ID rows")

    dates = [date for year in years for date in date_range(year, args.start_mmdd, args.end_mmdd)]
    inventory: list[dict[str, object]] = []
    api_calls = 0
    stop_error: Exception | None = None

    if args.build_only and not args.download_1400:
        for date in dates:
            date_value = int(date.strftime("%Y%m%d"))
            for channel in channels:
                path = cache_path(root, date_value, 1400, channel)
                if not valid_cache(path, stations, channel):
                    values, _ = master_1400_cache(master, stations, date_value, channel)
                    write_station_cache(path, stations, channel, values)

    if not args.build_only:
        import requests

        with requests.Session() as session:
            for date in dates:
                date_value = int(date.strftime("%Y%m%d"))
                for time_kst in times:
                    for channel in channels:
                        path = cache_path(root, date_value, time_kst, channel)
                        api_date = api_timestamp(date, time_kst)
                        row = {
                            "Date": date_value, "TimeKST": time_kst,
                            "TimestampUTC": api_date, "Channel": channel,
                            "Status": "", "ValidStations": 0, "Error": "",
                        }
                        if valid_cache(path, stations, channel):
                            cached = pd.read_csv(path)[channel]
                            row.update(Status="cached", ValidStations=int(cached.notna().sum()))
                            inventory.append(row)
                            continue
                        if time_kst == 1400 and not args.download_1400:
                            values, valid_count = master_1400_cache(
                                master, stations, date_value, channel
                            )
                            write_station_cache(path, stations, channel, values)
                            row.update(Status="master_1400", ValidStations=valid_count)
                            inventory.append(row)
                            continue

                        nc_path = root / "raw_nc" / str(date_value) / f"{channel}_{api_date}.nc"
                        existed_before = nc_path.exists() and nc_path.stat().st_size >= 10_000
                        try:
                            satellite.download_nc(
                                session=session,
                                api_key=str(args.api_key),
                                channel=channel,
                                api_date=api_date,
                                destination=nc_path,
                                timeout_seconds=args.timeout_seconds,
                                max_retries=args.max_retries,
                            )
                            if not existed_before:
                                api_calls += 1
                            values = satellite.load_nc_station_values(nc_path, channel, stations)
                            write_station_cache(path, stations, channel, values)
                            row.update(Status="downloaded", ValidStations=int(np.isfinite(values).sum()))
                            if not args.keep_nc:
                                nc_path.unlink(missing_ok=True)
                        except Exception as exc:
                            row.update(Status="failed", Error=sanitise_error(exc, args.api_key))
                            # Remove only a bad raw cache; completed station caches are untouched.
                            if nc_path.exists() and not valid_cache(path, stations, channel):
                                nc_path.unlink(missing_ok=True)
                            inventory.append(row)
                            if "HTTP 429" in str(exc):
                                stop_error = RuntimeError(
                                    "KMA API quota/rate limit (HTTP 429). The completed caches are safe; "
                                    "rerun the same Colab cell after the quota resets."
                                )
                                break
                            print(
                                f"[FAILED] {date_value} {time_kst:04d} {channel}: {row['Error']}",
                                flush=True,
                            )
                        else:
                            inventory.append(row)
                        if args.request_interval_seconds > 0 and not existed_before:
                            time.sleep(args.request_interval_seconds)
                    if stop_error is not None:
                        break
                if stop_error is not None:
                    break

    # Build a complete grid even after interruption, so missing cells remain visible.
    long_table = build_long_table(root, dates, times, channels, stations, master)
    suffix = f"{years[0]}to{years[-1]}"
    table_path = root / f"june_shortterm_core8_{suffix}.csv"
    long_table.to_csv(table_path, index=False)

    if inventory:
        inventory_frame = pd.DataFrame(inventory)
    else:
        rows = []
        for date in dates:
            date_value = int(date.strftime("%Y%m%d"))
            for time_kst in times:
                for channel in channels:
                    path = cache_path(root, date_value, time_kst, channel)
                    rows.append({
                        "Date": date_value, "TimeKST": time_kst,
                        "TimestampUTC": api_timestamp(date, time_kst), "Channel": channel,
                        "Status": "cached" if valid_cache(path, stations, channel) else "missing",
                        "ValidStations": int(pd.read_csv(path)[channel].notna().sum())
                        if valid_cache(path, stations, channel) else 0,
                        "Error": "",
                    })
        inventory_frame = pd.DataFrame(rows)
    inventory_path = root / "collection_inventory.csv"
    inventory_frame.to_csv(inventory_path, index=False)

    expected = len(dates) * len(times) * len(channels)
    complete = 0
    missing_rows = []
    for date in dates:
        date_value = int(date.strftime("%Y%m%d"))
        for time_kst in times:
            for channel in channels:
                path = cache_path(root, date_value, time_kst, channel)
                if valid_cache(path, stations, channel):
                    complete += 1
                else:
                    missing_rows.append({
                        "Date": date_value, "TimeKST": time_kst,
                        "TimestampUTC": api_timestamp(date, time_kst), "Channel": channel,
                    })
    pd.DataFrame(
        missing_rows, columns=["Date", "TimeKST", "TimestampUTC", "Channel"]
    ).to_csv(root / "missing_inventory.csv", index=False)
    summary = {
        "years": years,
        "dates": len(dates),
        "stations": len(stations),
        "times_kst": times,
        "channels": channels,
        "expected_channel_files": expected,
        "complete_channel_files": complete,
        "missing_channel_files": expected - complete,
        "completion_fraction": complete / max(1, expected),
        "api_calls_this_run": api_calls,
        "long_rows": len(long_table),
        "long_table": str(table_path),
        "raw_nc_kept": bool(args.keep_nc),
        "rules_contract": {
            "path": "/GK2A/LE1B/{channel}/KO/data",
            "area": "KO",
            "asos_used_as_input": False,
            "official_station_coordinates": True,
        },
    }
    (root / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if stop_error is not None:
        raise stop_error


if __name__ == "__main__":
    main()
