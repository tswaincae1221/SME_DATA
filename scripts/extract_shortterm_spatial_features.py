#!/usr/bin/env python3
"""Extract rules-safe station-neighbourhood features from existing LE1B/KO NC files.

The default first spatial experiment uses only the 14:00 KST image and ten
thermal/water-vapour channels.  It writes one Date/STN_ID row with 5 km and
15 km patch statistics.  Missing NC files become NaN and are recorded; an
existing daily checkpoint is reused, so the command can be rerun after API
quota recovery without rebuilding completed dates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import inclusive_dates
from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.data.gk2a import api_timestamp, find_gk2a_file
from gk2a_weather.data.stations import load_station_list
from gk2a_weather.features.satellite import extract_channel_features, load_satellite_array
from gk2a_weather.utils.io import atomic_write_csv


DEFAULT_CHANNELS = [
    "SW038", "WV063", "WV069", "WV073", "IR087",
    "IR096", "IR105", "IR112", "IR123", "IR133",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--time-kst", default="14:00")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--radii-km", nargs=2, type=float, default=[5.0, 15.0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--station-list", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def expected_columns(channel: str, small: float, large: float) -> list[str]:
    small_name = f"r{small:g}km"
    large_name = f"r{large:g}km"
    return [
        f"spatial_{channel}_{small_name}_mean",
        f"spatial_{channel}_{small_name}_std",
        f"spatial_{channel}_{large_name}_mean",
        f"spatial_{channel}_{large_name}_std",
        f"spatial_{channel}_{large_name}_p10",
        f"spatial_{channel}_{large_name}_p90",
        f"spatial_{channel}_{large_name}_valid_ratio",
        f"spatial_{channel}_center_minus_{large_name}",
    ]


def extract_one_channel(
    path: Path,
    channel: str,
    stations_for_extract: pd.DataFrame,
    small: float,
    large: float,
) -> pd.DataFrame:
    array = load_satellite_array(path, channel)
    features = extract_channel_features(
        array,
        stations_for_extract,
        channel=channel,
        radii_km=(0.0, small, large),
    )
    small_name = f"r{small:g}km"
    large_name = f"r{large:g}km"
    mapping = {
        f"{channel}_{small_name}_mean": f"spatial_{channel}_{small_name}_mean",
        f"{channel}_{small_name}_std": f"spatial_{channel}_{small_name}_std",
        f"{channel}_{large_name}_mean": f"spatial_{channel}_{large_name}_mean",
        f"{channel}_{large_name}_std": f"spatial_{channel}_{large_name}_std",
        f"{channel}_{large_name}_p10": f"spatial_{channel}_{large_name}_p10",
        f"{channel}_{large_name}_p90": f"spatial_{channel}_{large_name}_p90",
        f"{channel}_{large_name}_valid_ratio": f"spatial_{channel}_{large_name}_valid_ratio",
        f"{channel}_center_minus_{large_name}": f"spatial_{channel}_center_minus_{large_name}",
    }
    return features[["STN_ID", *mapping]].rename(columns=mapping)


def main() -> None:
    args = parse_args()
    small, large = sorted(float(value) for value in args.radii_km)
    if small <= 0 or large <= small:
        raise ValueError("--radii-km requires two increasing positive radii")
    hour, minute = map(int, args.time_kst.split(":"))
    channels = list(dict.fromkeys(str(channel).upper() for channel in args.channels))

    config = load_yaml(args.config)
    configured = {str(channel).upper() for channel in config["satellite"]["channels"]}
    unknown = set(channels) - configured
    if unknown:
        raise ValueError(f"channels not present in LE1B config: {sorted(unknown)}")
    station_path = args.station_list or config["project"]["station_list"]
    stations = load_station_list(resolve_project_path(station_path)).rename(
        columns={"latitude": "LAT", "longitude": "LON", "altitude": "ALT"}
    )
    stations_for_extract = stations.rename(
        columns={"LAT": "latitude", "LON": "longitude", "ALT": "altitude"}
    )

    root = Path(args.input_root).expanduser().resolve()
    raw_root = root / "raw_gk2a"
    output_dir = Path(args.output_dir).expanduser().resolve()
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    issue_rows: list[dict[str, object]] = []

    for day in inclusive_dates(args.start, args.end):
        date_value = int(day.strftime("%Y%m%d"))
        daily_path = daily_dir / f"spatial_{date_value}.csv"
        if daily_path.exists() and daily_path.stat().st_size > 0 and not args.force:
            print(f"[REUSE] {daily_path}", flush=True)
            continue
        timestamp = day.replace(hour=hour, minute=minute, second=0, microsecond=0).to_pydatetime()
        requested = api_timestamp(timestamp, str(config["satellite"]["api_time_basis"]))
        combined = stations[["STN_ID"]].copy()
        combined.insert(0, "Date", date_value)
        for channel in channels:
            columns = expected_columns(channel, small, large)
            raw_path = find_gk2a_file(raw_root, day, channel, requested)
            if raw_path is None:
                for column in columns:
                    combined[column] = np.nan
                issue_rows.append({
                    "Date": date_value, "channel": channel, "reason": "raw file missing",
                    "expected_timestamp": requested,
                })
                continue
            try:
                features = extract_one_channel(
                    raw_path, channel, stations_for_extract, small, large
                )
                combined = combined.merge(features, on="STN_ID", how="left", validate="1:1")
            except Exception as exc:
                for column in columns:
                    combined[column] = np.nan
                issue_rows.append({
                    "Date": date_value, "channel": channel, "reason": str(exc),
                    "expected_timestamp": requested,
                })
        atomic_write_csv(combined, daily_path)
        print(f"[DONE] {date_value}: {combined.shape} -> {daily_path}", flush=True)

    daily_files = sorted(daily_dir.glob("spatial_*.csv"))
    if not daily_files:
        raise RuntimeError("no daily spatial feature checkpoint was created")
    combined = pd.concat([pd.read_csv(path) for path in daily_files], ignore_index=True)
    combined["Date"] = pd.to_numeric(combined.Date, errors="raise").astype("int64")
    combined["STN_ID"] = pd.to_numeric(combined.STN_ID, errors="raise").astype("int64")
    combined = combined.sort_values(["Date", "STN_ID"]).drop_duplicates(
        ["Date", "STN_ID"], keep="last"
    ).reset_index(drop=True)
    combined_path = output_dir / "spatial_features_combined.csv"
    issue_path = output_dir / "spatial_extraction_issues.csv"
    atomic_write_csv(combined, combined_path)
    atomic_write_csv(
        pd.DataFrame(issue_rows, columns=["Date", "channel", "reason", "expected_timestamp"]),
        issue_path,
    )
    print(f"[COMBINED] {combined.shape} -> {combined_path}", flush=True)
    print(f"[ISSUES] {len(issue_rows)} -> {issue_path}", flush=True)


if __name__ == "__main__":
    main()
