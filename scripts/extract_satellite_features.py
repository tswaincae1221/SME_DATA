from __future__ import annotations

import argparse

import pandas as pd

from _common import inclusive_dates

from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.data.gk2a import api_timestamp, find_gk2a_file
from gk2a_weather.data.stations import load_station_list
from gk2a_weather.features.satellite import extract_channel_features, load_satellite_array
from gk2a_weather.utils.io import atomic_write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="다운로드한 위성에서 관측소 패치 특징을 추출합니다.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--station-list",
        help="기본값은 data.yaml의 운영진 제공 station_list.csv",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    project, satellite = config["project"], config["satellite"]
    station_path = args.station_list or project["station_list"]
    stations = load_station_list(resolve_project_path(station_path))
    raw_root = resolve_project_path(project["raw_gk2a_dir"])
    output_root = resolve_project_path(project["satellite_feature_dir"])
    radii_km = tuple(float(value) for value in satellite["patch_radii_km"])

    for day in inclusive_dates(args.start, args.end):
        timestamp = day.replace(
            hour=int(config["asos"]["observation_hour_kst"]), minute=0, second=0
        ).to_pydatetime()
        requested = api_timestamp(timestamp, str(satellite["api_time_basis"]))
        combined = stations[["STN_ID"]].copy()
        combined.insert(0, "date", day.strftime("%Y-%m-%d"))
        used_channels = 0
        for channel in satellite["channels"]:
            raw_path = find_gk2a_file(raw_root, day, str(channel), requested)
            if raw_path is None:
                continue
            array = load_satellite_array(raw_path, str(channel))
            features = extract_channel_features(
                array, stations, channel=str(channel), radii_km=radii_km
            )
            combined = combined.merge(features, on="STN_ID", how="left")
            used_channels += 1
        output_path = output_root / f"satellite_{day.strftime('%Y%m%d')}.csv"
        atomic_write_csv(combined, output_path)
        print(f"{day.date()}: {used_channels}개 채널 -> {output_path}")


if __name__ == "__main__":
    main()
