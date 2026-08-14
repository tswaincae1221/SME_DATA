from __future__ import annotations

import argparse

import pandas as pd

from _common import inclusive_dates, require_api_key

from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.data.asos import collect_asos_range
from gk2a_weather.data.stations import load_station_list
from gk2a_weather.utils.io import atomic_write_csv
from gk2a_weather.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="14시 ASOS TA/HM 라벨을 수집합니다.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    config = load_yaml(args.config)
    project = config["project"]
    asos = config["asos"]
    stations = load_station_list(resolve_project_path(project["station_list"]))
    _, failures = collect_asos_range(
        inclusive_dates(args.start, args.end),
        api_key=require_api_key(),
        raw_dir=resolve_project_path(project["raw_asos_dir"]),
        parsed_dir=resolve_project_path(project["parsed_asos_dir"]),
        station_ids=stations["STN_ID"],
        observation_hour_kst=int(asos["observation_hour_kst"]),
        timeout_seconds=float(asos["timeout_seconds"]),
        max_retries=int(asos["max_retries"]),
        retry_backoff_seconds=float(asos["retry_backoff_seconds"]),
        request_interval_seconds=float(asos["request_interval_seconds"]),
        force=args.force,
    )
    failure_path = resolve_project_path("outputs/asos_failures.csv")
    atomic_write_csv(pd.DataFrame({"timestamp": failures}), failure_path)
    print(f"완료. 실패 {len(failures)}건, 실패 목록: {failure_path}")


if __name__ == "__main__":
    main()

