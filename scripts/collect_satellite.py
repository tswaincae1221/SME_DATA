from __future__ import annotations

import argparse

import pandas as pd

from _common import inclusive_dates, require_api_key

from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.data.gk2a import collect_gk2a_range
from gk2a_weather.utils.io import atomic_write_csv
from gk2a_weather.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="GK-2A KO 16채널을 수집합니다.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--channels", nargs="*", help="미지정 시 16채널 전체")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    config = load_yaml(args.config)
    project = config["project"]
    satellite = config["satellite"]
    failures = collect_gk2a_range(
        inclusive_dates(args.start, args.end),
        api_key=require_api_key(),
        raw_dir=resolve_project_path(project["raw_gk2a_dir"]),
        channels=args.channels or satellite["channels"],
        observation_hour_kst=int(config["asos"]["observation_hour_kst"]),
        area=str(satellite["area"]),
        api_time_basis=str(satellite["api_time_basis"]),
        timeout_seconds=float(satellite["timeout_seconds"]),
        max_retries=int(satellite["max_retries"]),
        retry_backoff_seconds=float(satellite["retry_backoff_seconds"]),
        request_interval_seconds=float(satellite["request_interval_seconds"]),
        minimum_file_bytes=int(satellite["minimum_file_bytes"]),
        force=args.force,
    )
    failure_path = resolve_project_path("outputs/gk2a_failures.csv")
    atomic_write_csv(pd.DataFrame(failures), failure_path)
    print(f"완료. 실패 {len(failures)}건, 실패 목록: {failure_path}")


if __name__ == "__main__":
    main()

