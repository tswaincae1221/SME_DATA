from __future__ import annotations

import argparse

from _common import ROOT

from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.data.stations import load_station_list


def main() -> None:
    parser = argparse.ArgumentParser(description="필수 파일과 설정을 검사합니다.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    station_path = resolve_project_path(config["project"]["station_list"])
    submission_path = resolve_project_path(config["project"]["sample_submission"])
    stations = load_station_list(station_path)
    if len(stations) != 96:
        print(f"[주의] 관측소가 96개가 아니라 {len(stations)}개입니다.")
    if not submission_path.exists():
        raise SystemExit(f"sample_submission.csv가 없습니다: {submission_path}")
    print(f"프로젝트 경로: {ROOT}")
    print(f"관측소: {len(stations)}개")
    print(f"위성 API 시간 기준: {config['satellite']['api_time_basis']}")
    print("기본 파일 검사가 끝났습니다.")


if __name__ == "__main__":
    main()

