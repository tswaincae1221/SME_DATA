from __future__ import annotations

import argparse

from _common import ROOT

from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.data.stations import load_station_list
from gk2a_weather.features.dataset import build_training_table, read_csv_directory
from gk2a_weather.utils.io import atomic_write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="ASOS+관측소+위성 특징 학습 테이블을 만듭니다.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--without-satellite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    project = config["project"]
    labels = read_csv_directory(resolve_project_path(project["parsed_asos_dir"]))
    stations = load_station_list(resolve_project_path(project["station_list"]))
    satellite = None
    if not args.without_satellite:
        try:
            satellite = read_csv_directory(
                resolve_project_path(project["satellite_feature_dir"])
            )
        except FileNotFoundError:
            print("위성 특징이 없어 B1 정적 기준 데이터셋만 만듭니다.")
    dataset = build_training_table(labels, stations, satellite)
    output_path = resolve_project_path(project["processed_dataset"])
    atomic_write_csv(dataset, output_path)
    print(f"{len(dataset):,}행, {len(dataset.columns)}컬럼 저장: {output_path}")


if __name__ == "__main__":
    main()

