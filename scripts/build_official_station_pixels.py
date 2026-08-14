"""운영진 station_list.csv에서 GK-2A 16채널 픽셀 좌표를 계산한다.

이 파일은 검수·시각화용이다. 실제 전처리는 같은 함수를 이용해 공식
LAT/LON을 채널 원본 격자에 직접 투영하므로 외부 관측소 메타데이터가 없다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT

from gk2a_weather.constants import GK2A_CHANNELS
from gk2a_weather.data.stations import load_station_list
from gk2a_weather.features.satellite import (
    CHANNEL_RESOLUTION_KM,
    GRID_SPECS,
    official_station_pixels,
)
from gk2a_weather.utils.io import atomic_write_csv, atomic_write_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-list", default="data/metadata/station_list.csv"
    )
    parser.add_argument(
        "--output", default="data/metadata/station_pixels_official_16ch.csv"
    )
    parser.add_argument(
        "--report", default="data/metadata/station_pixels_official_validation.txt"
    )
    args = parser.parse_args()

    station_path = (ROOT / args.station_list).resolve()
    stations = load_station_list(station_path)
    if len(stations) != 96:
        raise ValueError(f"공식 관측소는 96개여야 합니다. 현재 {len(stations)}개")

    result = stations.copy()
    report_lines = [
        "Official station_list -> GK-2A LE1B KO pixel validation",
        f"stations={len(stations)}",
        "source_columns=STN_ID,LAT,LON,ALT (competition official)",
    ]
    for channel in GK2A_CHANNELS:
        resolution = CHANNEL_RESOLUTION_KM[channel]
        shape = tuple(GRID_SPECS[resolution]["shape"])
        pixels = official_station_pixels(stations, channel, shape)
        result[f"{channel}_row"] = pixels["row"].round(3)
        result[f"{channel}_col"] = pixels["col"].round(3)
        report_lines.append(
            f"{channel}: resolution={resolution:g}km shape={shape} inside=96/96"
        )

    output_path = (ROOT / args.output).resolve()
    report_path = (ROOT / args.report).resolve()
    atomic_write_csv(result, output_path)
    atomic_write_text(report_path, "\n".join(report_lines) + "\n")
    print(f"공식 좌표 기반 픽셀 매핑 저장: {output_path}")
    print(f"검증 보고서 저장: {report_path}")


if __name__ == "__main__":
    main()
