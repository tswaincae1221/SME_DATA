from __future__ import annotations

import pandas as pd

from gk2a_weather.data.stations import load_station_list


def test_load_official_station_list(tmp_path) -> None:
    path = tmp_path / "station_list.csv"
    pd.DataFrame(
        {
            "STN_ID": [90, 108],
            "LAT": [38.25085, 37.57142],
            "LON": [128.56473, 126.96580],
            "ALT": [17.53, 85.67],
        }
    ).to_csv(path, index=False)
    result = load_station_list(path)
    assert result.columns.tolist() == [
        "STN_ID",
        "latitude",
        "longitude",
        "altitude",
    ]
    assert result["STN_ID"].tolist() == [90, 108]
