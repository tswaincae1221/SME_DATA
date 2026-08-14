from __future__ import annotations

import pandas as pd

from gk2a_weather.features.dataset import build_training_table


def test_build_training_table() -> None:
    labels = pd.DataFrame(
        {
            "date": ["2025-08-24", "2025-08-24"],
            "STN_ID": [108, 159],
            "TA": [30.0, 29.0],
            "HM": [60.0, 65.0],
        }
    )
    stations = pd.DataFrame(
        {
            "STN_ID": [108, 159],
            "latitude": [37.5, 35.1],
            "longitude": [127.0, 129.0],
            "altitude": [85.0, 70.0],
        }
    )
    satellite = pd.DataFrame(
        {
            "date": ["2025-08-24", "2025-08-24"],
            "STN_ID": [108, 159],
            "IR105_center_mean": [290.0, 291.0],
            "IR112_center_mean": [289.0, 290.5],
        }
    )
    result = build_training_table(labels, stations, satellite)
    assert len(result) == 2
    assert "year" in result
    assert "day" in result
    assert "day_sin" in result
    assert result.loc[0, "IR105_minus_IR112"] == 1.0
