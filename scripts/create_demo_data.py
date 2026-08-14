"""API 없이 팀원 셋이 전체 코드 흐름을 연습할 합성 데이터를 만든다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ROOT

from gk2a_weather.features.static import add_calendar_features
from gk2a_weather.utils.io import atomic_write_csv


def main() -> None:
    rng = np.random.default_rng(42)
    stations = pd.DataFrame(
        {
            "STN_ID": np.arange(100, 112),
            "latitude": np.linspace(33.4, 38.0, 12),
            "longitude": np.linspace(126.3, 129.3, 12),
            "altitude": np.linspace(5, 450, 12),
            "row": np.arange(12) + 20,
            "col": np.arange(12) + 30,
        }
    )
    dates = pd.date_range("2025-07-01", periods=42, freq="D")
    records: list[dict[str, float | int | str]] = []
    for day_index, day in enumerate(dates):
        seasonal = np.sin(day_index / 8)
        for station in stations.itertuples(index=False):
            ir105 = 291 + 2.5 * seasonal - 0.003 * station.altitude + rng.normal(0, 0.4)
            ir112 = ir105 - 0.7 + rng.normal(0, 0.15)
            wv073 = 245 + 1.8 * seasonal + rng.normal(0, 0.5)
            ta = (
                30
                + 1.6 * seasonal
                - 0.005 * station.altitude
                - 0.25 * (station.latitude - 35)
                + 0.35 * (ir105 - 291)
                + rng.normal(0, 0.3)
            )
            hm = (
                67
                - 3.0 * seasonal
                + 0.45 * (wv073 - 245)
                + 0.8 * (station.longitude - 127.5)
                + rng.normal(0, 1.2)
            )
            records.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "STN_ID": station.STN_ID,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "altitude": station.altitude,
                    "IR105_center_mean": ir105,
                    "IR112_center_mean": ir112,
                    "WV073_center_mean": wv073,
                    "TA": ta,
                    "HM": np.clip(hm, 0, 100),
                }
            )
    full = add_calendar_features(pd.DataFrame(records))
    train = full[full["date"] < dates[-3].strftime("%Y-%m-%d")].copy()
    inference = full[full["date"] >= dates[-3].strftime("%Y-%m-%d")].drop(
        columns=["TA", "HM"]
    )
    ids = (
        pd.to_datetime(inference["date"]).dt.strftime("%Y%m%d")
        + "_"
        + inference["STN_ID"].astype(str)
    )
    sample = pd.DataFrame({"ID": ids, "TA": 0.0, "HM": 0.0})

    paths = {
        "stations": ROOT / "data/metadata/demo_station_list.csv",
        "train": ROOT / "data/processed/demo_train.csv",
        "features": ROOT / "data/processed/demo_inference_features.csv",
        "sample": ROOT / "data/metadata/demo_sample_submission.csv",
    }
    atomic_write_csv(stations, paths["stations"])
    atomic_write_csv(train, paths["train"])
    atomic_write_csv(inference, paths["features"])
    atomic_write_csv(sample, paths["sample"])
    print("데모 파일 생성 완료")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

