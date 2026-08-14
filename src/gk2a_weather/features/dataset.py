"""ASOS, 관측소 메타데이터, 위성 특징을 학습 테이블로 결합한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gk2a_weather.features.satellite import add_channel_differences
from gk2a_weather.features.static import add_calendar_features


def read_csv_directory(path: str | Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV 파일이 없습니다: {path}")
    return pd.concat((pd.read_csv(file) for file in files), ignore_index=True)


def build_training_table(
    labels: pd.DataFrame,
    stations: pd.DataFrame,
    satellite_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_labels = {"date", "STN_ID", "TA", "HM"}
    missing = required_labels - set(labels.columns)
    if missing:
        raise ValueError(f"ASOS 라벨 컬럼이 부족합니다: {sorted(missing)}")

    work = labels.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.strftime("%Y-%m-%d")
    work["STN_ID"] = pd.to_numeric(work["STN_ID"], errors="raise").astype(int)
    if work.duplicated(["date", "STN_ID"]).any():
        raise ValueError("ASOS 라벨에 중복된 date×STN_ID 행이 있습니다.")

    station_meta = stations.copy()
    station_meta["STN_ID"] = station_meta["STN_ID"].astype(int)
    work = work.merge(station_meta, on="STN_ID", how="left", validate="many_to_one")

    if satellite_features is not None and not satellite_features.empty:
        satellite = satellite_features.copy()
        satellite["date"] = pd.to_datetime(satellite["date"]).dt.strftime("%Y-%m-%d")
        satellite["STN_ID"] = satellite["STN_ID"].astype(int)
        if satellite.duplicated(["date", "STN_ID"]).any():
            raise ValueError("위성 특징에 중복된 date×STN_ID 행이 있습니다.")
        work = work.merge(
            satellite,
            on=["date", "STN_ID"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_sat"),
        )

    work = add_calendar_features(work)
    work = add_channel_differences(work)
    return work.sort_values(["date", "STN_ID"]).reset_index(drop=True)

