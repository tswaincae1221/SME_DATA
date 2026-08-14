"""운영진이 제공한 공식 station_list.csv를 검증·정규화한다.

규정 개정 후 허용되는 관측소 입력은 STN_ID, 위도, 경도, 고도뿐이다.
외부 관측소명·지형·해안선 거리 같은 열이 실수로 모델 입력에 섞이지 않도록
공식 네 열만 반환한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STATION_ALIASES = {
    "STN": "STN_ID",
    "stn": "STN_ID",
    "stn_id": "STN_ID",
    "LAT": "latitude",
    "lat": "latitude",
    "Latitude": "latitude",
    "LON": "longitude",
    "lon": "longitude",
    "Longitude": "longitude",
    "HT": "altitude",
    "ALT": "altitude",
    "alt": "altitude",
    "height": "altitude",
}

OFFICIAL_COLUMNS = ("STN_ID", "latitude", "longitude", "altitude")


def load_station_list(path: str | Path) -> pd.DataFrame:
    station_path = Path(path)
    if not station_path.exists():
        raise FileNotFoundError(
            f"station_list.csv가 없습니다: {station_path}\n"
            "캐글 제공 파일을 data/metadata/station_list.csv에 넣어주세요."
        )

    frame = pd.read_csv(station_path).rename(columns=STATION_ALIASES)
    if "STN_ID" not in frame.columns:
        raise ValueError(
            f"관측소 파일에 STN_ID 컬럼이 필요합니다. 현재: {frame.columns.tolist()}"
        )
    frame["STN_ID"] = pd.to_numeric(frame["STN_ID"], errors="raise").astype(int)
    missing_columns = sorted(set(OFFICIAL_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "개정 규정의 공식 station_list.csv에는 STN_ID, LAT, LON, ALT가 "
            f"모두 필요합니다. 누락: {missing_columns}"
        )
    for column in ("latitude", "longitude", "altitude"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if frame[list(OFFICIAL_COLUMNS)].isna().any().any():
        raise ValueError("공식 station_list.csv의 네 필수 열에 결측값이 있습니다.")
    if frame["STN_ID"].duplicated().any():
        duplicated = frame.loc[frame["STN_ID"].duplicated(), "STN_ID"].tolist()
        raise ValueError(f"중복 관측소가 있습니다: {duplicated[:10]}")
    if not frame["latitude"].between(30.0, 40.5).all():
        raise ValueError("LAT가 한반도 KO 영역 검수 범위(30~40.5)를 벗어났습니다.")
    if not frame["longitude"].between(120.0, 133.5).all():
        raise ValueError("LON이 한반도 KO 영역 검수 범위(120~133.5)를 벗어났습니다.")
    if not frame["altitude"].between(-100.0, 3000.0).all():
        raise ValueError("ALT가 관측소 고도 검수 범위(-100~3000m)를 벗어났습니다.")
    return frame[list(OFFICIAL_COLUMNS)].reset_index(drop=True)
