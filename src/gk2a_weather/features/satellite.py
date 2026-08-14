"""공식 관측소 좌표를 GK-2A KO 격자에 투영해 주변 특징을 추출한다.

위도·경도·고도는 운영진이 제공한 station_list.csv의 값만 사용한다.
투영·격자 상수는 GK-2A LE1B KO NetCDF 속성과 채널별 원본 크기에서 검증한
값이며, 외부 지형·해안선·관측소 메타데이터는 사용하지 않는다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


CHANNEL_RESOLUTION_KM = {
    "VI006": 0.5,
    "VI004": 1.0,
    "VI005": 1.0,
    "VI008": 1.0,
    "NR013": 2.0,
    "NR016": 2.0,
    "SW038": 2.0,
    "WV063": 2.0,
    "WV069": 2.0,
    "WV073": 2.0,
    "IR087": 2.0,
    "IR096": 2.0,
    "IR105": 2.0,
    "IR112": 2.0,
    "IR123": 2.0,
    "IR133": 2.0,
}

GRID_SPECS = {
    0.5: {
        "shape": (3600, 3600),
        "pixel_size_m": 500.0,
        "upper_left_easting_m": -899750.0,
        "upper_left_northing_m": 899750.0,
    },
    1.0: {
        "shape": (1800, 1800),
        "pixel_size_m": 1000.0,
        "upper_left_easting_m": -899500.0,
        "upper_left_northing_m": 899500.0,
    },
    2.0: {
        "shape": (900, 900),
        "pixel_size_m": 2000.0,
        "upper_left_easting_m": -899000.0,
        "upper_left_northing_m": 899000.0,
    },
}

PROJECTION = {
    "standard_parallel1": 30.0,
    "standard_parallel2": 60.0,
    "origin_latitude": 38.0,
    "central_meridian": 126.0,
    "false_easting": 0.0,
    "false_northing": 0.0,
}

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


class CoordinateMappingError(ValueError):
    """검증된 위성 픽셀 좌표가 없을 때 발생한다."""


def lcc_forward(longitude: float, latitude: float) -> tuple[float, float]:
    """공식 WGS84 좌표를 LE1B KO Lambert Conformal Conic으로 변환한다."""
    eccentricity = math.sqrt(WGS84_F * (2.0 - WGS84_F))

    def m(phi: float) -> float:
        return math.cos(phi) / math.sqrt(
            1.0 - eccentricity**2 * math.sin(phi) ** 2
        )

    def t(phi: float) -> float:
        ratio = (1.0 - eccentricity * math.sin(phi)) / (
            1.0 + eccentricity * math.sin(phi)
        )
        return math.tan(math.pi / 4.0 - phi / 2.0) / (
            ratio ** (eccentricity / 2.0)
        )

    phi1 = math.radians(PROJECTION["standard_parallel1"])
    phi2 = math.radians(PROJECTION["standard_parallel2"])
    phi0 = math.radians(PROJECTION["origin_latitude"])
    lambda0 = math.radians(PROJECTION["central_meridian"])
    phi = math.radians(latitude)
    lam = math.radians(longitude)
    n_value = (math.log(m(phi1)) - math.log(m(phi2))) / (
        math.log(t(phi1)) - math.log(t(phi2))
    )
    f_value = m(phi1) / (n_value * t(phi1) ** n_value)
    rho0 = WGS84_A * f_value * t(phi0) ** n_value
    rho = WGS84_A * f_value * t(phi) ** n_value
    theta = n_value * (lam - lambda0)
    x_coord = PROJECTION["false_easting"] + rho * math.sin(theta)
    y_coord = PROJECTION["false_northing"] + rho0 - rho * math.cos(theta)
    return x_coord, y_coord


def official_station_pixels(
    stations: pd.DataFrame,
    channel: str,
    array_shape: tuple[int, int],
) -> pd.DataFrame:
    """공식 LAT/LON을 해당 채널 원본 격자의 부동소수 row/col로 바꾼다."""
    required = {"STN_ID", "latitude", "longitude"}
    missing = required - set(stations.columns)
    if missing:
        raise CoordinateMappingError(
            "공식 station_list.csv 좌표가 필요합니다. 누락 열: "
            + ", ".join(sorted(missing))
        )
    channel = channel.upper()
    if channel not in CHANNEL_RESOLUTION_KM:
        raise CoordinateMappingError(f"지원하지 않는 GK-2A 채널: {channel}")
    resolution = CHANNEL_RESOLUTION_KM[channel]
    grid = GRID_SPECS[resolution]
    if tuple(array_shape) != tuple(grid["shape"]):
        raise CoordinateMappingError(
            f"{channel} 배열 크기 {array_shape}가 공식 {resolution:g}km KO 격자 "
            f"{grid['shape']}와 다릅니다. 파일이 LE1B/KO인지 확인하세요."
        )

    rows: list[float] = []
    cols: list[float] = []
    for station in stations.itertuples(index=False):
        x_coord, y_coord = lcc_forward(
            float(getattr(station, "longitude")),
            float(getattr(station, "latitude")),
        )
        rows.append(
            (float(grid["upper_left_northing_m"]) - y_coord)
            / float(grid["pixel_size_m"])
        )
        cols.append(
            (x_coord - float(grid["upper_left_easting_m"]))
            / float(grid["pixel_size_m"])
        )
    result = stations[["STN_ID"]].reset_index(drop=True).copy()
    result["row"] = rows
    result["col"] = cols
    height, width = array_shape
    inside = (
        result["row"].between(-0.5, height - 0.5)
        & result["col"].between(-0.5, width - 0.5)
    )
    if not inside.all():
        bad = result.loc[~inside, "STN_ID"].astype(int).tolist()
        raise CoordinateMappingError(f"{channel} KO 격자 밖의 공식 지점: {bad}")
    return result


def _pick_2d_variable(dataset: object, channel: str) -> str:
    preferred = [channel, channel.lower(), "image_pixel_values", "image"]
    for name in preferred:
        if name in dataset.data_vars and dataset[name].ndim >= 2:
            return name
    candidates = [
        (name, variable.size)
        for name, variable in dataset.data_vars.items()
        if variable.ndim == 2 and np.issubdtype(variable.dtype, np.number)
    ]
    if not candidates:
        raise ValueError(
            f"2차원 영상 변수를 찾지 못했습니다. 변수: {list(dataset.data_vars)}"
        )
    return max(candidates, key=lambda item: item[1])[0]


def load_satellite_array(path: str | Path, channel: str) -> np.ndarray:
    """npy 또는 NetCDF/HDF 계열 파일에서 보정된 2차원 배열을 읽는다."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        array = np.load(source)
    else:
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "위성 NetCDF/HDF 파일을 읽으려면 xarray와 h5netcdf가 필요합니다. "
                "pip install -r requirements.txt를 실행하세요."
            ) from exc
        with xr.open_dataset(source, mask_and_scale=True) as dataset:
            variable = _pick_2d_variable(dataset, channel)
            array = dataset[variable].squeeze().to_numpy()
    if array.ndim != 2:
        raise ValueError(f"위성 배열은 2차원이어야 합니다. 현재 shape={array.shape}")
    return np.asarray(array, dtype=float)


def resolve_pixel_columns(stations: pd.DataFrame, channel: str) -> tuple[str, str]:
    channel_pair = (f"{channel}_row", f"{channel}_col")
    if set(channel_pair).issubset(stations.columns):
        return channel_pair
    if {"row", "col"}.issubset(stations.columns):
        return "row", "col"
    raise CoordinateMappingError(
        "공식 station_list 좌표 또는 검증된 픽셀 좌표가 없습니다. "
        f"{channel}_row/{channel}_col 또는 row/col 컬럼이 필요합니다. "
        "운영진 제공 LAT/LON/ALT 파일을 사용하세요."
    )


def extract_channel_features(
    array: np.ndarray,
    stations: pd.DataFrame,
    *,
    channel: str,
    radii_pixels: tuple[int, ...] | None = None,
    radii_km: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    if radii_pixels is not None and radii_km is not None:
        raise ValueError("radii_pixels와 radii_km 중 하나만 지정하세요.")
    if radii_pixels is None and radii_km is None:
        radii_km = (0.0, 5.0, 15.0)
    if {"latitude", "longitude"}.issubset(stations.columns):
        pixels = official_station_pixels(stations, channel, array.shape)
        working_stations = stations.reset_index(drop=True).copy()
        working_stations["row"] = pixels["row"]
        working_stations["col"] = pixels["col"]
        row_col, col_col = "row", "col"
    else:
        working_stations = stations.reset_index(drop=True).copy()
        row_col, col_col = resolve_pixel_columns(working_stations, channel)

    radius_specs: list[tuple[int, str]] = []
    if radii_km is not None:
        resolution = CHANNEL_RESOLUTION_KM[channel.upper()]
        for radius_km in radii_km:
            if radius_km < 0:
                raise ValueError("패치 반경은 0 이상이어야 합니다.")
            radius_px = 0 if radius_km == 0 else int(math.ceil(radius_km / resolution))
            suffix = "center" if radius_km == 0 else f"r{radius_km:g}km"
            radius_specs.append((radius_px, suffix))
    else:
        for radius_px in radii_pixels or ():
            if radius_px < 0:
                raise ValueError("패치 반경은 0 이상이어야 합니다.")
            suffix = "center" if radius_px == 0 else f"r{radius_px}px"
            radius_specs.append((int(radius_px), suffix))
    if not radius_specs or radius_specs[0][0] != 0:
        raise ValueError("첫 패치 반경은 중심값을 뜻하는 0이어야 합니다.")

    result = stations[["STN_ID"]].reset_index(drop=True).copy()
    height, width = array.shape

    for station_index, station in working_stations.iterrows():
        row_raw = station[row_col]
        col_raw = station[col_col]
        invalid_coordinate = not np.isfinite(row_raw) or not np.isfinite(col_raw)
        row = int(round(row_raw)) if not invalid_coordinate else -1
        col = int(round(col_raw)) if not invalid_coordinate else -1
        inside = 0 <= row < height and 0 <= col < width

        for radius, suffix in radius_specs:
            if invalid_coordinate or not inside:
                values = np.array([], dtype=float)
            elif radius == 0:
                values = np.asarray([array[row, col]], dtype=float)
            else:
                row_min, row_max = max(0, row - radius), min(height, row + radius + 1)
                col_min, col_max = max(0, col - radius), min(width, col + radius + 1)
                values = array[row_min:row_max, col_min:col_max].ravel()

            finite = values[np.isfinite(values)]
            valid_ratio = float(len(finite) / len(values)) if len(values) else 0.0
            prefix = f"{channel}_{suffix}"
            result.loc[station_index, f"{prefix}_mean"] = (
                float(np.mean(finite)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_std"] = (
                float(np.std(finite)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_min"] = (
                float(np.min(finite)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_max"] = (
                float(np.max(finite)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_p10"] = (
                float(np.quantile(finite, 0.1)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_p50"] = (
                float(np.quantile(finite, 0.5)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_p90"] = (
                float(np.quantile(finite, 0.9)) if len(finite) else np.nan
            )
            result.loc[station_index, f"{prefix}_valid_ratio"] = valid_ratio

        center = result.loc[station_index, f"{channel}_center_mean"]
        _, largest_suffix = radius_specs[-1]
        surrounding = result.loc[station_index, f"{channel}_{largest_suffix}_mean"]
        result.loc[station_index, f"{channel}_center_minus_{largest_suffix}"] = (
            center - surrounding if np.isfinite(center) and np.isfinite(surrounding) else np.nan
        )

    result[f"{channel}_missing"] = result[f"{channel}_center_mean"].isna().astype(int)
    return result


def add_channel_differences(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    pairs = (
        ("IR105", "IR112"),
        ("IR112", "IR123"),
        ("WV063", "WV073"),
    )
    for left, right in pairs:
        left_column = f"{left}_center_mean"
        right_column = f"{right}_center_mean"
        if left_column in result and right_column in result:
            result[f"{left}_minus_{right}"] = result[left_column] - result[right_column]
    return result
