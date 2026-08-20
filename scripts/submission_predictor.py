#!/usr/bin/env python3
"""Inference-only GK-2A LSTM + CatBoost submission predictor.

The API path is restricted to GK-2A ``/LE1B/`` and area ``KO``.  The module
never reads ASOS at inference.  It can also run against a historical short-term
long table for an offline, label-free rehearsal of the exact model loading,
feature ordering, blending, and submission formatting code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


CHANNELS = [
    "VI004", "VI005", "VI006", "VI008", "NR013", "NR016", "SW038",
    "WV063", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112",
    "IR123", "IR133",
]
EXPECTED_TIMES = np.asarray(
    [1200, 1210, 1220, 1230, 1240, 1250, 1300, 1310, 1320, 1330, 1340, 1350, 1400],
    dtype=np.int64,
)
STATIC_FEATURES = ["LAT", "LON", "ALT", "doy_sin", "doy_cos"]
TARGETS = ["TA", "HM"]
GK2A_URL = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{channel}/KO/data"

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
    0.5: (3600, 3600, 500.0, -899750.0, 899750.0),
    1.0: (1800, 1800, 1000.0, -899500.0, 899500.0),
    2.0: (900, 900, 2000.0, -899000.0, 899000.0),
}
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = pd.to_numeric(out["Date"], errors="raise").round().astype("int64")
    out["STN_ID"] = pd.to_numeric(out["STN_ID"], errors="raise").astype("int64")
    date = pd.to_datetime(out["Date"].astype(str), format="%Y%m%d", errors="raise")
    out["month"] = date.dt.month.astype("int8")
    out["day"] = date.dt.day.astype("int8")
    out["dayofyear"] = date.dt.dayofyear.astype("int16")
    angle = 2.0 * np.pi * out["dayofyear"].to_numpy(dtype=float) / 365.25
    out["doy_sin"] = np.sin(angle)
    out["doy_cos"] = np.cos(angle)
    return out


def validate_stations(stations: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "STN": "STN_ID", "stn": "STN_ID", "stn_id": "STN_ID",
        "latitude": "LAT", "lat": "LAT", "longitude": "LON", "lon": "LON",
        "altitude": "ALT", "alt": "ALT", "HT": "ALT",
    }
    frame = stations.rename(columns=aliases)
    required = ["STN_ID", "LAT", "LON", "ALT"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"official station_list.csv missing columns: {missing}")
    result = frame[required].copy()
    result["STN_ID"] = pd.to_numeric(result["STN_ID"], errors="raise").astype(int)
    for column in ("LAT", "LON", "ALT"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    result = result.sort_values("STN_ID").reset_index(drop=True)
    if len(result) != 96 or result["STN_ID"].nunique() != 96:
        raise ValueError("official station_list.csv must contain 96 unique stations")
    if result.isna().any().any():
        raise ValueError("official station_list.csv contains missing coordinates")
    return result


def lcc_forward(longitude: float, latitude: float) -> tuple[float, float]:
    eccentricity = math.sqrt(WGS84_F * (2.0 - WGS84_F))

    def m(phi: float) -> float:
        return math.cos(phi) / math.sqrt(1.0 - eccentricity**2 * math.sin(phi) ** 2)

    def t(phi: float) -> float:
        ratio = (1.0 - eccentricity * math.sin(phi)) / (1.0 + eccentricity * math.sin(phi))
        return math.tan(math.pi / 4.0 - phi / 2.0) / ratio ** (eccentricity / 2.0)

    phi1, phi2, phi0 = map(math.radians, (30.0, 60.0, 38.0))
    lambda0 = math.radians(126.0)
    phi, lam = math.radians(latitude), math.radians(longitude)
    n_value = (math.log(m(phi1)) - math.log(m(phi2))) / (math.log(t(phi1)) - math.log(t(phi2)))
    f_value = m(phi1) / (n_value * t(phi1) ** n_value)
    rho0 = WGS84_A * f_value * t(phi0) ** n_value
    rho = WGS84_A * f_value * t(phi) ** n_value
    theta = n_value * (lam - lambda0)
    return rho * math.sin(theta), rho0 - rho * math.cos(theta)


def station_pixels(stations: pd.DataFrame, channel: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    resolution = CHANNEL_RESOLUTION_KM[channel]
    height, width, pixel_size, upper_left_easting, upper_left_northing = GRID_SPECS[resolution]
    if tuple(shape) != (height, width):
        raise ValueError(f"{channel} shape={shape}, expected={(height, width)} for LE1B/KO")
    rows: list[int] = []
    cols: list[int] = []
    for station in stations.itertuples(index=False):
        x_coord, y_coord = lcc_forward(float(station.LON), float(station.LAT))
        rows.append(int(round((upper_left_northing - y_coord) / pixel_size)))
        cols.append(int(round((x_coord - upper_left_easting) / pixel_size)))
    row_array = np.asarray(rows, dtype=int)
    col_array = np.asarray(cols, dtype=int)
    inside = (
        (row_array >= 0) & (row_array < height) &
        (col_array >= 0) & (col_array < width)
    )
    if not inside.all():
        raise ValueError(f"{channel}: official stations outside the KO grid")
    return row_array, col_array


def load_nc_station_values(path: Path, channel: str, stations: pd.DataFrame) -> np.ndarray:
    import xarray as xr

    with xr.open_dataset(path, mask_and_scale=True, decode_times=False) as dataset:
        preferred = (channel, channel.lower(), "image_pixel_values", "image")
        variable_name = next(
            (name for name in preferred if name in dataset.data_vars and dataset[name].ndim >= 2),
            None,
        )
        if variable_name is None:
            candidates = [
                (name, variable.size)
                for name, variable in dataset.data_vars.items()
                if variable.ndim == 2 and np.issubdtype(variable.dtype, np.number)
            ]
            if not candidates:
                raise ValueError(f"no numeric 2-D variable in {path.name}")
            variable_name = max(candidates, key=lambda item: item[1])[0]
        array = np.asarray(dataset[variable_name].squeeze().to_numpy(), dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2-D image, got {array.shape}")
    rows, cols = station_pixels(stations, channel, array.shape)
    return array[rows, cols].astype(float)


def observation_times(target_dt: object) -> list[pd.Timestamp]:
    target = pd.Timestamp(target_dt)
    return [target - pd.Timedelta(minutes=120 - 10 * index) for index in range(13)]


def _looks_like_error_document(content: bytes, content_type: str) -> bool:
    prefix = content[:512].lstrip().lower()
    return (
        "text/" in content_type.lower()
        or "json" in content_type.lower()
        or prefix.startswith((b"<html", b"<!doctype", b"<?xml", b"{"))
    )


def download_nc(
    *,
    session: object,
    api_key: str,
    channel: str,
    api_date: str,
    destination: Path,
    timeout_seconds: float,
    max_retries: int,
) -> Path:
    if destination.exists() and destination.stat().st_size >= 10_000:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = GK2A_URL.format(channel=channel)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                url,
                params={"date": api_date, "authKey": api_key},
                timeout=timeout_seconds,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 5.0 * (2**attempt))
                raise RuntimeError(f"HTTP 429; retry_after={wait_seconds:g}s")
            response.raise_for_status()
            content = response.content
            if len(content) < 10_000 or _looks_like_error_document(
                content, response.headers.get("content-type", "")
            ):
                raise RuntimeError(
                    f"non-NetCDF response: content_type={response.headers.get('content-type', '')}, "
                    f"bytes={len(content)}"
                )
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(content)
            os.replace(temporary, destination)
            return destination
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            if "retry_after=" in str(exc):
                try:
                    wait_seconds = float(str(exc).split("retry_after=")[1].split("s")[0])
                except ValueError:
                    wait_seconds = min(60.0, 5.0 * (2**attempt))
            else:
                wait_seconds = min(30.0, 2.0 * (2**attempt))
            time.sleep(wait_seconds)
    raise RuntimeError(f"{channel} {api_date} download failed: {type(last_error).__name__}: {last_error}")


def collect_api_long_table(
    *,
    api_key: str,
    pred_dates: Iterable[object],
    stations: pd.DataFrame,
    to_api_datetime: Callable[[object, object], str],
    cache_dir: str | Path,
    timeout_seconds: float = 90.0,
    max_retries: int = 2,
    request_interval_seconds: float = 0.2,
    max_missing_fraction: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import requests

    stations = validate_stations(stations)
    cache_root = Path(cache_dir)
    records: list[pd.DataFrame] = []
    failure_rows: list[dict[str, object]] = []
    expected_files = 0
    consecutive_rate_limit_failures = 0
    with requests.Session() as session:
        for target_dt in pred_dates:
            date_value = int(pd.Timestamp(target_dt).strftime("%Y%m%d"))
            for obs_dt in observation_times(target_dt):
                time_kst = int(pd.Timestamp(obs_dt).strftime("%H%M"))
                api_date = to_api_datetime(obs_dt, target_dt)
                frame = stations.copy()
                frame.insert(0, "TimestampUTC", api_date)
                frame.insert(0, "TimeKST", time_kst)
                frame.insert(0, "Date", date_value)
                for channel in CHANNELS:
                    expected_files += 1
                    path = cache_root / str(date_value) / f"{channel}_{api_date}.nc"
                    try:
                        download_nc(
                            session=session,
                            api_key=api_key,
                            channel=channel,
                            api_date=api_date,
                            destination=path,
                            timeout_seconds=timeout_seconds,
                            max_retries=max_retries,
                        )
                        values = load_nc_station_values(path, channel, stations)
                        consecutive_rate_limit_failures = 0
                    except Exception as exc:
                        # A response can be large enough to pass the byte-size
                        # check yet still be a truncated HDF/NetCDF file. Remove
                        # only this cache entry so a rerun downloads it again.
                        if path.exists():
                            path.unlink(missing_ok=True)
                        values = np.full(len(stations), np.nan, dtype=float)
                        failure_rows.append(
                            {
                                "Date": date_value,
                                "TimeKST": time_kst,
                                "TimestampUTC": api_date,
                                "Channel": channel,
                                "Error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        print(f"[SATELLITE FAILURE] {date_value} {time_kst:04d} {channel}: {type(exc).__name__}: {exc}")
                        if "HTTP 429" in str(exc):
                            consecutive_rate_limit_failures += 1
                        else:
                            consecutive_rate_limit_failures = 0
                        if consecutive_rate_limit_failures >= 2:
                            raise RuntimeError(
                                "KMA API returned HTTP 429 for two consecutive files. "
                                "Stop this run and retry after the API quota resets."
                            ) from exc
                    frame[channel] = values
                    if request_interval_seconds > 0:
                        time.sleep(request_interval_seconds)
                records.append(frame)
    failures = pd.DataFrame(
        failure_rows,
        columns=["Date", "TimeKST", "TimestampUTC", "Channel", "Error"],
    )
    missing_fraction = len(failures) / max(1, expected_files)
    if missing_fraction > max_missing_fraction:
        raise RuntimeError(
            f"satellite failure ratio {missing_fraction:.1%} exceeds "
            f"the allowed {max_missing_fraction:.1%} ({len(failures)}/{expected_files} files)"
        )
    long_table = pd.concat(records, ignore_index=True).sort_values(
        ["Date", "STN_ID", "TimeKST"]
    ).reset_index(drop=True)
    return long_table, failures


def find_artifact_dir(root: str | Path) -> Path:
    required = {
        "feature_spec.json", "ensemble_component_weights.csv",
        "lstm_model.pt", "lstm_normalization.npz",
        "catboost_TA.cbm", "catboost_HM.cbm",
    }
    candidates: list[Path] = []
    for spec_path in Path(root).rglob("feature_spec.json"):
        parent = spec_path.parent
        if all((parent / name).is_file() for name in required):
            candidates.append(parent)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one model artifact folder under {root}; found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def load_artifacts(artifact_dir: str | Path, device: str = "cpu") -> dict[str, object]:
    import torch
    import torch.nn as nn
    from catboost import CatBoostRegressor

    root = Path(artifact_dir)
    spec = json.loads((root / "feature_spec.json").read_text(encoding="utf-8"))
    normalization_file = np.load(root / "lstm_normalization.npz")
    normalization = {name: normalization_file[name] for name in normalization_file.files}
    checkpoint = torch.load(root / "lstm_model.pt", map_location=device, weights_only=False)

    class ShortTermLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = checkpoint["dropout"] if checkpoint["lstm_layers"] > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["lstm_layers"],
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(checkpoint["hidden_size"] + checkpoint["static_size"], checkpoint["hidden_size"]),
                nn.ReLU(),
                nn.Dropout(checkpoint["dropout"]),
                nn.Linear(checkpoint["hidden_size"], len(TARGETS)),
            )

        def forward(self, dynamic, static):
            _, (hidden, _) = self.lstm(dynamic)
            return self.head(torch.cat([hidden[-1], static], dim=1))

    lstm = ShortTermLSTM().to(device)
    lstm.load_state_dict(checkpoint["state_dict"])
    lstm.eval()
    catboost_models: dict[str, object] = {}
    for target in TARGETS:
        model = CatBoostRegressor()
        model.load_model(root / f"catboost_{target}.cbm")
        catboost_models[target] = model
    weight_frame = pd.read_csv(root / "ensemble_component_weights.csv").set_index("target")
    weights = {target: float(weight_frame.loc[target, "lstm_weight"]) for target in TARGETS}
    return {
        "lstm": lstm,
        "normalization": normalization,
        "catboost": catboost_models,
        "weights": weights,
        "spec": spec,
        "device": device,
    }


def build_arrays(
    long_table: pd.DataFrame,
    pred_dates: Iterable[object],
    stations: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    stations = validate_stations(stations)
    frame = add_date_features(long_table)
    frame["TimeKST"] = pd.to_numeric(frame["TimeKST"], errors="raise").astype(int)
    date_values = [int(pd.Timestamp(value).strftime("%Y%m%d")) for value in pred_dates]
    station_ids = stations["STN_ID"].to_numpy(dtype=int)
    dynamic_parts: list[np.ndarray] = []
    static_parts: list[np.ndarray] = []
    catboost_parts: list[pd.DataFrame] = []
    key_parts: list[pd.DataFrame] = []
    for date_value in date_values:
        date_frame = frame[frame["Date"].eq(date_value)]
        expected_rows = len(stations) * len(EXPECTED_TIMES)
        if len(date_frame) != expected_rows:
            raise ValueError(
                f"{date_value}: expected {expected_rows} station-time rows, found {len(date_frame)}"
            )
        date_dynamic = np.empty((len(stations), len(EXPECTED_TIMES), len(CHANNELS)), dtype=np.float32)
        for time_index, time_kst in enumerate(EXPECTED_TIMES):
            slot = date_frame[date_frame["TimeKST"].eq(int(time_kst))].sort_values("STN_ID")
            if not np.array_equal(slot["STN_ID"].to_numpy(dtype=int), station_ids):
                raise ValueError(f"{date_value} {time_kst:04d}: station grid mismatch")
            date_dynamic[:, time_index, :] = slot[CHANNELS].to_numpy(dtype=np.float32)
        day_rows = date_frame[date_frame["TimeKST"].eq(1400)].sort_values("STN_ID").reset_index(drop=True)
        dynamic_parts.append(date_dynamic)
        static_parts.append(day_rows[STATIC_FEATURES].to_numpy(dtype=np.float32))
        catboost_parts.append(day_rows)
        keys = day_rows[["Date", "STN_ID"]].copy()
        key_parts.append(keys)
    return (
        np.concatenate(dynamic_parts, axis=0),
        np.concatenate(static_parts, axis=0),
        pd.concat(catboost_parts, ignore_index=True),
        pd.concat(key_parts, ignore_index=True),
    )


def predict_from_long_table(
    *,
    long_table: pd.DataFrame,
    pred_dates: Iterable[object],
    stations: pd.DataFrame,
    artifact_dir: str | Path,
    device: str = "cpu",
) -> pd.DataFrame:
    import torch

    artifacts = load_artifacts(artifact_dir, device=device)
    dynamic_raw, static_raw, catboost_frame, keys = build_arrays(
        long_table, pred_dates, stations
    )
    normalization = artifacts["normalization"]
    missing = np.isnan(dynamic_raw).astype(np.float32)
    imputed = np.where(
        np.isnan(dynamic_raw), normalization["x_mean"][None, None, :], dynamic_raw
    )
    scaled = (imputed - normalization["x_mean"][None, None, :]) / normalization["x_std"][None, None, :]
    dynamic = np.concatenate([scaled.astype(np.float32), missing], axis=2)
    static_imputed = np.where(
        np.isnan(static_raw), normalization["static_mean"][None, :], static_raw
    )
    static = (
        (static_imputed - normalization["static_mean"][None, :])
        / normalization["static_std"][None, :]
    ).astype(np.float32)
    with torch.no_grad():
        scaled_lstm = artifacts["lstm"](
            torch.from_numpy(dynamic).to(device), torch.from_numpy(static).to(device)
        ).cpu().numpy()
    lstm_prediction = (
        scaled_lstm * normalization["y_std"][None, :]
        + normalization["y_mean"][None, :]
    )
    lstm_prediction[:, 1] = np.clip(lstm_prediction[:, 1], 0.0, 100.0)

    catboost_prediction = np.empty((len(keys), len(TARGETS)), dtype=float)
    for target_index, target in enumerate(TARGETS):
        features_by_target = artifacts["spec"].get("catboost_features_by_target")
        if features_by_target is None:
            catboost_features = artifacts["spec"]["catboost_features"]
        else:
            catboost_features = features_by_target[target]
        catboost_prediction[:, target_index] = artifacts["catboost"][target].predict(
            catboost_frame[catboost_features]
        )
    catboost_prediction[:, 1] = np.clip(catboost_prediction[:, 1], 0.0, 100.0)

    ensemble = np.empty_like(catboost_prediction)
    for target_index, target in enumerate(TARGETS):
        weight = artifacts["weights"][target]
        ensemble[:, target_index] = (
            weight * lstm_prediction[:, target_index]
            + (1.0 - weight) * catboost_prediction[:, target_index]
        )
    result = keys.copy()
    result["TA"] = ensemble[:, 0]
    result["HM"] = np.clip(ensemble[:, 1], 0.0, 100.0)
    if result[["TA", "HM"]].isna().any().any():
        raise RuntimeError("model produced missing TA/HM predictions")
    return result


def predict_from_api(
    *,
    api_key: str,
    pred_dates: Iterable[object],
    stations: pd.DataFrame,
    artifact_dir: str | Path,
    to_api_datetime: Callable[[object, object], str],
    cache_dir: str | Path,
    failure_csv: str | Path,
    device: str = "cpu",
    max_missing_fraction: float = 0.05,
) -> pd.DataFrame:
    dates = list(pred_dates)
    long_table, failures = collect_api_long_table(
        api_key=api_key,
        pred_dates=dates,
        stations=stations,
        to_api_datetime=to_api_datetime,
        cache_dir=cache_dir,
        max_missing_fraction=max_missing_fraction,
    )
    Path(failure_csv).parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(failure_csv, index=False)
    print(
        f"GK-2A files: expected={len(dates) * len(EXPECTED_TIMES) * len(CHANNELS)}, "
        f"failed={len(failures)}"
    )
    return predict_from_long_table(
        long_table=long_table,
        pred_dates=dates,
        stations=stations,
        artifact_dir=artifact_dir,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline submission inference rehearsal")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--station-list", required=True)
    parser.add_argument("--long-csv", required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    stations = pd.read_csv(args.station_list)
    long_table = pd.read_csv(args.long_csv)
    dates = [pd.Timestamp(value) for value in args.dates]
    pred = predict_from_long_table(
        long_table=long_table,
        pred_dates=dates,
        stations=stations,
        artifact_dir=args.artifact_dir,
        device=args.device,
    )
    submission = pd.DataFrame(
        {
            "ID": pred["Date"].astype(int).astype(str) + "_" + pred["STN_ID"].astype(int).astype(str),
            "TA": np.clip(pred["TA"], -50.0, 50.0).round(2),
            "HM": np.clip(pred["HM"], 0.0, 100.0).round(2),
        }
    ).sort_values("ID").reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(
        f"saved {output}: rows={len(submission)}, duplicate_ID={submission['ID'].duplicated().sum()}, "
        f"TA={submission.TA.min():.2f}..{submission.TA.max():.2f}, "
        f"HM={submission.HM.min():.2f}..{submission.HM.max():.2f}"
    )


if __name__ == "__main__":
    main()
