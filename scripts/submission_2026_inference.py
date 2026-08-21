#!/usr/bin/env python3
"""Inference-only implementation for the 2026 competition submission.

The module downloads only GK-2A ``LE1B/KO`` files, uses observations from
12:00 through the 14:00 KST target, and never reads ASOS at inference time.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import experiment_team_catboost_feature_ensemble as team  # noqa: E402
import submission_predictor as satellite  # noqa: E402
import train_dualbranch_ta_baseline as base  # noqa: E402


KEYS = ["Date", "STN_ID"]


def find_bundle(root: str | Path = "/kaggle/input") -> Path:
    candidates = []
    for path in Path(root).rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("bundle_version", "").startswith("2026."):
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one 2026 model bundle under {root}; found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _load_lstm(path: Path, device: str):
    import torch
    import torch.nn as nn

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    class ResidualLSTM(nn.Module):
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
                nn.Linear(
                    checkpoint["hidden_size"] + checkpoint["static_size"],
                    checkpoint["hidden_size"],
                ),
                nn.ReLU(),
                nn.Dropout(checkpoint["dropout"]),
                nn.Linear(checkpoint["hidden_size"], 1),
            )

        def forward(self, dynamic, static):
            _, (hidden, _) = self.lstm(dynamic)
            return self.head(torch.cat([hidden[-1], static], dim=1)).squeeze(1)

    model = ResidualLSTM().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalization_file = np.load(path.with_name(path.stem + "_normalization.npz"))
    normalization = {name: normalization_file[name] for name in normalization_file.files}
    return model, normalization


def load_bundle(bundle_dir: str | Path, device: str = "cpu") -> dict[str, object]:
    from catboost import CatBoostRegressor

    root = Path(bundle_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    models = {}
    for seed in manifest["seeds"]:
        seed_dir = root / f"seed_{seed}"
        seed_models = {"cat": {}, "lstm": {}}
        for target in ["TA", "HM"]:
            cat = CatBoostRegressor()
            cat.load_model(seed_dir / f"base_catboost_{target}.cbm")
            seed_models["cat"][target] = cat
            seed_models["lstm"][target] = _load_lstm(
                seed_dir / f"residual_lstm_{target}.pt", device
            )
        team_hm = CatBoostRegressor()
        team_hm.load_model(seed_dir / "team_spatial_catboost_HM.cbm")
        seed_models["team_hm"] = team_hm
        seed_models["ta_ridge"] = joblib.load(seed_dir / "ta_residual_ridge.joblib")
        seed_models["june_ta_ridge"] = joblib.load(seed_dir / "june_direct_ridge_TA.joblib")
        models[str(seed)] = seed_models
    return {"root": root, "manifest": manifest, "models": models, "device": device}


def _prepare_long(long_table: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    stations = satellite.validate_stations(stations)
    required = ["Date", "TimeKST", "STN_ID", *base.CHANNELS]
    missing = [column for column in required if column not in long_table]
    if missing:
        raise ValueError(f"satellite long table missing columns: {missing}")
    out = long_table.drop(columns=[c for c in ["LAT", "LON", "ALT"] if c in long_table]).merge(
        stations, on="STN_ID", how="left", validate="m:1"
    )
    out["Date"] = pd.to_numeric(out["Date"], errors="raise").round().astype("int64")
    out["TimeKST"] = pd.to_numeric(out["TimeKST"], errors="raise").round().astype("int64")
    out["STN_ID"] = pd.to_numeric(out["STN_ID"], errors="raise").round().astype("int64")
    if out.duplicated(["Date", "TimeKST", "STN_ID"]).any():
        raise ValueError("duplicate Date/TimeKST/STN_ID rows")
    for target in ["TA", "HM"]:
        out[target] = np.nan
    return out.sort_values(["Date", "STN_ID", "TimeKST"]).reset_index(drop=True)


def _sequence_arrays(long_table: pd.DataFrame, manifest: dict[str, object]):
    expected_times = np.asarray(manifest["times_kst"], dtype=int)
    dynamic, static, keys = [], [], []
    solar = base.add_solar_features(long_table)
    for (date_value, station), group in solar.groupby(KEYS, sort=True):
        group = group.sort_values("TimeKST")
        if len(group) != len(expected_times) or not np.array_equal(
            group["TimeKST"].to_numpy(int), expected_times
        ):
            raise ValueError(f"invalid 13-step grid: {date_value}/{station}")
        values = group[base.CHANNELS].to_numpy(np.float32)
        first = group.iloc[0]
        missing_count = int(np.isnan(values).sum())
        context = np.asarray(
            [
                first.LAT, first.LON, first.ALT, first.doy_sin, first.doy_cos,
                first.cos_solar_zenith, first.solar_azimuth,
                missing_count / float(values.size),
            ], dtype=np.float32,
        )
        dynamic.append(values)
        static.append(context)
        keys.append((int(date_value), int(station)))
    return np.stack(dynamic), np.stack(static), pd.DataFrame(keys, columns=KEYS)


def _predict_lstm(model, normalization, x_raw, static_raw, device: str) -> np.ndarray:
    import torch

    dynamic, static = base.transform_lstm_inputs(x_raw, static_raw, normalization)
    parts = []
    with torch.no_grad():
        for start in range(0, len(dynamic), 512):
            prediction = model(
                torch.from_numpy(dynamic[start:start + 512]).to(device),
                torch.from_numpy(static[start:start + 512]).to(device),
            )
            parts.append(prediction.cpu().numpy())
    scaled = np.concatenate(parts)
    return scaled * normalization["y_std"][0] + normalization["y_mean"][0]


def _tabular_frames(
    long_table: pd.DataFrame, stations: pd.DataFrame,
    manifest: dict[str, object], keys: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    last = long_table[long_table.TimeKST.eq(1400)].copy()
    last = last.sort_values(KEYS).reset_index(drop=True)
    if not last[KEYS].equals(keys):
        raise ValueError("14:00 tabular keys do not match sequence keys")
    solar = base.add_solar_features(last)
    tabular = base.engineer_tabular(solar, manifest["tabular_channel_stats"])
    team_frame = team.engineer_team_features(last, satellite.validate_stations(stations))
    team_frame = base.engineer_tabular(team_frame, manifest["tabular_channel_stats"])
    team_frame = keys.merge(team_frame, on=KEYS, how="left", validate="1:1")
    return tabular, team_frame


def _ridge_features(long_table: pd.DataFrame, manifest: dict[str, object]) -> pd.DataFrame:
    stats = manifest["sequence_channel_stats"]
    rows = []
    pairs = {name: (left, right) for left, right, name in base.STANDARDIZED_PAIRS}
    for (date_value, station), group in long_table.groupby(KEYS, sort=True):
        group = group.sort_values("TimeKST")
        row = {"Date": int(date_value), "STN_ID": int(station)}
        z = {}
        for channel in base.CHANNELS:
            values = group[channel].to_numpy(float)
            z[channel] = (values - stats["mean"][channel]) / stats["std"][channel]
            row[f"{channel}_last"] = float(values[-1])
            row[f"{channel}_mean_z"] = float(np.nanmean(z[channel]))
        for name, (left, right) in pairs.items():
            row[f"{name}_mean"] = float(np.nanmean(z[left] - z[right]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


def _predict_14h(prepared: pd.DataFrame, stations: pd.DataFrame, artifacts: dict[str, object]) -> pd.DataFrame:
    manifest = artifacts["manifest"]
    last = prepared[prepared.TimeKST.eq(1400)].sort_values(KEYS).reset_index(drop=True)
    keys = last[KEYS].copy()
    tabular, team_frame = _tabular_frames(last, stations, manifest, keys)
    recipe = manifest["june_recipe"]
    ta_parts, hm_parts = [], []
    for seed in manifest["seeds"]:
        models = artifacts["models"][str(seed)]
        cat_ta = np.asarray(
            models["cat"]["TA"].predict(tabular[manifest["tabular_features"]]), dtype=float
        )
        ridge_ta = np.asarray(
            models["june_ta_ridge"].predict(tabular[manifest["tabular_features"]]), dtype=float
        )
        ta_parts.append(
            recipe["ta_catboost_weight"] * cat_ta
            + recipe["ta_ridge_weight"] * ridge_ta
            + recipe["ta_offset"]
        )
        base_hm = np.asarray(
            models["cat"]["HM"].predict(tabular[manifest["tabular_features"]]), dtype=float
        )
        spatial_hm = np.asarray(
            models["team_hm"].predict(team_frame[manifest["team_hm_features"]]), dtype=float
        )
        hm_parts.append(np.clip(
            recipe["hm_base_catboost_weight"] * base_hm
            + recipe["hm_spatial_catboost_weight"] * spatial_hm,
            0.0, 100.0,
        ))
    result = keys.copy()
    result["TA"] = np.mean(np.stack(ta_parts), axis=0)
    result["HM"] = np.clip(np.mean(np.stack(hm_parts), axis=0), 0.0, 100.0)
    return result


def _predict_sequence(prepared: pd.DataFrame, stations: pd.DataFrame,
                      artifacts: dict[str, object]) -> pd.DataFrame:
    manifest = artifacts["manifest"]
    device = artifacts["device"]
    x_raw, static_raw, keys = _sequence_arrays(prepared, manifest)
    tabular, team_frame = _tabular_frames(prepared, stations, manifest, keys)
    ridge_frame = _ridge_features(prepared, manifest)
    if not ridge_frame[KEYS].equals(keys):
        raise ValueError("Ridge feature keys do not match sequence keys")

    ta_parts, hm_parts = [], []
    for seed in manifest["seeds"]:
        recipe = manifest["recipes"][str(seed)]
        models = artifacts["models"][str(seed)]
        base_predictions = {}
        for target in ["TA", "HM"]:
            cat_prediction = np.asarray(
                models["cat"][target].predict(tabular[manifest["tabular_features"]]),
                dtype=float,
            )
            lstm, normalization = models["lstm"][target]
            residual = _predict_lstm(lstm, normalization, x_raw, static_raw, device)
            prediction = cat_prediction + recipe[target]["residual_scale"] * residual
            if target == "TA":
                prediction += recipe[target]["pooled_offset"] + recipe[target]["latest_offset"]
            else:
                prediction = np.clip(prediction, 0.0, 100.0)
            base_predictions[target] = prediction

        ridge_residual = np.asarray(
            models["ta_ridge"].predict(ridge_frame[manifest["ta_ridge_features"]]), dtype=float
        )
        ta_parts.append(
            base_predictions["TA"] + recipe["TA"]["ridge_weight"] * ridge_residual
        )
        team_hm = np.asarray(
            models["team_hm"].predict(team_frame[manifest["team_hm_features"]]), dtype=float
        )
        weight = recipe["HM"]["team_weight"]
        hm_parts.append(np.clip((1.0 - weight) * base_predictions["HM"] + weight * team_hm, 0.0, 100.0))

    result = keys.copy()
    result["TA"] = np.mean(np.stack(ta_parts), axis=0)
    result["HM"] = np.clip(np.mean(np.stack(hm_parts), axis=0), 0.0, 100.0)
    if result[["TA", "HM"]].isna().any().any():
        raise RuntimeError("model produced missing predictions")
    return result


def predict_from_long_table(
    *, long_table: pd.DataFrame, pred_dates: Iterable[object], stations: pd.DataFrame,
    bundle_dir: str | Path, device: str = "cpu",
) -> pd.DataFrame:
    artifacts = load_bundle(bundle_dir, device=device)
    manifest = artifacts["manifest"]
    prepared = _prepare_long(long_table, stations)
    requested = sorted({int(pd.Timestamp(value).strftime("%Y%m%d")) for value in pred_dates})
    prepared = prepared[prepared.Date.isin(requested)].copy()
    support_start, support_end = manifest["sequence_supported_mmdd"]
    sequence_dates = [date for date in requested if support_start <= date % 10000 <= support_end]
    tabular_dates = [date for date in requested if date not in sequence_dates]
    parts = []
    if tabular_dates:
        parts.append(_predict_14h(
            prepared[prepared.Date.isin(tabular_dates)].copy(), stations, artifacts
        ))
    if sequence_dates:
        parts.append(_predict_sequence(
            prepared[prepared.Date.isin(sequence_dates)].copy(), stations, artifacts
        ))
    result = pd.concat(parts, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    expected = len(requested) * 96
    if len(result) != expected or result.duplicated(KEYS).any():
        raise ValueError(f"expected {expected} unique predictions, found {len(result)}")
    if result[["TA", "HM"]].isna().any().any():
        raise RuntimeError("model produced missing predictions")
    return result


def predict_from_api(
    *, api_key: str, pred_dates: Iterable[object], stations: pd.DataFrame,
    bundle_dir: str | Path, to_api_datetime: Callable[[object, object], str],
    cache_dir: str | Path = "/kaggle/working/gk2a_cache",
    failure_csv: str | Path = "/kaggle/working/gk2a_failures.csv",
    device: str = "cpu", max_missing_fraction: float = 0.05,
) -> pd.DataFrame:
    dates = list(pred_dates)
    manifest = json.loads((Path(bundle_dir) / "manifest.json").read_text(encoding="utf-8"))
    support_start, support_end = manifest["sequence_supported_mmdd"]
    sequence_dates = [
        value for value in dates
        if support_start <= int(pd.Timestamp(value).strftime("%m%d")) <= support_end
    ]
    tabular_dates = [value for value in dates if value not in sequence_dates]
    table_parts, failure_parts = [], []
    if sequence_dates:
        table, failed = satellite.collect_api_long_table(
            api_key=api_key, pred_dates=sequence_dates, stations=stations,
            to_api_datetime=to_api_datetime, cache_dir=cache_dir,
            max_missing_fraction=max_missing_fraction,
            request_interval_seconds=0.05,
        )
        table_parts.append(table)
        failure_parts.append(failed)
    if tabular_dates:
        table, failed = _collect_14h_api(
            api_key=api_key, pred_dates=tabular_dates, stations=stations,
            to_api_datetime=to_api_datetime, cache_dir=cache_dir,
            max_missing_fraction=max_missing_fraction,
        )
        table_parts.append(table)
        failure_parts.append(failed)
    long_table = pd.concat(table_parts, ignore_index=True)
    failures = pd.concat(failure_parts, ignore_index=True) if failure_parts else pd.DataFrame()
    Path(failure_csv).parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(failure_csv, index=False)
    print(
        f"GK-2A LE1B files expected={(len(sequence_dates) * 13 + len(tabular_dates)) * 16}, "
        f"failed={len(failures)}",
        flush=True,
    )
    return predict_from_long_table(
        long_table=long_table, pred_dates=dates, stations=stations,
        bundle_dir=bundle_dir, device=device,
    )


def _collect_14h_api(
    *, api_key: str, pred_dates: Iterable[object], stations: pd.DataFrame,
    to_api_datetime: Callable[[object, object], str], cache_dir: str | Path,
    max_missing_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import time
    import requests

    official = satellite.validate_stations(stations)
    records, errors = [], []
    expected = 0
    with requests.Session() as session:
        for target_dt in pred_dates:
            target = pd.Timestamp(target_dt)
            date_value = int(target.strftime("%Y%m%d"))
            api_date = to_api_datetime(target_dt, target_dt)
            frame = official.copy()
            frame.insert(0, "TimestampUTC", api_date)
            frame.insert(0, "TimeKST", 1400)
            frame.insert(0, "Date", date_value)
            for channel in base.CHANNELS:
                expected += 1
                path = Path(cache_dir) / str(date_value) / f"{channel}_{api_date}.nc"
                try:
                    satellite.download_nc(
                        session=session, api_key=api_key, channel=channel,
                        api_date=api_date, destination=path,
                        timeout_seconds=90.0, max_retries=2,
                    )
                    values = satellite.load_nc_station_values(path, channel, official)
                except Exception as exc:
                    path.unlink(missing_ok=True)
                    values = np.full(len(official), np.nan, dtype=float)
                    errors.append({
                        "Date": date_value, "TimeKST": 1400,
                        "TimestampUTC": api_date, "Channel": channel,
                        "Error": f"{type(exc).__name__}: {exc}",
                    })
                    print(
                        f"[SATELLITE FAILURE] {date_value} 1400 {channel}: "
                        f"{type(exc).__name__}: {exc}", flush=True,
                    )
                    if "HTTP 429" in str(exc):
                        raise RuntimeError(
                            "KMA API quota reached. Stop and rerun after the quota resets; "
                            "the cache will reuse completed files."
                        ) from exc
                frame[channel] = values
                time.sleep(0.05)
            records.append(frame)
    failures = pd.DataFrame(
        errors, columns=["Date", "TimeKST", "TimestampUTC", "Channel", "Error"]
    )
    if len(failures) / max(1, expected) > max_missing_fraction:
        raise RuntimeError(
            f"satellite failure ratio {len(failures) / max(1, expected):.1%} exceeds "
            f"the allowed {max_missing_fraction:.1%}"
        )
    return pd.concat(records, ignore_index=True), failures
