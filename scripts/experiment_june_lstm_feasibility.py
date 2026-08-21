#!/usr/bin/env python3
"""Test whether June 12:00--14:00 sequences add signal to the 14:00 model.

The experiment has one purpose: decide whether an LSTM is worth adding to the
current rules-safe June submission model.  It compares four candidates:

1. current 14:00 CatBoost/Ridge baseline,
2. direct LSTM,
3. baseline + direct-LSTM convex blend,
4. baseline + residual LSTM correction.

Blend weights and residual scales are selected on rolling 2022--2024 OOF only.
The year 2025 is printed once as a locked report and is never used to tune a
weight, feature, epoch count, or threshold.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import confirm_team_catboost_feature_ensemble_multiseed as frozen_team  # noqa: E402
import experiment_team_catboost_feature_ensemble as team  # noqa: E402
import train_dualbranch_ta_baseline as base  # noqa: E402


CHANNELS = ["IR087", "IR096", "IR105", "IR112", "IR123", "SW038", "WV069", "WV073"]
TIMES = np.asarray([1200, 1230, 1300, 1330, 1400], dtype=np.int64)
STATIC_FEATURES = [
    "LAT", "LON", "ALT", "doy_sin", "doy_cos", "cos_solar_zenith",
    "sequence_missing_fraction",
]
KEYS = ["Date", "STN_ID"]
TARGETS = ["TA", "HM"]
JUNE_RECIPE = {
    "ta_catboost_weight": 0.78,
    "ta_ridge_weight": 0.22,
    "ta_offset": 0.5421463057934491,
    "hm_base_catboost_weight": 0.36,
    "hm_spatial_catboost_weight": 0.64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--shortterm-csv", required=True)
    parser.add_argument("--station-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--report-year", type=int, default=2025)
    parser.add_argument("--sequence-years", nargs="+", type=int, default=list(range(2020, 2026)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cat-iterations", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-sequence-missing-fraction", type=float, default=0.20)
    parser.add_argument("--weight-step", type=float, default=0.02)
    parser.add_argument("--residual-scale-max", type=float, default=2.0)
    parser.add_argument("--min-improvement-ta", type=float, default=0.03)
    parser.add_argument("--min-improvement-hm", type=float, default=0.10)
    parser.add_argument("--force-baseline", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="One LSTM seed, 25 epochs and 250 CatBoost iterations for a pipeline smoke run.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(prediction)
    if not mask.any():
        return math.nan
    return float(np.sqrt(np.mean((actual[mask] - prediction[mask]) ** 2)))


def clip_target(values: np.ndarray, target: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    return np.clip(result, 0.0, 100.0) if target == "HM" else result


def load_sequences(
    path: str | Path,
    stations: pd.DataFrame,
    maximum_missing: float,
) -> dict[str, object]:
    frame = pd.read_csv(path)
    required = ["Date", "TimeKST", "STN_ID", *CHANNELS, "TA", "HM"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"short-term table missing columns: {missing}")
    frame["Date"] = pd.to_numeric(frame.Date, errors="raise").round().astype("int64")
    frame["TimeKST"] = pd.to_numeric(frame.TimeKST, errors="raise").round().astype("int64")
    frame["STN_ID"] = pd.to_numeric(frame.STN_ID, errors="raise").round().astype("int64")
    frame = frame.drop(columns=[c for c in ["LAT", "LON", "ALT"] if c in frame]).merge(
        stations, on="STN_ID", how="left", validate="m:1"
    )
    if frame[["LAT", "LON", "ALT"]].isna().any().any():
        raise ValueError("short-term table contains stations absent from official station_list.csv")

    dynamic_parts: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    key_rows: list[tuple[int, int, int]] = []
    labels = {target: [] for target in TARGETS}
    quality_rows = []
    for (date_value, station), group in frame.groupby(KEYS, sort=True):
        group = group.sort_values("TimeKST")
        got_times = group.TimeKST.to_numpy(dtype=int)
        if len(group) != len(TIMES) or not np.array_equal(got_times, TIMES):
            continue
        values = group[CHANNELS].to_numpy(dtype=np.float32)
        missing_fraction = float(np.isnan(values).mean())
        first = group.iloc[0]
        solar = base.add_solar_features(
            pd.DataFrame([{
                "Date": int(date_value), "STN_ID": int(station),
                "LAT": float(first.LAT), "LON": float(first.LON), "ALT": float(first.ALT),
            }])
        ).iloc[0]
        dynamic_parts.append(values)
        static_rows.append(np.asarray([
            solar.LAT, solar.LON, solar.ALT, solar.doy_sin, solar.doy_cos,
            solar.cos_solar_zenith, missing_fraction,
        ], dtype=np.float32))
        year = int(date_value) // 10000
        key_rows.append((int(date_value), int(station), year))
        final = group.iloc[-1]
        for target in TARGETS:
            labels[target].append(float(final[target]) if pd.notna(final[target]) else np.nan)
        quality_rows.append({
            "Date": int(date_value), "STN_ID": int(station), "year": year,
            "missing_fraction": missing_fraction,
            "usable": bool(missing_fraction <= maximum_missing),
        })
    if not dynamic_parts:
        raise ValueError("no complete five-time sequences were found")
    return {
        "x": np.stack(dynamic_parts),
        "static": np.stack(static_rows),
        "keys": pd.DataFrame(key_rows, columns=[*KEYS, "year"]),
        "labels": {name: np.asarray(values, dtype=np.float32) for name, values in labels.items()},
        "quality": pd.DataFrame(quality_rows),
    }


def fit_normalization(x: np.ndarray, static: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    x_mean = np.nanmean(x, axis=(0, 1)).astype(np.float32)
    x_std = np.nanstd(x, axis=(0, 1)).astype(np.float32)
    x_std[~np.isfinite(x_std) | (x_std < 1e-6)] = 1.0
    static_mean = np.nanmean(static, axis=0).astype(np.float32)
    static_std = np.nanstd(static, axis=0).astype(np.float32)
    static_std[~np.isfinite(static_std) | (static_std < 1e-6)] = 1.0
    y_mean = float(np.nanmean(y))
    y_std = max(float(np.nanstd(y)), 1e-6)
    return {
        "x_mean": x_mean, "x_std": x_std,
        "static_mean": static_mean, "static_std": static_std,
        "y_mean": np.asarray([y_mean], dtype=np.float32),
        "y_std": np.asarray([y_std], dtype=np.float32),
    }


def transform_inputs(
    x: np.ndarray, static: np.ndarray, norm: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    missing = np.isnan(x).astype(np.float32)
    imputed = np.where(np.isnan(x), norm["x_mean"][None, None, :], x)
    scaled = (imputed - norm["x_mean"][None, None, :]) / norm["x_std"][None, None, :]
    dynamic = np.concatenate([scaled.astype(np.float32), missing], axis=2)
    static_imputed = np.where(np.isnan(static), norm["static_mean"][None, :], static)
    static_scaled = (static_imputed - norm["static_mean"][None, :]) / norm["static_std"][None, :]
    return dynamic.astype(np.float32), static_scaled.astype(np.float32)


def build_model(args: argparse.Namespace):
    import torch.nn as nn

    class JuneLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = args.dropout if args.lstm_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=len(CHANNELS) * 2,
                hidden_size=args.hidden_size,
                num_layers=args.lstm_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(args.hidden_size + len(STATIC_FEATURES), args.hidden_size),
                nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.hidden_size, 1),
            )

        def forward(self, dynamic, static):
            _, (hidden, _) = self.lstm(dynamic)
            return self.head(__import__("torch").cat([hidden[-1], static], dim=1)).squeeze(1)

    return JuneLSTM()


def device_for(args: argparse.Namespace):
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.threads))
    return device


def predict_model(model, x: np.ndarray, static: np.ndarray, norm, device) -> np.ndarray:
    import torch

    dynamic, static_scaled = transform_inputs(x, static, norm)
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(dynamic), 512):
            parts.append(model(
                torch.from_numpy(dynamic[start:start + 512]).to(device),
                torch.from_numpy(static_scaled[start:start + 512]).to(device),
            ).cpu().numpy())
    scaled = np.concatenate(parts)
    return scaled * norm["y_std"][0] + norm["y_mean"][0]


def train_dev(
    x_train, static_train, y_train, x_dev, static_dev, y_dev,
    args: argparse.Namespace, seed: int,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)
    device = device_for(args)
    norm = fit_normalization(x_train, static_train, y_train)
    dynamic, static_scaled = transform_inputs(x_train, static_train, norm)
    target = ((y_train - norm["y_mean"][0]) / norm["y_std"][0]).astype(np.float32)
    model = build_model(args).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(dynamic), torch.from_numpy(static_scaled), torch.from_numpy(target)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    best_state = None
    best_epoch = 1
    best_score = math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for dynamic_batch, static_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(
                model(dynamic_batch.to(device), static_batch.to(device)), target_batch.to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        prediction = predict_model(model, x_dev, static_dev, norm, device)
        score = rmse(y_dev, prediction)
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)), "dev_RMSE": score,
        })
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("LSTM early-stopping checkpoint was not created")
    return best_epoch, best_score, pd.DataFrame(history)


def train_fixed_predict(
    x_train, static_train, y_train, x_test, static_test,
    epochs: int, args: argparse.Namespace, seed: int,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)
    device = device_for(args)
    norm = fit_normalization(x_train, static_train, y_train)
    dynamic, static_scaled = transform_inputs(x_train, static_train, norm)
    target = ((y_train - norm["y_mean"][0]) / norm["y_std"][0]).astype(np.float32)
    model = build_model(args).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(dynamic), torch.from_numpy(static_scaled), torch.from_numpy(target)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    for _ in range(max(1, int(epochs))):
        model.train()
        for dynamic_batch, static_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(
                model(dynamic_batch.to(device), static_batch.to(device)), target_batch.to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return predict_model(model, x_test, static_test, norm, device)


def fit_catboost(train, features, target, args, seed):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=args.cat_iterations, depth=7, learning_rate=0.04, l2_leaf_reg=8.0,
        loss_function="RMSE", random_seed=seed, thread_count=max(1, args.threads),
        verbose=False, allow_writing_files=False,
    )
    model.fit(train[features], train[target])
    return model


def fit_ridge(train: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=100.0)),
    ])
    model.fit(train[base.TABULAR_FEATURES], train["TA"])
    return model


def build_baseline_predictions(
    master_path: str | Path,
    stations: pd.DataFrame,
    sequence_keys: pd.DataFrame,
    years: list[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(master_path)
    team_frame = team.engineer_team_features(raw, stations)
    predictions = []
    importance_rows = []
    for year in years:
        print(f"[BASELINE OOF] train <= {year - 1}, predict {year}", flush=True)
        stats = base.fit_channel_stats(team_frame, year - 1)
        engineered = base.engineer_tabular(team_frame, stats)
        train = engineered[engineered.year.le(year - 1)].copy()
        keys = sequence_keys.loc[sequence_keys.year.eq(year), KEYS]
        validation = keys.merge(engineered, on=KEYS, how="left", validate="1:1")
        if validation[base.TABULAR_FEATURES].isna().all(axis=1).any():
            raise ValueError(f"{year}: sequence keys are absent from the master 14:00 table")

        ta_train = train[train.TA.notna()]
        hm_train = train[train.HM.notna()]
        ta_cat = fit_catboost(
            ta_train, base.TABULAR_FEATURES, "TA", args, args.baseline_seed + year
        )
        ta_ridge = fit_ridge(ta_train)
        hm_base = fit_catboost(
            hm_train, base.TABULAR_FEATURES, "HM", args, args.baseline_seed + 1000 + year
        )
        hm_spatial = fit_catboost(
            hm_train, frozen_team.FROZEN_HM_FEATURES, "HM", args,
            args.baseline_seed + 2000 + year,
        )
        part = validation[[*KEYS, "year", "TA", "HM"]].rename(
            columns={"TA": "actual_TA", "HM": "actual_HM"}
        )
        part["baseline_TA"] = (
            JUNE_RECIPE["ta_catboost_weight"] * ta_cat.predict(validation[base.TABULAR_FEATURES])
            + JUNE_RECIPE["ta_ridge_weight"] * ta_ridge.predict(validation[base.TABULAR_FEATURES])
            + JUNE_RECIPE["ta_offset"]
        )
        part["baseline_HM"] = np.clip(
            JUNE_RECIPE["hm_base_catboost_weight"] * hm_base.predict(validation[base.TABULAR_FEATURES])
            + JUNE_RECIPE["hm_spatial_catboost_weight"]
            * hm_spatial.predict(validation[frozen_team.FROZEN_HM_FEATURES]),
            0.0, 100.0,
        )
        predictions.append(part)
        if year == max(years):
            for target, model, features, branch in [
                ("TA", ta_cat, base.TABULAR_FEATURES, "TA_base_catboost"),
                ("HM", hm_base, base.TABULAR_FEATURES, "HM_base_catboost"),
                ("HM", hm_spatial, frozen_team.FROZEN_HM_FEATURES, "HM_spatial_catboost"),
            ]:
                for feature, value in zip(features, model.get_feature_importance()):
                    importance_rows.append({
                        "target": target, "branch": branch, "feature": feature,
                        "importance": float(value),
                    })
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(importance_rows)


def optimize_direct_weight(actual, baseline, direct, target, step) -> tuple[float, float]:
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    scores = [rmse(actual, clip_target((1.0 - w) * baseline + w * direct, target)) for w in weights]
    index = int(np.nanargmin(scores))
    return float(weights[index]), float(scores[index])


def optimize_residual_scale(actual, baseline, residual, target, step, maximum) -> tuple[float, float]:
    scales = np.arange(0.0, maximum + step / 2.0, step)
    scores = [rmse(actual, clip_target(baseline + s * residual, target)) for s in scales]
    index = int(np.nanargmin(scores))
    return float(scales[index]), float(scores[index])


def metric_row(frame: pd.DataFrame, candidate: str, target: str, years: list[int]) -> dict[str, object]:
    mask = frame.year.isin(years)
    return {
        "period": ",".join(map(str, years)), "candidate": candidate, "target": target,
        "n": int(mask.sum()),
        "RMSE": rmse(
            frame.loc[mask, f"actual_{target}"].to_numpy(float),
            frame.loc[mask, f"{candidate}_{target}"].to_numpy(float),
        ),
    }


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds = [args.seeds[0]]
        args.epochs = min(args.epochs, 25)
        args.cat_iterations = min(args.cat_iterations, 250)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stations = team.clean_station_list(args.station_list)
    sequence = load_sequences(
        args.shortterm_csv, stations, args.max_sequence_missing_fraction
    )
    # The collector emits the complete 96-station grid.  A handful of historical
    # master rows can be absent; they are not fabricated and cannot be scored.
    labelled = np.isfinite(sequence["labels"]["TA"]) & np.isfinite(sequence["labels"]["HM"])
    for name in ["x", "static"]:
        sequence[name] = sequence[name][labelled]
    for target in TARGETS:
        sequence["labels"][target] = sequence["labels"][target][labelled]
    sequence["keys"] = sequence["keys"].loc[labelled].reset_index(drop=True)
    sequence["quality"] = sequence["quality"].loc[labelled].reset_index(drop=True)
    sequence["quality"].to_csv(output_dir / "sequence_quality.csv", index=False)
    keys = sequence["keys"]
    quality_ok = sequence["quality"].usable.to_numpy(dtype=bool)
    wanted_years = sorted(set(args.sequence_years))
    absent = sorted(set([*args.selection_years, args.report_year]) - set(keys.year.unique()))
    if absent:
        raise ValueError(f"required sequence years absent: {absent}")

    baseline_path = output_dir / "baseline_chronological_predictions.csv"
    importance_path = output_dir / "baseline_feature_importance.csv"
    if baseline_path.is_file() and not args.force_baseline:
        baseline = pd.read_csv(baseline_path)
        required_years = set(wanted_years)
        if not required_years.issubset(set(pd.to_numeric(baseline.year).astype(int).unique())):
            raise ValueError("cached baseline lacks requested years; rerun with --force-baseline")
        importance = pd.read_csv(importance_path) if importance_path.is_file() else pd.DataFrame()
        print(f"[BASELINE] reused {baseline_path}", flush=True)
    else:
        baseline, importance = build_baseline_predictions(
            args.master_csv, stations, keys, wanted_years, args
        )
        baseline.to_csv(baseline_path, index=False)
        importance.to_csv(importance_path, index=False)

    aligned = keys.reset_index().merge(
        baseline, on=[*KEYS, "year"], how="left", validate="1:1"
    ).sort_values("index").reset_index(drop=True)
    if aligned[["baseline_TA", "baseline_HM"]].isna().any().any():
        raise ValueError("baseline prediction alignment produced missing rows")
    for target in TARGETS:
        seq_label = sequence["labels"][target]
        master_label = aligned[f"actual_{target}"].to_numpy(dtype=float)
        mismatch = np.isfinite(seq_label) & np.isfinite(master_label) & ~np.isclose(seq_label, master_label)
        if mismatch.any():
            raise ValueError(f"{target}: short-term labels do not match master labels")
        aligned[f"actual_{target}"] = np.where(np.isfinite(seq_label), seq_label, master_label)

    all_seed_parts = []
    epoch_rows = []
    for seed in args.seeds:
        for predict_year in sorted(set([*args.selection_years, args.report_year])):
            dev_year = predict_year - 1
            train_years = [year for year in wanted_years if year < dev_year]
            refit_years = [year for year in wanted_years if year < predict_year]
            if not train_years or dev_year not in wanted_years:
                raise ValueError(
                    f"predict {predict_year}: need at least one train year plus dev year {dev_year}"
                )
            for target in TARGETS:
                actual = aligned[f"actual_{target}"].to_numpy(float)
                baseline_values = aligned[f"baseline_{target}"].to_numpy(float)
                valid = quality_ok & np.isfinite(actual) & np.isfinite(baseline_values)
                train_mask = valid & keys.year.isin(train_years).to_numpy()
                dev_mask = valid & keys.year.eq(dev_year).to_numpy()
                refit_mask = valid & keys.year.isin(refit_years).to_numpy()
                test_mask = valid & keys.year.eq(predict_year).to_numpy()
                if min(train_mask.sum(), dev_mask.sum(), refit_mask.sum(), test_mask.sum()) == 0:
                    raise ValueError(f"empty LSTM split: seed={seed}, target={target}, year={predict_year}")
                for branch in ["direct", "residual"]:
                    y_values = actual if branch == "direct" else actual - baseline_values
                    fit_seed = seed + 10000 * (target == "HM") + 1000 * (branch == "residual") + predict_year
                    best_epoch, dev_score, history = train_dev(
                        sequence["x"][train_mask], sequence["static"][train_mask], y_values[train_mask],
                        sequence["x"][dev_mask], sequence["static"][dev_mask], y_values[dev_mask],
                        args, fit_seed,
                    )
                    prediction = train_fixed_predict(
                        sequence["x"][refit_mask], sequence["static"][refit_mask], y_values[refit_mask],
                        sequence["x"][test_mask], sequence["static"][test_mask],
                        best_epoch, args, fit_seed,
                    )
                    test_indices = np.flatnonzero(test_mask)
                    part = aligned.loc[test_indices, [*KEYS, "year", f"actual_{target}", f"baseline_{target}"]].copy()
                    part["seed"] = seed
                    part["target"] = target
                    part["branch"] = branch
                    part["lstm_output"] = prediction
                    all_seed_parts.append(part)
                    history.insert(0, "predict_year", predict_year)
                    history.insert(0, "branch", branch)
                    history.insert(0, "target", target)
                    history.insert(0, "seed", seed)
                    history.to_csv(
                        output_dir / f"history_seed{seed}_{target}_{branch}_{predict_year}.csv", index=False
                    )
                    epoch_rows.append({
                        "seed": seed, "target": target, "branch": branch,
                        "predict_year": predict_year, "dev_year": dev_year,
                        "best_epoch": best_epoch, "dev_target_RMSE": dev_score,
                        "train_n": int(train_mask.sum()), "dev_n": int(dev_mask.sum()),
                        "refit_n": int(refit_mask.sum()), "test_n": int(test_mask.sum()),
                    })
                    print(
                        f"[LSTM] seed={seed} {target}/{branch} predict={predict_year} "
                        f"epoch={best_epoch} dev_RMSE={dev_score:.5f}", flush=True,
                    )

    by_seed = pd.concat(all_seed_parts, ignore_index=True)
    by_seed.to_csv(output_dir / "lstm_predictions_by_seed.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(output_dir / "lstm_epoch_selection.csv", index=False)
    # Concatenation creates separate target-specific columns.  Collapse those
    # into common columns before averaging the LSTM seeds.
    by_seed["actual"] = np.where(
        by_seed.target.eq("TA"), by_seed.get("actual_TA"), by_seed.get("actual_HM")
    )
    by_seed["baseline"] = np.where(
        by_seed.target.eq("TA"), by_seed.get("baseline_TA"), by_seed.get("baseline_HM")
    )
    averaged = by_seed.groupby([*KEYS, "year", "target", "branch"], as_index=False).agg(
        actual=("actual", "first"), baseline=("baseline", "first"),
        lstm_output=("lstm_output", "mean"),
    )
    wide_parts = []
    for target in TARGETS:
        direct = averaged[(averaged.target.eq(target)) & averaged.branch.eq("direct")].copy()
        residual = averaged[(averaged.target.eq(target)) & averaged.branch.eq("residual")].copy()
        part = direct[[*KEYS, "year", "actual", "baseline", "lstm_output"]].rename(columns={
            "actual": f"actual_{target}", "baseline": f"baseline_{target}",
            "lstm_output": f"direct_{target}",
        })
        part = part.merge(
            residual[[*KEYS, "year", "lstm_output"]].rename(
                columns={"lstm_output": f"residual_{target}"}
            ), on=[*KEYS, "year"], how="inner", validate="1:1",
        )
        wide_parts.append(part)
    predictions = wide_parts[0].merge(wide_parts[1], on=[*KEYS, "year"], validate="1:1")

    selection_mask = predictions.year.isin(args.selection_years)
    recipes = {}
    for target in TARGETS:
        actual = predictions.loc[selection_mask, f"actual_{target}"].to_numpy(float)
        baseline_values = predictions.loc[selection_mask, f"baseline_{target}"].to_numpy(float)
        direct_values = predictions.loc[selection_mask, f"direct_{target}"].to_numpy(float)
        residual_values = predictions.loc[selection_mask, f"residual_{target}"].to_numpy(float)
        weight, direct_score = optimize_direct_weight(
            actual, baseline_values, direct_values, target, args.weight_step
        )
        scale, residual_score = optimize_residual_scale(
            actual, baseline_values, residual_values, target,
            args.weight_step, args.residual_scale_max,
        )
        baseline_score = rmse(actual, baseline_values)
        candidate_scores = {
            "baseline": baseline_score,
            "direct_blend": direct_score,
            "residual_blend": residual_score,
        }
        best_name = min(candidate_scores, key=candidate_scores.get)
        improvement = baseline_score - candidate_scores[best_name]
        minimum = args.min_improvement_ta if target == "TA" else args.min_improvement_hm
        # Require non-trivial contribution and improvement in at least two OOF years.
        provisional = best_name != "baseline" and improvement >= minimum
        predictions[f"direct_blend_{target}"] = clip_target(
            (1.0 - weight) * predictions[f"baseline_{target}"]
            + weight * predictions[f"direct_{target}"], target,
        )
        predictions[f"residual_blend_{target}"] = clip_target(
            predictions[f"baseline_{target}"] + scale * predictions[f"residual_{target}"], target,
        )
        chosen_column = f"{best_name}_{target}"
        if best_name == "baseline":
            chosen_column = f"baseline_{target}"
        year_improvements = {}
        for year in args.selection_years:
            mask = predictions.year.eq(year)
            base_year = rmse(
                predictions.loc[mask, f"actual_{target}"].to_numpy(float),
                predictions.loc[mask, f"baseline_{target}"].to_numpy(float),
            )
            best_year = rmse(
                predictions.loc[mask, f"actual_{target}"].to_numpy(float),
                predictions.loc[mask, chosen_column].to_numpy(float),
            )
            year_improvements[str(year)] = base_year - best_year
        stable_years = sum(value >= 0.0 for value in year_improvements.values())
        accepted = bool(provisional and stable_years >= 2)
        selected_name = best_name if accepted else "baseline"
        predictions[f"selected_{target}"] = predictions[
            f"{selected_name}_{target}" if selected_name != "baseline" else f"baseline_{target}"
        ]
        recipes[target] = {
            "direct_lstm_weight": weight,
            "residual_scale": scale,
            "OOF_baseline_RMSE": baseline_score,
            "OOF_direct_LSTM_RMSE": rmse(actual, direct_values),
            "OOF_direct_blend_RMSE": direct_score,
            "OOF_residual_blend_RMSE": residual_score,
            "best_candidate_before_gate": best_name,
            "best_OOF_improvement": improvement,
            "minimum_required_improvement": minimum,
            "non_worse_OOF_years": stable_years,
            "OOF_year_improvements": year_improvements,
            "accepted": accepted,
            "selected_candidate": selected_name,
        }

    predictions.to_csv(output_dir / "all_candidate_predictions.csv", index=False)
    metric_rows = []
    candidates = ["baseline", "direct", "direct_blend", "residual_blend", "selected"]
    for period, years in [
        ("selection_OOF", args.selection_years), ("locked_report", [args.report_year])
    ]:
        for target in TARGETS:
            for candidate in candidates:
                row = metric_row(predictions, candidate, target, years)
                row["period"] = period
                metric_rows.append(row)
        period_frame = predictions[predictions.year.isin(years)]
        for candidate in candidates:
            ta_score = rmse(period_frame.actual_TA.to_numpy(float), period_frame[f"{candidate}_TA"].to_numpy(float))
            hm_score = rmse(period_frame.actual_HM.to_numpy(float), period_frame[f"{candidate}_HM"].to_numpy(float))
            metric_rows.append({
                "period": period, "candidate": candidate, "target": "competition_score",
                "n": len(period_frame), "RMSE": ta_score + 0.1 * hm_score,
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "model_comparison.csv", index=False)

    report_metrics = metrics[metrics.period.eq("locked_report")]
    report_base = float(report_metrics.loc[
        (report_metrics.candidate.eq("baseline")) &
        (report_metrics.target.eq("competition_score")), "RMSE"
    ].iloc[0])
    report_selected = float(report_metrics.loc[
        (report_metrics.candidate.eq("selected")) &
        (report_metrics.target.eq("competition_score")), "RMSE"
    ].iloc[0])
    selection_metrics = metrics[metrics.period.eq("selection_OOF")]
    selection_base = float(selection_metrics.loc[
        (selection_metrics.candidate.eq("baseline")) &
        (selection_metrics.target.eq("competition_score")), "RMSE"
    ].iloc[0])
    selection_selected = float(selection_metrics.loc[
        (selection_metrics.candidate.eq("selected")) &
        (selection_metrics.target.eq("competition_score")), "RMSE"
    ].iloc[0])
    summary = {
        "experiment": "June core8 five-time LSTM feasibility",
        "channels": CHANNELS,
        "times_kst": TIMES.astype(int).tolist(),
        "static_features": STATIC_FEATURES,
        "selection_years": args.selection_years,
        "locked_report_year": args.report_year,
        "lstm_seeds": args.seeds,
        "current_14h_recipe": JUNE_RECIPE,
        "target_recipes": recipes,
        "selection_competition_score": {
            "baseline": selection_base,
            "selected": selection_selected,
            "improvement": selection_base - selection_selected,
        },
        "locked_report_competition_score": {
            "baseline": report_base,
            "selected": report_selected,
            "improvement": report_base - report_selected,
        },
        "OOF_decision": {
            "add_LSTM_to_TA": recipes["TA"]["accepted"],
            "add_LSTM_to_HM": recipes["HM"]["accepted"],
            "note": "Decision uses 2022-2024 OOF only. 2025 is an untouched diagnostic report.",
        },
        "rules_contract": {
            "asos_used_as_labels_only": True,
            "asos_lag_features": False,
            "direct_year_feature": False,
            "satellite_path": "/GK2A/LE1B/{channel}/KO/data",
            "official_station_coordinates": True,
        },
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + metrics.to_string(index=False), flush=True)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
