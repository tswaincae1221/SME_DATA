#!/usr/bin/env python3
"""Train a two-branch TA baseline with branch-specific 40-feature inputs.

Branch A (specialist)
    12:00--14:00 KST, 13 steps x 16 GK-2A channels -> small LSTM.
    Sixteen missing masks and eight static/context values are included, for a
    total of 40 named input features.

Branch B (generalist)
    All available June--August 14:00 rows -> CatBoost and Ridge.  The branch
    uses 16 raw channels, 11 training-standardised DN differences, three
    official station values, and ten date/solar values: 40 features total.

The development split is 2019--2023 / 2024.  Hyperparameters and hierarchical
blend weights are selected only on 2024.  An evaluation refit through 2024 is
reported on 2025, then deployment models are refit on every 2019--2025 label.
ASOS TA is a label only.  Direct year terms and ASOS lag features are excluded.

If the current runtime lacks PyTorch/CatBoost, ``--reuse-development-dir`` and
``--reuse-final-dir`` can reuse already-generated LSTM/CatBoost predictions
while training the new 40-feature Ridge branch.  This mode is a compatibility
diagnostic, not a replacement for the exact Colab training run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = [
    "VI004", "VI005", "VI006", "VI008", "NR013", "NR016", "SW038",
    "WV063", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112",
    "IR123", "IR133",
]
STANDARDIZED_PAIRS = [
    ("IR105", "IR112", "clean_window_105_112"),
    ("IR112", "IR123", "split_window_112_123"),
    ("IR105", "IR123", "window_105_123"),
    ("IR096", "IR112", "ozone_window_096_112"),
    ("IR087", "IR112", "cloud_phase_087_112"),
    ("WV073", "IR112", "low_wv_window_073_112"),
    ("WV069", "IR112", "mid_wv_window_069_112"),
    ("WV063", "IR112", "upper_wv_window_063_112"),
    ("SW038", "IR112", "fog_window_038_112"),
    ("IR133", "IR112", "co2_window_133_112"),
    ("WV063", "WV073", "wv_vertical_063_073"),
]
PAIR_FEATURES = [name for _, _, name in STANDARDIZED_PAIRS]
TABULAR_FEATURES = [
    *CHANNELS,
    *PAIR_FEATURES,
    "LAT", "LON", "ALT",
    "doy_sin", "doy_cos", "cos_solar_zenith",
    "month", "day", "dayofyear", "solar_azimuth", "solar_declination",
    "local_solar_time", "equation_of_time",
]
LSTM_STATIC_FEATURES = [
    "LAT", "LON", "ALT", "doy_sin", "doy_cos", "cos_solar_zenith",
    "solar_azimuth", "sequence_missing_fraction",
]
EXPECTED_TIMES = np.array(
    [1200, 1210, 1220, 1230, 1240, 1250, 1300, 1310, 1320, 1330, 1340, 1350, 1400],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cat-iterations", type=int, default=500)
    parser.add_argument("--cat-depth", type=int, default=7)
    parser.add_argument("--cat-learning-rate", type=float, default=0.04)
    parser.add_argument("--cat-l2", type=float, default=8.0)
    parser.add_argument("--cat-early-stopping-rounds", type=int, default=70)
    parser.add_argument("--weight-step", type=float, default=0.02)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-development-dir")
    parser.add_argument("--reuse-final-dir")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clean_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = pd.to_numeric(out["Date"], errors="raise").round().astype("int64")
    out["STN_ID"] = pd.to_numeric(out["STN_ID"], errors="raise").round().astype("int64")
    date = pd.to_datetime(out["Date"].astype(str), format="%Y%m%d", errors="raise")
    out["year"] = date.dt.year.astype("int16")
    out["month"] = date.dt.month.astype("int8")
    out["day"] = date.dt.day.astype("int8")
    out["dayofyear"] = date.dt.dayofyear.astype("int16")
    return out


def add_solar_features(frame: pd.DataFrame, hour_kst: float = 14.0) -> pd.DataFrame:
    out = clean_keys(frame)
    doy = out["dayofyear"].to_numpy(dtype=float)
    year = out["year"].to_numpy(dtype=int)
    leap = ((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))).astype(float)
    days_in_year = 365.0 + leap
    gamma = 2.0 * np.pi / days_in_year * (doy - 1.0 + (hour_kst - 12.0) / 24.0)
    equation = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma) - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma) + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma) + 0.00148 * np.sin(3.0 * gamma)
    )
    longitude = out["LON"].to_numpy(dtype=float)
    latitude = np.radians(out["LAT"].to_numpy(dtype=float))
    true_solar_minutes = hour_kst * 60.0 + equation + 4.0 * longitude - 60.0 * 9.0
    hour_angle_deg = true_solar_minutes / 4.0 - 180.0
    hour_angle = np.radians(hour_angle_deg)
    cos_zenith = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    azimuth = (
        np.degrees(
            np.arctan2(
                np.sin(hour_angle),
                np.cos(hour_angle) * np.sin(latitude) - np.tan(declination) * np.cos(latitude),
            )
        )
        + 180.0
    ) % 360.0
    angle = 2.0 * np.pi * doy / 365.25
    out["doy_sin"] = np.sin(angle)
    out["doy_cos"] = np.cos(angle)
    out["cos_solar_zenith"] = cos_zenith
    out["solar_azimuth"] = azimuth
    out["solar_declination"] = np.degrees(declination)
    out["local_solar_time"] = true_solar_minutes / 60.0
    out["equation_of_time"] = equation
    return out


def fit_channel_stats(frame: pd.DataFrame, train_end_year: int) -> dict[str, dict[str, float]]:
    train = frame[frame["year"] <= train_end_year]
    means = train[CHANNELS].mean(skipna=True)
    stds = train[CHANNELS].std(skipna=True).replace(0.0, 1.0).fillna(1.0)
    return {
        "mean": {channel: float(means[channel]) for channel in CHANNELS},
        "std": {channel: float(stds[channel]) for channel in CHANNELS},
    }


def engineer_tabular(
    solar_master: pd.DataFrame,
    stats: dict[str, dict[str, float]],
) -> pd.DataFrame:
    out = solar_master.copy()
    for left, right, name in STANDARDIZED_PAIRS:
        left_z = (out[left] - stats["mean"][left]) / stats["std"][left]
        right_z = (out[right] - stats["mean"][right]) / stats["std"][right]
        out[name] = left_z - right_z
    missing = [feature for feature in TABULAR_FEATURES if feature not in out]
    if missing:
        raise ValueError(f"engineered tabular features are missing: {missing}")
    return out


def feature_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for channel in CHANNELS:
        rows.append({"branch": "LSTM", "feature": f"sequence_{channel}", "priority": "P1", "role": "13-step channel sequence"})
    for feature in ["LAT", "LON", "ALT", "cos_solar_zenith"]:
        rows.append({"branch": "LSTM", "feature": feature, "priority": "P1", "role": "station/solar context"})
    for channel in CHANNELS:
        rows.append({"branch": "LSTM", "feature": f"missing_mask_{channel}", "priority": "P2", "role": "13-step missing mask"})
    for feature in ["doy_sin", "doy_cos"]:
        rows.append({"branch": "LSTM", "feature": feature, "priority": "P2", "role": "seasonal context"})
    for feature in ["solar_azimuth", "sequence_missing_fraction"]:
        rows.append({"branch": "LSTM", "feature": feature, "priority": "P3", "role": "quality/solar context"})

    for channel in CHANNELS:
        rows.append({"branch": "Tabular", "feature": channel, "priority": "P1", "role": "14:00 absolute DN"})
    for feature in ["LAT", "LON", "ALT", "cos_solar_zenith"]:
        rows.append({"branch": "Tabular", "feature": feature, "priority": "P1", "role": "station/solar context"})
    for feature in [*PAIR_FEATURES, "doy_sin", "doy_cos"]:
        rows.append({"branch": "Tabular", "feature": feature, "priority": "P2", "role": "spectral/seasonal relation"})
    for feature in ["month", "day", "dayofyear", "solar_azimuth", "solar_declination", "local_solar_time", "equation_of_time"]:
        rows.append({"branch": "Tabular", "feature": feature, "priority": "P3", "role": "calendar/solar refinement"})
    result = pd.DataFrame(rows)
    counts = result.groupby("branch").size().to_dict()
    if counts != {"LSTM": 40, "Tabular": 40}:
        raise AssertionError(f"unexpected feature counts: {counts}")
    return result


def build_sequences(long_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    required = ["Date", "TimeKST", "STN_ID", "LAT", "LON", "ALT", "TA", *CHANNELS]
    missing = [column for column in required if column not in long_frame]
    if missing:
        raise ValueError(f"short-term CSV is missing: {missing}")
    frame = add_solar_features(long_frame)
    frame["TimeKST"] = pd.to_numeric(frame["TimeKST"], errors="raise").astype("int64")
    dynamic: list[np.ndarray] = []
    static: list[np.ndarray] = []
    labels: list[float] = []
    keys: list[tuple[int, int, int]] = []
    missing_counts: list[int] = []
    for (date, station), group in frame.groupby(["Date", "STN_ID"], sort=True):
        group = group.sort_values("TimeKST")
        if len(group) != len(EXPECTED_TIMES) or not np.array_equal(group["TimeKST"].to_numpy(), EXPECTED_TIMES):
            continue
        values = group[CHANNELS].to_numpy(dtype=np.float32)
        missing_count = int(np.isnan(values).sum())
        first = group.iloc[0]
        context = np.array(
            [
                first["LAT"], first["LON"], first["ALT"], first["doy_sin"], first["doy_cos"],
                first["cos_solar_zenith"], first["solar_azimuth"], missing_count / float(values.size),
            ],
            dtype=np.float32,
        )
        label = float(group.loc[group["TimeKST"] == 1400, "TA"].iloc[0])
        dynamic.append(values)
        static.append(context)
        labels.append(label)
        keys.append((int(date), int(station), int(date) // 10000))
        missing_counts.append(missing_count)
    return (
        np.stack(dynamic), np.stack(static), np.asarray(labels, dtype=np.float32),
        pd.DataFrame(keys, columns=["Date", "STN_ID", "year"]),
        np.asarray(missing_counts, dtype=np.int32),
    )


def fit_lstm_normalization(x: np.ndarray, static: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    x_mean = np.nanmean(x, axis=(0, 1)).astype(np.float32)
    x_std = np.nanstd(x, axis=(0, 1)).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    static_mean = np.nanmean(static, axis=0).astype(np.float32)
    static_std = np.nanstd(static, axis=0).astype(np.float32)
    static_std[static_std < 1e-6] = 1.0
    return {
        "x_mean": x_mean, "x_std": x_std,
        "static_mean": static_mean, "static_std": static_std,
        "y_mean": np.asarray([np.nanmean(y)], dtype=np.float32),
        "y_std": np.asarray([max(float(np.nanstd(y)), 1e-6)], dtype=np.float32),
    }


def transform_lstm_inputs(
    x: np.ndarray,
    static: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    missing = np.isnan(x).astype(np.float32)
    imputed = np.where(np.isnan(x), normalization["x_mean"][None, None, :], x)
    scaled = (imputed - normalization["x_mean"][None, None, :]) / normalization["x_std"][None, None, :]
    dynamic = np.concatenate([scaled.astype(np.float32), missing], axis=2)
    static_imputed = np.where(np.isnan(static), normalization["static_mean"][None, :], static)
    static_scaled = (static_imputed - normalization["static_mean"][None, :]) / normalization["static_std"][None, :]
    return dynamic.astype(np.float32), static_scaled.astype(np.float32)


def build_lstm_model(args: argparse.Namespace):
    import torch
    import torch.nn as nn

    class TABaselineLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = args.dropout if args.lstm_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=32,
                hidden_size=args.hidden_size,
                num_layers=args.lstm_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(args.hidden_size + 8, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, 1),
            )

        def forward(self, dynamic, static):
            _, (hidden, _) = self.lstm(dynamic)
            return self.head(torch.cat([hidden[-1], static], dim=1)).squeeze(1)

    return TABaselineLSTM()


def torch_device(args: argparse.Namespace):
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.threads))
    return device


def predict_lstm(model, dynamic: np.ndarray, static: np.ndarray, normalization, device) -> np.ndarray:
    import torch

    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(dynamic), 512):
            d = torch.from_numpy(dynamic[start : start + 512]).to(device)
            s = torch.from_numpy(static[start : start + 512]).to(device)
            parts.append(model(d, s).cpu().numpy())
    scaled = np.concatenate(parts)
    return scaled * normalization["y_std"][0] + normalization["y_mean"][0]


def train_lstm_dev(x_train, static_train, y_train, x_val, static_val, y_val, args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch_device(args)
    normalization = fit_lstm_normalization(x_train, static_train, y_train)
    train_dynamic, train_static = transform_lstm_inputs(x_train, static_train, normalization)
    val_dynamic, val_static = transform_lstm_inputs(x_val, static_val, normalization)
    train_target = ((y_train - normalization["y_mean"][0]) / normalization["y_std"][0]).astype(np.float32)
    model = build_lstm_model(args).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_dynamic), torch.from_numpy(train_static), torch.from_numpy(train_target)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = torch.nn.MSELoss()
    best_state = None
    best_rmse = math.inf
    best_epoch = 1
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for dynamic_batch, static_batch, target_batch in loader:
            dynamic_batch = dynamic_batch.to(device)
            static_batch = static_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(dynamic_batch, static_batch), target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        prediction = predict_lstm(model, val_dynamic, val_static, normalization, device)
        score = rmse(y_val, prediction)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_RMSE": score})
        if score < best_rmse - 1e-6:
            best_rmse = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"[LSTM] epoch={epoch} val_RMSE={score:.5f} best={best_rmse:.5f}", flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("LSTM checkpoint selection failed")
    model.load_state_dict(best_state)
    return model, normalization, device, best_epoch, pd.DataFrame(history)


def train_lstm_fixed(x, static, y, epochs: int, args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch_device(args)
    normalization = fit_lstm_normalization(x, static, y)
    dynamic, static_scaled = transform_lstm_inputs(x, static, normalization)
    target = ((y - normalization["y_mean"][0]) / normalization["y_std"][0]).astype(np.float32)
    model = build_lstm_model(args).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(dynamic), torch.from_numpy(static_scaled), torch.from_numpy(target)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = torch.nn.MSELoss()
    for _ in range(max(1, epochs)):
        model.train()
        for dynamic_batch, static_batch, target_batch in loader:
            dynamic_batch = dynamic_batch.to(device)
            static_batch = static_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(dynamic_batch, static_batch), target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model, normalization, device


def save_lstm(path: Path, model, normalization, best_epoch: int, args) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_size": args.hidden_size,
            "lstm_layers": args.lstm_layers,
            "dropout": args.dropout,
            "epochs": best_epoch,
            "channels": CHANNELS,
            "static_features": LSTM_STATIC_FEATURES,
        },
        path,
    )
    np.savez_compressed(path.with_name(path.stem + "_normalization.npz"), **normalization)


def fit_ridge(train: pd.DataFrame, validation: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    best_model = None
    best_alpha = None
    best_score = math.inf
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(train[TABULAR_FEATURES], train["TA"])
        score = rmse(validation["TA"].to_numpy(), model.predict(validation[TABULAR_FEATURES]))
        if score < best_score:
            best_model, best_alpha, best_score = model, alpha, score
    return best_model, float(best_alpha), float(best_score)


def refit_ridge(train: pd.DataFrame, alpha: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(train[TABULAR_FEATURES], train["TA"])
    return model


def train_catboost_dev(train: pd.DataFrame, validation: pd.DataFrame, args):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=args.cat_iterations,
        depth=args.cat_depth,
        learning_rate=args.cat_learning_rate,
        l2_leaf_reg=args.cat_l2,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=args.seed,
        thread_count=max(1, args.threads),
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(
        train[TABULAR_FEATURES], train["TA"],
        eval_set=(validation[TABULAR_FEATURES], validation["TA"]),
        early_stopping_rounds=args.cat_early_stopping_rounds,
        use_best_model=True,
    )
    iteration = int(model.get_best_iteration()) + 1
    return model, max(1, iteration)


def train_catboost_fixed(train: pd.DataFrame, iterations: int, args):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=max(1, iterations),
        depth=args.cat_depth,
        learning_rate=args.cat_learning_rate,
        l2_leaf_reg=args.cat_l2,
        loss_function="RMSE",
        random_seed=args.seed,
        thread_count=max(1, args.threads),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train[TABULAR_FEATURES], train["TA"])
    return model


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(prediction)
    return float(np.sqrt(np.mean((prediction[mask] - actual[mask]) ** 2)))


def diagnostic_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(actual) & np.isfinite(prediction)
    actual = actual[mask]
    prediction = prediction[mask]
    error = prediction - actual
    return {
        "n": int(mask.sum()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "corr": float(np.corrcoef(actual, prediction)[0, 1]),
    }


def optimize_weight(actual: np.ndarray, first: np.ndarray, second: np.ndarray, step: float) -> tuple[float, float]:
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    scores = [rmse(actual, weight * first + (1.0 - weight) * second) for weight in weights]
    index = int(np.argmin(scores))
    return float(weights[index]), float(scores[index])


def align_master(engineered: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    return keys[["Date", "STN_ID"]].merge(
        engineered[["Date", "STN_ID", "TA", *TABULAR_FEATURES]],
        on=["Date", "STN_ID"], how="inner", validate="1:1",
    )


def load_reused_predictions(directory: Path, year: int) -> pd.DataFrame:
    frame = pd.read_csv(directory / f"predictions_{year}.csv")
    required = ["Date", "STN_ID", "actual_TA", "lstm_TA", "catboost_TA"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"reuse predictions missing columns: {missing}")
    frame["Date"] = pd.to_numeric(frame["Date"], errors="raise").round().astype("int64")
    frame["STN_ID"] = pd.to_numeric(frame["STN_ID"], errors="raise").round().astype("int64")
    return frame[required + [column for column in ["sequence_missing_cells", "sequence_complete"] if column in frame]]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = output_dir / "evaluation_models"
    deployment_dir = output_dir / "deployment_models"
    evaluation_dir.mkdir(exist_ok=True)
    deployment_dir.mkdir(exist_ok=True)

    if len(TABULAR_FEATURES) != 40 or len(CHANNELS) * 2 + len(LSTM_STATIC_FEATURES) != 40:
        raise AssertionError("each branch must expose exactly 40 named inputs")
    feature_manifest().to_csv(output_dir / "branch_feature_manifest.csv", index=False)

    master_raw = pd.read_csv(args.master_csv)
    master_solar = add_solar_features(master_raw)
    if master_solar.duplicated(["Date", "STN_ID"]).any():
        raise ValueError("master CSV has duplicate Date/STN_ID rows")
    shortterm = pd.read_csv(args.shortterm_long_csv)
    x_raw, static_raw, sequence_y, sequence_keys, missing_counts = build_sequences(shortterm)

    dev_stats = fit_channel_stats(master_solar, args.train_end_year)
    dev_master = engineer_tabular(master_solar, dev_stats)
    final_stats = fit_channel_stats(master_solar, args.validation_year)
    final_master = engineer_tabular(master_solar, final_stats)
    deployment_stats = fit_channel_stats(master_solar, args.test_year)
    deployment_master = engineer_tabular(master_solar, deployment_stats)

    dev_train = dev_master[(dev_master["year"] <= args.train_end_year) & dev_master["TA"].notna()].copy()
    dev_validation_all = dev_master[(dev_master["year"] == args.validation_year) & dev_master["TA"].notna()].copy()
    final_train = final_master[(final_master["year"] <= args.validation_year) & final_master["TA"].notna()].copy()
    deployment_train = deployment_master[(deployment_master["year"] <= args.test_year) & deployment_master["TA"].notna()].copy()

    ridge_dev, ridge_alpha, ridge_validation_rmse = fit_ridge(dev_train, dev_validation_all)
    ridge_final = refit_ridge(final_train, ridge_alpha)
    ridge_deployment = refit_ridge(deployment_train, ridge_alpha)
    import joblib

    joblib.dump(ridge_final, evaluation_dir / "ridge_TA.joblib")
    joblib.dump(ridge_deployment, deployment_dir / "ridge_TA.joblib")

    reuse_mode = bool(args.reuse_development_dir or args.reuse_final_dir)
    if reuse_mode and not (args.reuse_development_dir and args.reuse_final_dir):
        raise ValueError("both reuse directories are required together")

    if reuse_mode:
        validation = load_reused_predictions(Path(args.reuse_development_dir), args.validation_year)
        test = load_reused_predictions(Path(args.reuse_final_dir), args.test_year)
        validation_master = align_master(dev_master, validation)
        test_master = align_master(final_master, test)
        validation = validation.merge(
            validation_master[["Date", "STN_ID", "TA"]], on=["Date", "STN_ID"], how="inner", validate="1:1"
        )
        test = test.merge(
            test_master[["Date", "STN_ID", "TA"]], on=["Date", "STN_ID"], how="inner", validate="1:1"
        )
        validation_master = align_master(dev_master, validation)
        test_master = align_master(final_master, test)
        if not np.allclose(validation["actual_TA"], validation["TA"], equal_nan=True):
            raise ValueError("reused validation labels do not match master")
        if not np.allclose(test["actual_TA"], test["TA"], equal_nan=True):
            raise ValueError("reused test labels do not match master")
        lstm_validation = validation["lstm_TA"].to_numpy(dtype=float)
        cat_validation = validation["catboost_TA"].to_numpy(dtype=float)
        lstm_test = test["lstm_TA"].to_numpy(dtype=float)
        cat_test = test["catboost_TA"].to_numpy(dtype=float)
        actual_validation = validation["TA"].to_numpy(dtype=float)
        actual_test = test["TA"].to_numpy(dtype=float)
        ridge_validation = ridge_dev.predict(validation_master[TABULAR_FEATURES])
        ridge_test = ridge_final.predict(test_master[TABULAR_FEATURES])
        validation_keys = validation[["Date", "STN_ID"]].copy()
        test_keys = test[["Date", "STN_ID"]].copy()
        test_missing = test.get("sequence_missing_cells", pd.Series(np.nan, index=test.index)).to_numpy()
        cat_iterations = None
        lstm_epoch = None
        exact_training_completed = False
    else:
        try:
            import torch  # noqa: F401
            import catboost  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Exact training requires PyTorch and CatBoost. In Colab run `pip install -q catboost`, "
                "or provide both reuse prediction directories for the local compatibility diagnostic."
            ) from error

        train_sequence_mask = (sequence_keys["year"] <= args.train_end_year).to_numpy() & np.isfinite(sequence_y)
        validation_sequence_mask = (sequence_keys["year"] == args.validation_year).to_numpy() & np.isfinite(sequence_y)
        test_sequence_mask = (sequence_keys["year"] == args.test_year).to_numpy() & np.isfinite(sequence_y)
        validation_indices = np.flatnonzero(validation_sequence_mask)
        test_indices = np.flatnonzero(test_sequence_mask)

        lstm_dev, lstm_norm, lstm_device, lstm_epoch, history = train_lstm_dev(
            x_raw[train_sequence_mask], static_raw[train_sequence_mask], sequence_y[train_sequence_mask],
            x_raw[validation_indices], static_raw[validation_indices], sequence_y[validation_indices], args,
        )
        history.to_csv(output_dir / "lstm_training_history.csv", index=False)
        all_dynamic, all_static = transform_lstm_inputs(x_raw, static_raw, lstm_norm)
        lstm_validation_all = predict_lstm(
            lstm_dev, all_dynamic[validation_indices], all_static[validation_indices], lstm_norm, lstm_device
        )

        cat_dev, cat_iterations = train_catboost_dev(dev_train, dev_validation_all, args)
        ridge_validation_all = ridge_dev.predict(dev_validation_all[TABULAR_FEATURES])
        cat_validation_all = cat_dev.predict(dev_validation_all[TABULAR_FEATURES])

        validation_sequence_keys = sequence_keys.iloc[validation_indices].reset_index(drop=True)
        validation = validation_sequence_keys.copy()
        validation["actual_TA"] = sequence_y[validation_indices]
        validation["lstm_TA"] = lstm_validation_all
        validation_master = align_master(dev_master, validation)
        validation = validation.merge(
            validation_master[["Date", "STN_ID", "TA"]], on=["Date", "STN_ID"], how="inner", validate="1:1"
        )
        validation_master = align_master(dev_master, validation)
        actual_validation = validation["TA"].to_numpy(dtype=float)
        lstm_validation = validation["lstm_TA"].to_numpy(dtype=float)
        cat_validation = cat_dev.predict(validation_master[TABULAR_FEATURES])
        ridge_validation = ridge_dev.predict(validation_master[TABULAR_FEATURES])

        final_sequence_mask = (sequence_keys["year"] <= args.validation_year).to_numpy() & np.isfinite(sequence_y)
        lstm_final, final_norm, final_device = train_lstm_fixed(
            x_raw[final_sequence_mask], static_raw[final_sequence_mask], sequence_y[final_sequence_mask],
            lstm_epoch, args,
        )
        save_lstm(evaluation_dir / "lstm_TA.pt", lstm_final, final_norm, lstm_epoch, args)
        final_dynamic, final_static = transform_lstm_inputs(x_raw, static_raw, final_norm)
        lstm_test_all = predict_lstm(
            lstm_final, final_dynamic[test_indices], final_static[test_indices], final_norm, final_device
        )
        cat_final = train_catboost_fixed(final_train, cat_iterations, args)
        cat_final.save_model(evaluation_dir / "catboost_TA.cbm")

        test_sequence_keys = sequence_keys.iloc[test_indices].reset_index(drop=True)
        test = test_sequence_keys.copy()
        test["actual_TA"] = sequence_y[test_indices]
        test["lstm_TA"] = lstm_test_all
        test_master = align_master(final_master, test)
        test = test.merge(test_master[["Date", "STN_ID", "TA"]], on=["Date", "STN_ID"], how="inner", validate="1:1")
        test_master = align_master(final_master, test)
        actual_test = test["TA"].to_numpy(dtype=float)
        lstm_test = test["lstm_TA"].to_numpy(dtype=float)
        cat_test = cat_final.predict(test_master[TABULAR_FEATURES])
        ridge_test = ridge_final.predict(test_master[TABULAR_FEATURES])
        validation_keys = validation[["Date", "STN_ID"]].copy()
        test_keys = test[["Date", "STN_ID"]].copy()
        test_missing_map = sequence_keys.iloc[test_indices][["Date", "STN_ID"]].copy()
        test_missing_map["sequence_missing_cells"] = missing_counts[test_indices]
        test_missing = test_keys.merge(test_missing_map, on=["Date", "STN_ID"], how="left")["sequence_missing_cells"].to_numpy()

        deployment_sequence_mask = (sequence_keys["year"] <= args.test_year).to_numpy() & np.isfinite(sequence_y)
        lstm_deployment, deployment_norm, _ = train_lstm_fixed(
            x_raw[deployment_sequence_mask], static_raw[deployment_sequence_mask], sequence_y[deployment_sequence_mask],
            lstm_epoch, args,
        )
        save_lstm(deployment_dir / "lstm_TA.pt", lstm_deployment, deployment_norm, lstm_epoch, args)
        cat_deployment = train_catboost_fixed(deployment_train, cat_iterations, args)
        cat_deployment.save_model(deployment_dir / "catboost_TA.cbm")
        exact_training_completed = True

    cat_weight, tabular_validation_rmse = optimize_weight(
        actual_validation, cat_validation, ridge_validation, args.weight_step
    )
    tabular_validation = cat_weight * cat_validation + (1.0 - cat_weight) * ridge_validation
    tabular_test = cat_weight * cat_test + (1.0 - cat_weight) * ridge_test
    lstm_weight, final_validation_rmse = optimize_weight(
        actual_validation, lstm_validation, tabular_validation, args.weight_step
    )
    final_validation = lstm_weight * lstm_validation + (1.0 - lstm_weight) * tabular_validation
    final_test = lstm_weight * lstm_test + (1.0 - lstm_weight) * tabular_test

    metric_rows = []
    for split, actual, predictions in [
        (
            f"validation_{args.validation_year}", actual_validation,
            {"LSTM": lstm_validation, "CatBoost": cat_validation, "Ridge": ridge_validation, "TabularBlend": tabular_validation, "FinalBlend": final_validation},
        ),
        (
            f"test_{args.test_year}", actual_test,
            {"LSTM": lstm_test, "CatBoost": cat_test, "Ridge": ridge_test, "TabularBlend": tabular_test, "FinalBlend": final_test},
        ),
    ]:
        for model_name, prediction in predictions.items():
            metric_rows.append({"split": split, "model": model_name, **diagnostic_metrics(actual, prediction)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    weights = pd.DataFrame(
        [
            {"stage": "tabular", "first_component": "CatBoost", "first_weight": cat_weight, "second_component": "Ridge", "second_weight": 1.0 - cat_weight, "validation_RMSE": tabular_validation_rmse},
            {"stage": "final", "first_component": "LSTM", "first_weight": lstm_weight, "second_component": "TabularBlend", "second_weight": 1.0 - lstm_weight, "validation_RMSE": final_validation_rmse},
        ]
    )
    weights.to_csv(output_dir / "ensemble_weights.csv", index=False)

    prediction_table = test_keys.copy()
    prediction_table["actual_TA"] = actual_test
    prediction_table["lstm_TA"] = lstm_test
    prediction_table["catboost_TA"] = cat_test
    prediction_table["ridge_TA"] = ridge_test
    prediction_table["tabular_blend_TA"] = tabular_test
    prediction_table["final_blend_TA"] = final_test
    prediction_table["sequence_missing_cells"] = test_missing
    prediction_table["sequence_complete"] = np.asarray(test_missing) == 0
    prediction_table.to_csv(output_dir / f"predictions_{args.test_year}.csv", index=False)

    ridge_coefficients = pd.DataFrame(
        {"feature": TABULAR_FEATURES, "standardized_coefficient": ridge_final.named_steps["ridge"].coef_}
    )
    ridge_coefficients["abs_coefficient"] = ridge_coefficients["standardized_coefficient"].abs()
    ridge_coefficients = ridge_coefficients.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    ridge_coefficients["rank"] = np.arange(1, len(ridge_coefficients) + 1)
    ridge_coefficients.to_csv(output_dir / "ridge_feature_importance.csv", index=False)

    if exact_training_completed:
        importance = pd.DataFrame(
            {"feature": TABULAR_FEATURES, "importance": cat_final.get_feature_importance(type="PredictionValuesChange")}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        importance["rank"] = np.arange(1, len(importance) + 1)
        importance.to_csv(output_dir / "catboost_feature_importance.csv", index=False)

    (evaluation_dir / "tabular_feature_stats.json").write_text(
        json.dumps(final_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (deployment_dir / "tabular_feature_stats.json").write_text(
        json.dumps(deployment_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    weights.to_csv(evaluation_dir / "ensemble_weights.csv", index=False)
    weights.to_csv(deployment_dir / "ensemble_weights.csv", index=False)
    feature_manifest().to_csv(evaluation_dir / "branch_feature_manifest.csv", index=False)
    feature_manifest().to_csv(deployment_dir / "branch_feature_manifest.csv", index=False)

    summary = {
        "mode": "exact" if exact_training_completed else "reuse_compatibility_diagnostic",
        "feature_counts": {"LSTM": 40, "Tabular": 40},
        "split": {
            "development_train": f"2019-{args.train_end_year}",
            "weight_selection": args.validation_year,
            "evaluation_refit": f"2019-{args.validation_year}",
            "evaluation_test": args.test_year,
            "deployment_refit": f"2019-{args.test_year}",
        },
        "counts": {
            "master_rows": int(len(master_solar)),
            "shortterm_sequences": int(len(sequence_keys)),
            "validation_common": int(len(actual_validation)),
            "test_common": int(len(actual_test)),
        },
        "selected": {
            "ridge_alpha": ridge_alpha,
            "ridge_validation_RMSE_all_rows": ridge_validation_rmse,
            "catboost_iterations": cat_iterations,
            "lstm_epoch": lstm_epoch,
            "catboost_weight_inside_tabular": cat_weight,
            "lstm_weight_inside_final": lstm_weight,
        },
        "metrics": metrics.to_dict(orient="records"),
        "limitations": (
            ["Reused LSTM/CatBoost predictions come from the earlier baseline feature definitions; run exact mode in Colab for the new branch-specific 40-feature models."]
            if not exact_training_completed else []
        ),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n[METRICS]")
    print(metrics.to_string(index=False))
    print("\n[WEIGHTS]")
    print(weights.to_string(index=False))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
