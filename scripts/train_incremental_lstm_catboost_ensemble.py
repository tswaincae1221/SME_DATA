#!/usr/bin/env python3
"""Train and analyze a short-term LSTM + 14:00 CatBoost ensemble.

The two models intentionally keep their own internally consistent inputs:

* LSTM: the new 12:00--14:00 KST, 13-step incremental long table.
* CatBoost: the existing 14:00 master table for train/validation/test.

Predictions are joined by Date/STN_ID. 2019--2023 is training, 2024 is used
for early stopping and target-specific ensemble weights, and 2025 is touched
only for the final report. Missing satellite values are imputed with training
means and accompanied by explicit missing indicators in the LSTM input.
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
STATIC_FEATURES = ["LAT", "LON", "ALT", "doy_sin", "doy_cos"]
CATBOOST_FEATURES = CHANNELS + ["LAT", "LON", "ALT", "month", "day", "dayofyear", "doy_sin", "doy_cos"]
TARGETS = ["TA", "HM"]
EXPECTED_TIMES = np.array(
    [1200, 1210, 1220, 1230, 1240, 1250, 1300, 1310, 1320, 1330, 1340, 1350, 1400],
    dtype=np.int64,
)
COMPETITION_HM_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--cat-iterations", type=int, default=1200)
    parser.add_argument("--cat-depth", type=int, default=8)
    parser.add_argument("--cat-learning-rate", type=float, default=0.03)
    parser.add_argument("--cat-l2", type=float, default=5.0)
    parser.add_argument("--cat-early-stopping-rounds", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = pd.to_numeric(out["Date"], errors="raise").round().astype("int64")
    out["STN_ID"] = pd.to_numeric(out["STN_ID"], errors="raise").astype("int64")
    date = pd.to_datetime(out["Date"].astype(str), format="%Y%m%d", errors="raise")
    out["year"] = date.dt.year.astype("int16")
    out["month"] = date.dt.month.astype("int8")
    out["day"] = date.dt.day.astype("int8")
    out["dayofyear"] = date.dt.dayofyear.astype("int16")
    angle = 2.0 * np.pi * out["dayofyear"].to_numpy(dtype=float) / 365.25
    out["doy_sin"] = np.sin(angle)
    out["doy_cos"] = np.cos(angle)
    return out


def validate_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def build_sequences(long_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    validate_columns(
        long_frame,
        ["Date", "TimeKST", "STN_ID", *CHANNELS, "LAT", "LON", "ALT", *TARGETS],
        "shortterm long CSV",
    )
    frame = add_date_features(long_frame)
    frame["TimeKST"] = pd.to_numeric(frame["TimeKST"], errors="raise").astype("int64")
    xs: list[np.ndarray] = []
    statics: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    keys: list[tuple[int, int, int]] = []
    missing_counts: list[int] = []
    bad_grids: list[tuple[int, int]] = []
    for (date, stn_id), group in frame.groupby(["Date", "STN_ID"], sort=True):
        group = group.sort_values("TimeKST")
        times = group["TimeKST"].to_numpy(dtype=np.int64)
        if len(group) != len(EXPECTED_TIMES) or not np.array_equal(times, EXPECTED_TIMES):
            bad_grids.append((int(date), int(stn_id)))
            continue
        x = group[CHANNELS].to_numpy(dtype=np.float32)
        static = group.iloc[0][STATIC_FEATURES].to_numpy(dtype=np.float32)
        label = group.loc[group["TimeKST"] == 1400, TARGETS].iloc[0].to_numpy(dtype=np.float32)
        xs.append(x)
        statics.append(static)
        ys.append(label)
        keys.append((int(date), int(stn_id), int(date) // 10000))
        missing_counts.append(int(np.isnan(x).sum()))
    if bad_grids:
        print(f"[SEQUENCE] skipped invalid time grids: {len(bad_grids)}", flush=True)
    if not xs:
        raise RuntimeError("No valid 13-step sequences were built")
    return (
        np.stack(xs),
        np.stack(statics),
        np.stack(ys),
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
    y_mean = np.nanmean(y, axis=0).astype(np.float32)
    y_std = np.nanstd(y, axis=0).astype(np.float32)
    y_std[y_std < 1e-6] = 1.0
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "static_mean": static_mean,
        "static_std": static_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def transform_lstm_inputs(
    x: np.ndarray,
    static: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    missing = np.isnan(x).astype(np.float32)
    imputed = np.where(np.isnan(x), normalization["x_mean"][None, None, :], x)
    scaled = (imputed - normalization["x_mean"][None, None, :]) / normalization["x_std"][None, None, :]
    x_with_mask = np.concatenate([scaled.astype(np.float32), missing], axis=2)
    static_imputed = np.where(np.isnan(static), normalization["static_mean"][None, :], static)
    static_scaled = (static_imputed - normalization["static_mean"][None, :]) / normalization["static_std"][None, :]
    return x_with_mask.astype(np.float32), static_scaled.astype(np.float32)


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, target in enumerate(TARGETS):
        mask = np.isfinite(actual[:, index]) & np.isfinite(prediction[:, index])
        error = prediction[mask, index] - actual[mask, index]
        result[f"{target}_n"] = int(mask.sum())
        result[f"{target}_MAE"] = float(np.mean(np.abs(error)))
        result[f"{target}_RMSE"] = float(np.sqrt(np.mean(error * error)))
    result["competition_score"] = result["TA_RMSE"] + COMPETITION_HM_WEIGHT * result["HM_RMSE"]
    return result


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(prediction)
    error = prediction[mask] - actual[mask]
    return float(np.sqrt(np.mean(error * error)))


def build_lstm_model(input_size: int, static_size: int, args: argparse.Namespace):
    import torch
    import torch.nn as nn

    class ShortTermLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = args.dropout if args.lstm_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=args.hidden_size,
                num_layers=args.lstm_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(args.hidden_size + static_size, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, len(TARGETS)),
            )

        def forward(self, dynamic, static):
            _, (hidden, _) = self.lstm(dynamic)
            return self.head(torch.cat([hidden[-1], static], dim=1))

    return ShortTermLSTM()


def predict_lstm(model, x: np.ndarray, static: np.ndarray, normalization: dict[str, np.ndarray], device) -> np.ndarray:
    import torch

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            dynamic_tensor = torch.from_numpy(x[start : start + 512]).to(device)
            static_tensor = torch.from_numpy(static[start : start + 512]).to(device)
            predictions.append(model(dynamic_tensor, static_tensor).cpu().numpy())
    scaled = np.concatenate(predictions)
    return scaled * normalization["y_std"][None, :] + normalization["y_mean"][None, :]


def train_lstm(
    x_train: np.ndarray,
    static_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    static_validation: np.ndarray,
    y_validation: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.threads))
    normalization = fit_lstm_normalization(x_train, static_train, y_train)
    x_train_scaled, static_train_scaled = transform_lstm_inputs(x_train, static_train, normalization)
    x_validation_scaled, static_validation_scaled = transform_lstm_inputs(x_validation, static_validation, normalization)
    y_train_scaled = ((y_train - normalization["y_mean"]) / normalization["y_std"]).astype(np.float32)
    model = build_lstm_model(x_train_scaled.shape[2], static_train_scaled.shape[1], args).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train_scaled),
            torch.from_numpy(static_train_scaled),
            torch.from_numpy(y_train_scaled),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = torch.nn.MSELoss()
    best_state = None
    best_score = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for dynamic_batch, static_batch, target_batch in loader:
            dynamic_batch = dynamic_batch.to(device)
            static_batch = static_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(dynamic_batch, static_batch), target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        validation_prediction = predict_lstm(
            model, x_validation_scaled, static_validation_scaled, normalization, device
        )
        validation_metrics = metrics(y_validation, validation_prediction)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            **validation_metrics,
        }
        history.append(row)
        score = validation_metrics["competition_score"]
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"[LSTM] epoch={epoch:03d} val_score={score:.5f} best={best_score:.5f}", flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("LSTM did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": x_train_scaled.shape[2],
            "static_size": static_train_scaled.shape[1],
            "hidden_size": args.hidden_size,
            "lstm_layers": args.lstm_layers,
            "dropout": args.dropout,
            "channels": CHANNELS,
            "static_features": STATIC_FEATURES,
        },
        output_dir / "lstm_model.pt",
    )
    np.savez_compressed(output_dir / "lstm_normalization.npz", **normalization)
    pd.DataFrame(history).to_csv(output_dir / "lstm_training_history.csv", index=False)
    return model, normalization, device, best_epoch


def align_master_rows(master: pd.DataFrame, keys: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    requested = keys[["Date", "STN_ID"]].reset_index().rename(columns={"index": "sequence_index"})
    joined = requested.merge(
        master[["Date", "STN_ID", *CATBOOST_FEATURES, *TARGETS]],
        on=["Date", "STN_ID"],
        how="inner",
        validate="1:1",
    )
    joined = joined.sort_values("sequence_index").reset_index(drop=True)
    return joined, joined["sequence_index"].to_numpy(dtype=int)


def train_catboost_models(
    master_train: pd.DataFrame,
    master_validation: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
):
    from catboost import CatBoostRegressor

    models = {}
    importance_parts = []
    for target in TARGETS:
        train = master_train[np.isfinite(master_train[target])]
        validation = master_validation[np.isfinite(master_validation[target])]
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
            train[CATBOOST_FEATURES],
            train[target],
            eval_set=(validation[CATBOOST_FEATURES], validation[target]),
            early_stopping_rounds=args.cat_early_stopping_rounds,
            use_best_model=True,
        )
        model.save_model(output_dir / f"catboost_{target}.cbm")
        models[target] = model
        importance_parts.append(
            pd.DataFrame(
                {
                    "feature": CATBOOST_FEATURES,
                    "target": target,
                    "importance": model.get_feature_importance(type="PredictionValuesChange"),
                    "best_iteration": model.get_best_iteration(),
                }
            )
        )
    importance = pd.concat(importance_parts, ignore_index=True)
    importance["rank"] = importance.groupby("target")["importance"].rank(method="first", ascending=False).astype(int)
    importance.sort_values(["target", "rank"]).to_csv(output_dir / "catboost_feature_importance.csv", index=False)
    return models, importance


def predict_catboost(models: dict, frame: pd.DataFrame) -> np.ndarray:
    result = np.empty((len(frame), len(TARGETS)), dtype=float)
    for target_index, target in enumerate(TARGETS):
        result[:, target_index] = models[target].predict(frame[CATBOOST_FEATURES])
    result[:, 1] = np.clip(result[:, 1], 0.0, 100.0)
    return result


def optimize_ensemble_weights(actual: np.ndarray, lstm_prediction: np.ndarray, cat_prediction: np.ndarray, step: float) -> dict[str, float]:
    if not 0 < step <= 1:
        raise ValueError("--weight-step must be in (0, 1]")
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    result: dict[str, float] = {}
    for index, target in enumerate(TARGETS):
        scores = []
        for weight in weights:
            prediction = weight * lstm_prediction[:, index] + (1.0 - weight) * cat_prediction[:, index]
            scores.append(rmse(actual[:, index], prediction))
        result[target] = float(weights[int(np.argmin(scores))])
    return result


def blend(lstm_prediction: np.ndarray, cat_prediction: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    result = np.empty_like(lstm_prediction, dtype=float)
    for index, target in enumerate(TARGETS):
        weight = weights[target]
        result[:, index] = weight * lstm_prediction[:, index] + (1.0 - weight) * cat_prediction[:, index]
    result[:, 1] = np.clip(result[:, 1], 0.0, 100.0)
    return result


def permutation_importance_lstm(
    model,
    x_scaled: np.ndarray,
    static_scaled: np.ndarray,
    actual: np.ndarray,
    normalization: dict[str, np.ndarray],
    device,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    baseline_prediction = predict_lstm(model, x_scaled, static_scaled, normalization, device)
    baseline = {target: rmse(actual[:, i], baseline_prediction[:, i]) for i, target in enumerate(TARGETS)}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for feature_index, feature in enumerate(CHANNELS):
        deltas = {target: [] for target in TARGETS}
        for _ in range(repeats):
            order = rng.permutation(len(x_scaled))
            permuted = x_scaled.copy()
            permuted[:, :, feature_index] = x_scaled[order, :, feature_index]
            permuted[:, :, len(CHANNELS) + feature_index] = x_scaled[order, :, len(CHANNELS) + feature_index]
            prediction = predict_lstm(model, permuted, static_scaled, normalization, device)
            for target_index, target in enumerate(TARGETS):
                deltas[target].append(rmse(actual[:, target_index], prediction[:, target_index]) - baseline[target])
        for target in TARGETS:
            rows.append({"feature": feature, "target": target, "delta_RMSE": float(np.mean(deltas[target])), "std": float(np.std(deltas[target]))})
    for feature_index, feature in enumerate(STATIC_FEATURES):
        deltas = {target: [] for target in TARGETS}
        for _ in range(repeats):
            order = rng.permutation(len(static_scaled))
            permuted = static_scaled.copy()
            permuted[:, feature_index] = static_scaled[order, feature_index]
            prediction = predict_lstm(model, x_scaled, permuted, normalization, device)
            for target_index, target in enumerate(TARGETS):
                deltas[target].append(rmse(actual[:, target_index], prediction[:, target_index]) - baseline[target])
        for target in TARGETS:
            rows.append({"feature": feature, "target": target, "delta_RMSE": float(np.mean(deltas[target])), "std": float(np.std(deltas[target]))})
    importance = pd.DataFrame(rows)
    importance["rank"] = importance.groupby("target")["delta_RMSE"].rank(method="first", ascending=False).astype(int)
    return importance.sort_values(["target", "rank"]).reset_index(drop=True)


def write_prediction_table(
    path: Path,
    keys: pd.DataFrame,
    actual: np.ndarray,
    lstm_prediction: np.ndarray,
    cat_prediction: np.ndarray,
    ensemble_prediction: np.ndarray,
    missing_counts: np.ndarray,
) -> pd.DataFrame:
    frame = keys[["Date", "STN_ID", "year"]].reset_index(drop=True).copy()
    frame["sequence_missing_cells"] = missing_counts
    frame["sequence_complete"] = missing_counts == 0
    for index, target in enumerate(TARGETS):
        frame[f"actual_{target}"] = actual[:, index]
        frame[f"lstm_{target}"] = lstm_prediction[:, index]
        frame[f"catboost_{target}"] = cat_prediction[:, index]
        frame[f"ensemble_{target}"] = ensemble_prediction[:, index]
    frame.to_csv(path, index=False)
    return frame


def add_metric_rows(rows: list[dict[str, object]], split: str, model_name: str, actual: np.ndarray, prediction: np.ndarray) -> None:
    rows.append({"split": split, "model": model_name, **metrics(actual, prediction)})


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    master = add_date_features(pd.read_csv(args.master_csv))
    shortterm = pd.read_csv(args.shortterm_long_csv)
    validate_columns(master, ["Date", "STN_ID", *CATBOOST_FEATURES, *TARGETS], "master CSV")
    if master.duplicated(["Date", "STN_ID"]).any():
        raise ValueError("master CSV has duplicate Date/STN_ID rows")
    x_raw, static_raw, y_all, keys, missing_counts = build_sequences(shortterm)
    print(f"[DATA] master={master.shape}, shortterm={shortterm.shape}, sequences={len(keys)}", flush=True)

    train_sequence_mask = (keys["year"] <= args.train_end_year).to_numpy() & np.isfinite(y_all).all(axis=1)
    validation_sequence_mask = (keys["year"] == args.validation_year).to_numpy()
    test_sequence_mask = (keys["year"] == args.test_year).to_numpy()
    validation_master, validation_sequence_indices = align_master_rows(
        master, keys.loc[validation_sequence_mask].reset_index(drop=True)
    )
    test_master, test_sequence_indices = align_master_rows(
        master, keys.loc[test_sequence_mask].reset_index(drop=True)
    )
    # align_master_rows indices are relative to the filtered keys, so map them back to global sequence indices.
    validation_global = np.flatnonzero(validation_sequence_mask)[validation_sequence_indices]
    test_global = np.flatnonzero(test_sequence_mask)[test_sequence_indices]
    validation_finite = np.isfinite(y_all[validation_global]).all(axis=1) & np.isfinite(validation_master[TARGETS].to_numpy()).all(axis=1)
    test_finite = np.isfinite(y_all[test_global]).all(axis=1) & np.isfinite(test_master[TARGETS].to_numpy()).all(axis=1)
    validation_global = validation_global[validation_finite]
    validation_master = validation_master.loc[validation_finite].reset_index(drop=True)
    test_global = test_global[test_finite]
    test_master = test_master.loc[test_finite].reset_index(drop=True)

    if not np.allclose(validation_master[TARGETS].to_numpy(), y_all[validation_global], equal_nan=True):
        raise ValueError("2024 master and shortterm labels do not agree")
    if not np.allclose(test_master[TARGETS].to_numpy(), y_all[test_global], equal_nan=True):
        raise ValueError("2025 master and shortterm labels do not agree")

    model, normalization, device, best_epoch = train_lstm(
        x_raw[train_sequence_mask],
        static_raw[train_sequence_mask],
        y_all[train_sequence_mask],
        x_raw[validation_global],
        static_raw[validation_global],
        y_all[validation_global],
        args,
        output_dir,
    )
    x_all_scaled, static_all_scaled = transform_lstm_inputs(x_raw, static_raw, normalization)
    lstm_validation = predict_lstm(model, x_all_scaled[validation_global], static_all_scaled[validation_global], normalization, device)
    lstm_test = predict_lstm(model, x_all_scaled[test_global], static_all_scaled[test_global], normalization, device)
    lstm_validation[:, 1] = np.clip(lstm_validation[:, 1], 0.0, 100.0)
    lstm_test[:, 1] = np.clip(lstm_test[:, 1], 0.0, 100.0)

    master_train = master[(master["year"] <= args.train_end_year) & np.isfinite(master[TARGETS]).any(axis=1)].copy()
    cat_models, cat_importance = train_catboost_models(master_train, validation_master, args, output_dir)
    cat_validation = predict_catboost(cat_models, validation_master)
    cat_test = predict_catboost(cat_models, test_master)

    weights = optimize_ensemble_weights(y_all[validation_global], lstm_validation, cat_validation, args.weight_step)
    ensemble_validation = blend(lstm_validation, cat_validation, weights)
    ensemble_test = blend(lstm_test, cat_test, weights)

    metric_rows: list[dict[str, object]] = []
    for split, actual, predictions in (
        ("validation_2024", y_all[validation_global], {"LSTM": lstm_validation, "CatBoost": cat_validation, "Ensemble": ensemble_validation}),
        ("test_2025", y_all[test_global], {"LSTM": lstm_test, "CatBoost": cat_test, "Ensemble": ensemble_test}),
    ):
        for model_name, prediction in predictions.items():
            add_metric_rows(metric_rows, split, model_name, actual, prediction)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)

    validation_keys = keys.iloc[validation_global].reset_index(drop=True)
    test_keys = keys.iloc[test_global].reset_index(drop=True)
    validation_predictions = write_prediction_table(
        output_dir / "predictions_2024.csv", validation_keys, y_all[validation_global],
        lstm_validation, cat_validation, ensemble_validation, missing_counts[validation_global],
    )
    test_predictions = write_prediction_table(
        output_dir / "predictions_2025.csv", test_keys, y_all[test_global],
        lstm_test, cat_test, ensemble_test, missing_counts[test_global],
    )

    subgroup_rows: list[dict[str, object]] = []
    for complete, group in test_predictions.groupby("sequence_complete"):
        actual = group[[f"actual_{target}" for target in TARGETS]].to_numpy()
        for model_name in ("lstm", "catboost", "ensemble"):
            prediction = group[[f"{model_name}_{target}" for target in TARGETS]].to_numpy()
            subgroup_rows.append(
                {
                    "split": "test_2025",
                    "sequence_group": "complete" if complete else "has_missing_satellite",
                    "model": model_name.title(),
                    **metrics(actual, prediction),
                }
            )
    pd.DataFrame(subgroup_rows).to_csv(output_dir / "test_missingness_subgroup_metrics.csv", index=False)

    residual_rows = []
    for split, actual, lstm_prediction, cat_prediction in (
        ("validation_2024", y_all[validation_global], lstm_validation, cat_validation),
        ("test_2025", y_all[test_global], lstm_test, cat_test),
    ):
        for index, target in enumerate(TARGETS):
            residual_rows.append(
                {
                    "split": split,
                    "target": target,
                    "lstm_catboost_residual_correlation": float(
                        np.corrcoef(actual[:, index] - lstm_prediction[:, index], actual[:, index] - cat_prediction[:, index])[0, 1]
                    ),
                }
            )
    pd.DataFrame(residual_rows).to_csv(output_dir / "residual_correlation.csv", index=False)

    lstm_importance = permutation_importance_lstm(
        model,
        x_all_scaled[validation_global],
        static_all_scaled[validation_global],
        y_all[validation_global],
        normalization,
        device,
        args.permutation_repeats,
        args.seed + 1000,
    )
    lstm_importance.to_csv(output_dir / "lstm_permutation_importance.csv", index=False)

    component_rows = []
    for target in TARGETS:
        component_rows.append(
            {
                "target": target,
                "lstm_weight": weights[target],
                "catboost_weight": 1.0 - weights[target],
                "dominant_component": "LSTM" if weights[target] > 0.5 else "CatBoost" if weights[target] < 0.5 else "Tie",
            }
        )
    pd.DataFrame(component_rows).to_csv(output_dir / "ensemble_component_weights.csv", index=False)

    source_consistency_rows = []
    shortterm_1400 = add_date_features(shortterm[shortterm["TimeKST"] == 1400])
    overlap = master[["Date", "STN_ID", *CHANNELS]].merge(
        shortterm_1400[["Date", "STN_ID", *CHANNELS]],
        on=["Date", "STN_ID"], suffixes=("_master", "_shortterm"), validate="1:1",
    )
    for year, group in overlap.groupby(overlap["Date"] // 10000):
        for channel in CHANNELS:
            left = group[f"{channel}_master"]
            right = group[f"{channel}_shortterm"]
            valid = left.notna() & right.notna()
            source_consistency_rows.append(
                {
                    "year": int(year),
                    "channel": channel,
                    "n": int(valid.sum()),
                    "exact_fraction": float((left[valid] == right[valid]).mean()),
                    "MAE": float((left[valid] - right[valid]).abs().mean()),
                    "correlation": float(left[valid].corr(right[valid])),
                }
            )
    pd.DataFrame(source_consistency_rows).to_csv(output_dir / "master_shortterm_feature_consistency.csv", index=False)

    summary = {
        "parameters": vars(args),
        "split": {
            "train_end_year": args.train_end_year,
            "validation_year": args.validation_year,
            "test_year": args.test_year,
        },
        "counts": {
            "master_train_rows": int(len(master_train)),
            "lstm_train_sequences": int(train_sequence_mask.sum()),
            "validation_common_rows": int(len(validation_global)),
            "test_common_rows": int(len(test_global)),
            "test_complete_sequences": int((missing_counts[test_global] == 0).sum()),
            "test_sequences_with_missing_satellite": int((missing_counts[test_global] > 0).sum()),
        },
        "lstm": {"best_epoch": int(best_epoch), "device": str(device)},
        "ensemble_weights": component_rows,
        "metrics": metrics_frame.to_dict(orient="records"),
        "top_catboost_features": {
            target: cat_importance[cat_importance["target"] == target].nsmallest(10, "rank")[["feature", "importance", "rank"]].to_dict(orient="records")
            for target in TARGETS
        },
        "top_lstm_features": {
            target: lstm_importance[lstm_importance["target"] == target].nsmallest(10, "rank")[["feature", "delta_RMSE", "std", "rank"]].to_dict(orient="records")
            for target in TARGETS
        },
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[FINAL METRICS]", flush=True)
    print(metrics_frame.to_string(index=False), flush=True)
    print("\n[ENSEMBLE WEIGHTS]", flush=True)
    print(pd.DataFrame(component_rows).to_string(index=False), flush=True)
    print(f"\nOutputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
