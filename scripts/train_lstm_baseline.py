#!/usr/bin/env python3
"""Train a dependency-light LSTM baseline on the 2019-2025 master CSV.

The script deliberately does not create extra weather features or impute values.
It uses only the existing 16 satellite channels plus LAT/LON/ALT, groups rows by
station, orders them by Date, and forms fixed-length observation sequences.

NumPy is used so the exact same code runs in this workspace and in Google Colab.
Input/target standardization is stored inside the model artifact and is used only
for numerical stability; predictions and reported metrics are converted back to
the original TA (deg C) and HM (%) units.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CHANNELS = [
    "IR087",
    "IR096",
    "IR105",
    "IR112",
    "IR123",
    "IR133",
    "NR013",
    "NR016",
    "SW038",
    "VI004",
    "VI005",
    "VI006",
    "VI008",
    "WV063",
    "WV069",
    "WV073",
]
FEATURES = CHANNELS + ["LAT", "LON", "ALT"]
TARGETS = ["TA", "HM"]
COMPETITION_HM_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_input(frame: pd.DataFrame) -> None:
    required = ["Date", "STN_ID", *FEATURES, *TARGETS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    duplicate_count = int(frame.duplicated(["Date", "STN_ID"]).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate Date/STN_ID rows: {duplicate_count}")
    feature_missing = frame[FEATURES].isna().sum()
    if int(feature_missing.sum()):
        details = feature_missing[feature_missing > 0].to_dict()
        raise ValueError(
            "Feature missing values were found. No imputation is performed: "
            f"{details}"
        )


def build_sequences(
    frame: pd.DataFrame, sequence_length: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    keys: list[tuple[int, int, int]] = []

    ordered = frame.copy()
    ordered["Date"] = pd.to_numeric(ordered["Date"], errors="raise").astype("int64")
    ordered["STN_ID"] = pd.to_numeric(ordered["STN_ID"], errors="raise").astype("int64")
    ordered = ordered.sort_values(["STN_ID", "Date"]).reset_index(drop=True)

    for station_id, station in ordered.groupby("STN_ID", sort=True):
        station = station.sort_values("Date").reset_index(drop=True)
        feature_values = station[FEATURES].to_numpy(dtype=np.float32)
        target_values = station[TARGETS].to_numpy(dtype=np.float32)
        dates = station["Date"].to_numpy(dtype=np.int64)

        for end in range(sequence_length - 1, len(station)):
            current_target = target_values[end]
            if np.isnan(current_target).any():
                continue
            start = end - sequence_length + 1
            sequences.append(feature_values[start : end + 1])
            targets.append(current_target)
            keys.append((int(dates[end]), int(station_id), int(dates[end] // 10000)))

    if not sequences:
        raise ValueError("No valid sequences could be created")
    return (
        np.stack(sequences).astype(np.float32, copy=False),
        np.stack(targets).astype(np.float32, copy=False),
        pd.DataFrame(keys, columns=["Date", "STN_ID", "year"]),
    )


def standardize(
    train_x: np.ndarray,
    train_y: np.ndarray,
    arrays_x: list[np.ndarray],
    arrays_y: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, np.ndarray]]:
    flat_train_x = train_x.reshape(-1, train_x.shape[-1])
    x_mean = flat_train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = flat_train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    y_mean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_std = train_y.std(axis=0, dtype=np.float64).astype(np.float32)
    y_std[y_std < 1e-6] = 1.0
    scaled_x = [((array - x_mean) / x_std).astype(np.float32) for array in arrays_x]
    scaled_y = [((array - y_mean) / y_std).astype(np.float32) for array in arrays_y]
    return scaled_x, scaled_y, {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


class NumpyLSTMRegressor:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int):
        rng = np.random.default_rng(seed)
        fan_in = input_size + hidden_size
        limit = math.sqrt(6.0 / (fan_in + 4 * hidden_size))
        self.hidden_size = hidden_size
        self.params = {
            "W": rng.uniform(-limit, limit, size=(fan_in, 4 * hidden_size)).astype(np.float32),
            "b": np.zeros(4 * hidden_size, dtype=np.float32),
            "Wy": rng.normal(0.0, 1.0 / math.sqrt(hidden_size), size=(hidden_size, output_size)).astype(np.float32),
            "by": np.zeros(output_size, dtype=np.float32),
        }
        self.params["b"][hidden_size : 2 * hidden_size] = 1.0
        self.adam_m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.adam_v = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.adam_step = 0

    def forward(self, x: np.ndarray, *, keep_cache: bool):
        batch_size = x.shape[0]
        hidden = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        cell = np.zeros_like(hidden)
        caches = [] if keep_cache else None
        h = self.hidden_size
        for step in range(x.shape[1]):
            previous_cell = cell
            joined = np.concatenate([x[:, step, :], hidden], axis=1)
            gates = joined @ self.params["W"] + self.params["b"]
            input_gate = sigmoid(gates[:, :h])
            forget_gate = sigmoid(gates[:, h : 2 * h])
            candidate = np.tanh(gates[:, 2 * h : 3 * h])
            output_gate = sigmoid(gates[:, 3 * h :])
            cell = forget_gate * previous_cell + input_gate * candidate
            tanh_cell = np.tanh(cell)
            hidden = output_gate * tanh_cell
            if caches is not None:
                caches.append((joined, input_gate, forget_gate, candidate, output_gate, previous_cell, cell, tanh_cell))
        return hidden @ self.params["Wy"] + self.params["by"], caches

    def loss_and_gradients(self, x: np.ndarray, target: np.ndarray, gradient_clip: float):
        prediction, caches = self.forward(x, keep_cache=True)
        assert caches is not None
        difference = prediction - target
        loss = float(np.mean(difference * difference))
        prediction_gradient = (2.0 / difference.size) * difference
        gradients = {name: np.zeros_like(value) for name, value in self.params.items()}
        final_hidden = caches[-1][4] * caches[-1][7]
        gradients["Wy"] = final_hidden.T @ prediction_gradient
        gradients["by"] = prediction_gradient.sum(axis=0)
        hidden_gradient = prediction_gradient @ self.params["Wy"].T
        cell_gradient = np.zeros_like(hidden_gradient)
        h = self.hidden_size
        for cache in reversed(caches):
            joined, input_gate, forget_gate, candidate, output_gate, previous_cell, _cell, tanh_cell = cache
            output_gradient = hidden_gradient * tanh_cell
            cell_total_gradient = cell_gradient + hidden_gradient * output_gate * (1.0 - tanh_cell * tanh_cell)
            forget_gradient = cell_total_gradient * previous_cell
            input_gradient = cell_total_gradient * candidate
            candidate_gradient = cell_total_gradient * input_gate
            cell_gradient = cell_total_gradient * forget_gate
            gate_gradient = np.concatenate([
                input_gradient * input_gate * (1.0 - input_gate),
                forget_gradient * forget_gate * (1.0 - forget_gate),
                candidate_gradient * (1.0 - candidate * candidate),
                output_gradient * output_gate * (1.0 - output_gate),
            ], axis=1)
            gradients["W"] += joined.T @ gate_gradient
            gradients["b"] += gate_gradient.sum(axis=0)
            joined_gradient = gate_gradient @ self.params["W"].T
            hidden_gradient = joined_gradient[:, -h:]
        global_norm = math.sqrt(sum(float(np.sum(value.astype(np.float64) ** 2)) for value in gradients.values()))
        if global_norm > gradient_clip:
            scale = gradient_clip / (global_norm + 1e-12)
            gradients = {name: value * scale for name, value in gradients.items()}
        return loss, gradients

    def adam_update(self, gradients, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.adam_step += 1
        for name, gradient in gradients.items():
            self.adam_m[name] = beta1 * self.adam_m[name] + (1.0 - beta1) * gradient
            self.adam_v[name] = beta2 * self.adam_v[name] + (1.0 - beta2) * (gradient * gradient)
            corrected_m = self.adam_m[name] / (1.0 - beta1**self.adam_step)
            corrected_v = self.adam_v[name] / (1.0 - beta2**self.adam_step)
            self.params[name] -= learning_rate * corrected_m / (np.sqrt(corrected_v) + epsilon)

    def predict(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        predictions = []
        for start in range(0, len(x), batch_size):
            batch_prediction, _ = self.forward(x[start : start + batch_size], keep_cache=False)
            predictions.append(batch_prediction)
        return np.concatenate(predictions, axis=0)


def evaluate(actual: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result = {}
    for index, target in enumerate(TARGETS):
        errors = prediction[:, index] - actual[:, index]
        denominator = float(np.sum((actual[:, index] - actual[:, index].mean()) ** 2))
        result[target] = {
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors * errors))),
            "R2": float(1.0 - np.sum(errors * errors) / denominator),
        }
    result["competition_score"] = float(result["TA"]["RMSE"] + COMPETITION_HM_WEIGHT * result["HM"]["RMSE"])
    result["mean_target_RMSE"] = float(np.mean([result[target]["RMSE"] for target in TARGETS]))
    return result


def save_model(path: Path, model: NumpyLSTMRegressor, normalization, args) -> None:
    np.savez_compressed(
        path,
        **model.params,
        **normalization,
        feature_names=np.asarray(FEATURES, dtype="U16"),
        target_names=np.asarray(TARGETS, dtype="U8"),
        sequence_length=np.asarray(args.sequence_length, dtype=np.int32),
        hidden_size=np.asarray(args.hidden_size, dtype=np.int32),
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    input_path = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    validate_input(frame)
    x_all, y_all, keys = build_sequences(frame, args.sequence_length)

    train_mask = keys["year"].to_numpy() <= args.train_end_year
    validation_mask = keys["year"].to_numpy() == args.validation_year
    test_mask = keys["year"].to_numpy() == args.test_year
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("One of train/validation/test splits is empty")

    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_validation, y_validation = x_all[validation_mask], y_all[validation_mask]
    x_test, y_test = x_all[test_mask], y_all[test_mask]
    validation_keys = keys.loc[validation_mask, ["Date", "STN_ID"]].reset_index(drop=True)
    test_keys = keys.loc[test_mask, ["Date", "STN_ID"]].reset_index(drop=True)

    scaled_x, scaled_y, normalization = standardize(
        x_train, y_train,
        [x_train, x_validation, x_test],
        [y_train, y_validation, y_test],
    )
    x_train_s, x_validation_s, x_test_s = scaled_x
    y_train_s, y_validation_s, y_test_s = scaled_y

    model = NumpyLSTMRegressor(len(FEATURES), args.hidden_size, len(TARGETS), args.seed)
    rng = np.random.default_rng(args.seed)
    history = []
    best_score = float("inf")
    best_validation_mse_scaled = float("inf")
    best_epoch = 0
    best_params = None
    stale_epochs = 0

    print(f"sequences train={len(x_train_s):,}, validation={len(x_validation_s):,}, test={len(x_test_s):,}; features={len(FEATURES)}; seq={args.sequence_length}")
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        order = rng.permutation(len(x_train_s))
        weighted_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch_indexes = order[start : start + args.batch_size]
            batch_loss, gradients = model.loss_and_gradients(x_train_s[batch_indexes], y_train_s[batch_indexes], args.gradient_clip)
            model.adam_update(gradients, args.learning_rate)
            weighted_loss += batch_loss * len(batch_indexes)
        train_loss = weighted_loss / len(order)
        validation_prediction_s = model.predict(x_validation_s, args.batch_size)
        validation_loss = float(np.mean((validation_prediction_s - y_validation_s) ** 2))
        validation_prediction_epoch = validation_prediction_s * normalization["y_std"] + normalization["y_mean"]
        validation_metrics_epoch = evaluate(y_validation, validation_prediction_epoch)
        validation_score = float(validation_metrics_epoch["competition_score"])
        row = {
            "epoch": epoch,
            "train_mse_scaled": train_loss,
            "validation_mse_scaled": validation_loss,
            "validation_RMSE_TA": validation_metrics_epoch["TA"]["RMSE"],
            "validation_RMSE_HM": validation_metrics_epoch["HM"]["RMSE"],
            "validation_competition_score": validation_score,
            "seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_mse_scaled={train_loss:.6f} valid_mse_scaled={validation_loss:.6f} "
            f"RMSE_TA={row['validation_RMSE_TA']:.4f} RMSE_HM={row['validation_RMSE_HM']:.4f} "
            f"competition={validation_score:.4f} seconds={row['seconds']:.2f}"
        )
        if validation_score < best_score - 1e-6:
            best_score = validation_score
            best_validation_mse_scaled = validation_loss
            best_epoch = epoch
            best_params = copy.deepcopy(model.params)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}; best competition score={best_score:.4f} at epoch {best_epoch}")
                break

    if best_params is None:
        raise RuntimeError("No best model was captured")
    model.params = best_params

    validation_prediction = model.predict(x_validation_s, args.batch_size) * normalization["y_std"] + normalization["y_mean"]
    test_prediction = model.predict(x_test_s, args.batch_size) * normalization["y_std"] + normalization["y_mean"]
    validation_metrics = evaluate(y_validation, validation_prediction)
    test_metrics = evaluate(y_test, test_prediction)

    def make_prediction_frame(split_keys, actual, prediction):
        frame_out = split_keys.copy()
        frame_out["TA_actual"] = actual[:, 0]
        frame_out["TA_pred"] = prediction[:, 0]
        frame_out["HM_actual"] = actual[:, 1]
        frame_out["HM_pred"] = prediction[:, 1]
        frame_out["TA_error"] = frame_out["TA_pred"] - frame_out["TA_actual"]
        frame_out["HM_error"] = frame_out["HM_pred"] - frame_out["HM_actual"]
        return frame_out

    validation_prediction_frame = make_prediction_frame(validation_keys, y_validation, validation_prediction)
    test_prediction_frame = make_prediction_frame(test_keys, y_test, test_prediction)

    history_path = output_dir / "lstm_training_history.csv"
    validation_predictions_path = output_dir / f"lstm_validation_predictions_{args.validation_year}.csv"
    predictions_path = output_dir / f"lstm_test_predictions_{args.test_year}.csv"
    model_path = output_dir / "lstm_baseline_model.npz"
    metrics_path = output_dir / "lstm_metrics.json"
    pd.DataFrame(history).to_csv(history_path, index=False)
    validation_prediction_frame.to_csv(validation_predictions_path, index=False, encoding="utf-8-sig")
    test_prediction_frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    save_model(model_path, model, normalization, args)

    metrics = {
        "model": "one-layer NumPy LSTM regressor",
        "input_csv": input_path.name,
        "no_feature_engineering": True,
        "features": FEATURES,
        "targets": TARGETS,
        "sequence_definition": "previous/current rows ordered by Date within each STN_ID",
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "train_years": f"<= {args.train_end_year}",
        "validation_year": args.validation_year,
        "test_year": args.test_year,
        "train_sequences": int(len(x_train)),
        "validation_sequences": int(len(x_validation)),
        "test_sequences": int(len(x_test)),
        "excluded_rows_with_missing_target": int(frame[TARGETS].isna().any(axis=1).sum()),
        "input_missing_imputation": False,
        "normalization": "training-set mean/std stored in model; no derived feature columns",
        "evaluation_metric": "RMSE_TA + 0.1 * RMSE_HM",
        "selection_metric": "validation competition_score (lower is better)",
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "best_validation_competition_score": float(best_score),
        "best_validation_mse_scaled": float(best_validation_mse_scaled),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "runtime_seconds": time.time() - started,
        "files": {
            "model": model_path.name,
            "history": history_path.name,
            "validation_predictions": validation_predictions_path.name,
            "test_predictions": predictions_path.name,
        },
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
