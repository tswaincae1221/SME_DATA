#!/usr/bin/env python3
"""Train a CatBoost generalist + short-term LSTM specialist ensemble baseline.

CatBoost:
- input: the existing 14:00 master tabular CSV
- train: <= 2023, using all available rows
- validation/test scoring: only Date/STN_ID keys that exist in the short-term dataset

LSTM:
- input: 12:00~14:00 KST short-term long CSV
- one sample = Date x STN_ID, 13 timesteps (10-minute interval)
- dynamic inputs: 16 GK-2A channels
- static inputs: LAT/LON/ALT
- train/validation/test = <=2023 / 2024 / 2025

Ensemble:
- optimize separate LSTM weights for TA and HM on 2024 validation
- final = w * LSTM + (1-w) * CatBoost
- report 2025 test metrics without using test labels for tuning
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CHANNELS = [
    "IR087", "IR096", "IR105", "IR112", "IR123", "IR133",
    "NR013", "NR016", "SW038", "VI004", "VI005", "VI006",
    "VI008", "WV063", "WV069", "WV073",
]
STATIC_FEATURES = ["LAT", "LON", "ALT"]
TARGETS = ["TA", "HM"]
CAT_DATE_FEATURES = ["month", "day", "dayofyear", "doy_sin", "doy_cos"]
CAT_FEATURES = CHANNELS + STATIC_FEATURES + CAT_DATE_FEATURES
HM_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master-csv", required=True)
    p.add_argument("--shortterm-long-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-end-year", type=int, default=2023)
    p.add_argument("--validation-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--cat-iterations", type=int, default=1200)
    p.add_argument("--cat-depth", type=int, default=8)
    p.add_argument("--cat-learning-rate", type=float, default=0.03)
    p.add_argument("--cat-l2", type=float, default=5.0)
    p.add_argument("--cat-early-stopping-rounds", type=int, default=120)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--weight-step", type=float, default=0.01)
    return p.parse_args()


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


def ensure_numeric_date(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = pd.to_numeric(out["Date"], errors="raise").astype("int64")
    out["STN_ID"] = pd.to_numeric(out["STN_ID"], errors="raise").astype("int64")
    out["year"] = out["Date"] // 10000
    return out


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = ensure_numeric_date(frame)
    dt = pd.to_datetime(out["Date"].astype(str), format="%Y%m%d", errors="raise")
    out["month"] = dt.dt.month.astype("int16")
    out["day"] = dt.dt.day.astype("int16")
    out["dayofyear"] = dt.dt.dayofyear.astype("int16")
    angle = 2.0 * np.pi * out["dayofyear"].to_numpy(dtype=np.float64) / 365.25
    out["doy_sin"] = np.sin(angle).astype(np.float32)
    out["doy_cos"] = np.cos(angle).astype(np.float32)
    return out


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for i, target in enumerate(TARGETS):
        err = pred[:, i] - actual[:, i]
        result[target] = {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err * err))),
        }
    result["competition_score"] = float(
        result["TA"]["RMSE"] + HM_WEIGHT * result["HM"]["RMSE"]
    )
    return result


def validate_master(frame: pd.DataFrame) -> None:
    required = ["Date", "STN_ID", *CAT_FEATURES, *TARGETS]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"master CSV missing columns: {missing}")
    duplicates = int(frame.duplicated(["Date", "STN_ID"]).sum())
    if duplicates:
        raise ValueError(f"master CSV has duplicate Date/STN_ID rows: {duplicates}")


def build_shortterm_samples(frame: pd.DataFrame):
    required = ["Date", "TimeKST", "STN_ID", *CHANNELS, *STATIC_FEATURES, *TARGETS]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"shortterm long CSV missing columns: {missing}")

    f = ensure_numeric_date(frame)
    f["TimeKST"] = pd.to_numeric(f["TimeKST"], errors="raise").astype("int64")
    expected_times = np.array(
        [1200, 1210, 1220, 1230, 1240, 1250, 1300, 1310, 1320, 1330, 1340, 1350, 1400],
        dtype=np.int64,
    )

    xs, statics, ys, keys, skipped = [], [], [], [], []
    for (date, stn), group in f.groupby(["Date", "STN_ID"], sort=True):
        g = group.sort_values("TimeKST")
        times = g["TimeKST"].to_numpy(dtype=np.int64)
        if len(g) != len(expected_times) or not np.array_equal(times, expected_times):
            skipped.append((int(date), int(stn), "incomplete_time_grid"))
            continue
        x = g[CHANNELS].to_numpy(dtype=np.float32)
        static = g.iloc[0][STATIC_FEATURES].to_numpy(dtype=np.float32)
        row1400 = g.loc[g["TimeKST"] == 1400]
        if len(row1400) != 1:
            skipped.append((int(date), int(stn), "missing_1400"))
            continue
        y = row1400.iloc[0][TARGETS].to_numpy(dtype=np.float32)
        if np.isnan(x).any():
            skipped.append((int(date), int(stn), "satellite_nan"))
            continue
        if np.isnan(static).any():
            skipped.append((int(date), int(stn), "static_nan"))
            continue
        if np.isnan(y).any():
            skipped.append((int(date), int(stn), "target_nan"))
            continue
        xs.append(x)
        statics.append(static)
        ys.append(y)
        keys.append((int(date), int(stn), int(date) // 10000))

    if not xs:
        raise ValueError("No valid short-term samples were created")
    if skipped:
        reason_counts = pd.Series([r[2] for r in skipped]).value_counts().to_dict()
        print(f"[shortterm] skipped={len(skipped)} reasons={reason_counts}", flush=True)

    return (
        np.stack(xs).astype(np.float32, copy=False),
        np.stack(statics).astype(np.float32, copy=False),
        np.stack(ys).astype(np.float32, copy=False),
        pd.DataFrame(keys, columns=["Date", "STN_ID", "year"]),
    )


def standardize_train(x_train, s_train, y_train):
    flat = x_train.reshape(-1, x_train.shape[-1])
    x_mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    s_mean = s_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    s_std = s_train.std(axis=0, dtype=np.float64).astype(np.float32)
    s_std[s_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_std = y_train.std(axis=0, dtype=np.float64).astype(np.float32)
    y_std[y_std < 1e-6] = 1.0
    return {"x_mean": x_mean, "x_std": x_std, "s_mean": s_mean, "s_std": s_std, "y_mean": y_mean, "y_std": y_std}


def scale_arrays(x, s, y, norm):
    return (
        ((x - norm["x_mean"]) / norm["x_std"]).astype(np.float32),
        ((s - norm["s_mean"]) / norm["s_std"]).astype(np.float32),
        ((y - norm["y_mean"]) / norm["y_std"]).astype(np.float32),
    )


def train_lstm(x_train, s_train, y_train, x_val, s_val, y_val, args, output_dir):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. In Colab: pip install torch") from exc

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ShortTermLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            lstm_dropout = args.dropout if args.lstm_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=len(CHANNELS), hidden_size=args.hidden_size,
                num_layers=args.lstm_layers, batch_first=True, dropout=lstm_dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(args.hidden_size + len(STATIC_FEATURES), args.hidden_size),
                nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.hidden_size, len(TARGETS)),
            )
        def forward(self, x, static):
            _, (h_n, _) = self.lstm(x)
            return self.head(torch.cat([h_n[-1], static], dim=1))

    norm = standardize_train(x_train, s_train, y_train)
    xtr, str_, ytr = scale_arrays(x_train, s_train, y_train, norm)
    xva, sva, _ = scale_arrays(x_val, s_val, y_val, norm)
    train_ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(str_), torch.from_numpy(ytr))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ShortTermLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    y_mean_t = torch.from_numpy(norm["y_mean"]).to(device)
    y_std_t = torch.from_numpy(norm["y_std"]).to(device)
    xva_t, sva_t = torch.from_numpy(xva).to(device), torch.from_numpy(sva).to(device)

    best_score, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for xb, sb, yb in train_loader:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, sb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total_n += len(xb)
        model.eval()
        with torch.no_grad():
            val_pred = (model(xva_t, sva_t) * y_std_t + y_mean_t).cpu().numpy()
        m = metrics(y_val, val_pred)
        score = float(m["competition_score"])
        train_loss = total_loss / max(total_n, 1)
        history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_TA_RMSE": m["TA"]["RMSE"], "val_HM_RMSE": m["HM"]["RMSE"], "val_score": score})
        print(f"[LSTM] epoch={epoch:03d} train_mse={train_loss:.5f} val_TA={m['TA']['RMSE']:.4f} val_HM={m['HM']['RMSE']:.4f} score={score:.4f}", flush=True)
        if score < best_score - 1e-6:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[LSTM] early stop at epoch={epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("LSTM training did not produce a best state")
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(), "channels": CHANNELS, "static_features": STATIC_FEATURES,
        "targets": TARGETS, "hidden_size": args.hidden_size, "lstm_layers": args.lstm_layers,
        "dropout": args.dropout, "normalization": {k: v.tolist() for k, v in norm.items()},
    }, output_dir / "lstm_shortterm_baseline.pt")
    hist = pd.DataFrame(history)
    hist.to_csv(output_dir / "lstm_history.csv", index=False)
    return model, norm


def lstm_predict(model, norm, x, s, args):
    import torch
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs = ((x - norm["x_mean"]) / norm["x_std"]).astype(np.float32)
    ss = ((s - norm["s_mean"]) / norm["s_std"]).astype(np.float32)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(xs), args.batch_size):
            xb = torch.from_numpy(xs[start:start + args.batch_size]).to(device)
            sb = torch.from_numpy(ss[start:start + args.batch_size]).to(device)
            preds.append(model(xb, sb).cpu().numpy())
    pred_s = np.concatenate(preds, axis=0)
    return pred_s * norm["y_std"] + norm["y_mean"]


def train_catboost(master, val_keys, test_keys, args, output_dir):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("catboost is required. In Colab: pip install catboost") from exc

    master = add_date_features(master)
    validate_master(master)
    train = master[(master["year"] <= args.train_end_year) & master[TARGETS].notna().all(axis=1) & master[CAT_FEATURES].notna().all(axis=1)].copy()

    def select_keys(keys):
        keyed = keys[["Date", "STN_ID"]].drop_duplicates()
        selected = keyed.merge(master, on=["Date", "STN_ID"], how="left", validate="1:1")
        bad = selected[CAT_FEATURES + TARGETS].isna().any(axis=1)
        if bad.any():
            raise ValueError(f"master CSV is missing usable CatBoost rows for {int(bad.sum())} short-term keys")
        return selected

    val, test = select_keys(val_keys), select_keys(test_keys)
    print(f"[CatBoost] train={len(train):,} val_week={len(val):,} test_week={len(test):,} features={len(CAT_FEATURES)}", flush=True)
    val_pred = np.zeros((len(val), 2), dtype=np.float32)
    test_pred = np.zeros((len(test), 2), dtype=np.float32)
    for i, target in enumerate(TARGETS):
        model = CatBoostRegressor(
            iterations=args.cat_iterations, depth=args.cat_depth, learning_rate=args.cat_learning_rate,
            l2_leaf_reg=args.cat_l2, loss_function="RMSE", eval_metric="RMSE",
            random_seed=args.seed, verbose=100, allow_writing_files=False,
        )
        model.fit(train[CAT_FEATURES], train[target], eval_set=(val[CAT_FEATURES], val[target]), use_best_model=True, early_stopping_rounds=args.cat_early_stopping_rounds)
        val_pred[:, i] = model.predict(val[CAT_FEATURES]).astype(np.float32)
        test_pred[:, i] = model.predict(test[CAT_FEATURES]).astype(np.float32)
        model.save_model(str(output_dir / f"catboost_{target.lower()}.cbm"))
    val_out = val[["Date", "STN_ID"]].copy()
    test_out = test[["Date", "STN_ID"]].copy()
    for i, target in enumerate(TARGETS):
        val_out[f"cat_{target}"] = val_pred[:, i]
        test_out[f"cat_{target}"] = test_pred[:, i]
    return val_out, test_out


def optimize_weights(actual, cat_pred, lstm_pred, step):
    if not (0 < step <= 1):
        raise ValueError("--weight-step must be in (0, 1]")
    grid = np.arange(0.0, 1.0 + step / 2.0, step)
    out = {}
    for i, target in enumerate(TARGETS):
        best_w, best_rmse = 0.0, float("inf")
        for w in grid:
            pred = w * lstm_pred[:, i] + (1.0 - w) * cat_pred[:, i]
            rmse = float(np.sqrt(np.mean((pred - actual[:, i]) ** 2)))
            if rmse < best_rmse:
                best_w, best_rmse = float(w), rmse
        out[target] = best_w
    return out


def blend(cat_pred, lstm_pred, weights):
    out = np.zeros_like(cat_pred, dtype=np.float32)
    for i, target in enumerate(TARGETS):
        w = weights[target]
        out[:, i] = w * lstm_pred[:, i] + (1.0 - w) * cat_pred[:, i]
    return out


def residual_corr(actual, a, b):
    out = {}
    for i, target in enumerate(TARGETS):
        ea, eb = actual[:, i] - a[:, i], actual[:, i] - b[:, i]
        out[target] = float(np.corrcoef(ea, eb)[0, 1])
    return out


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(Path(args.master_csv).expanduser())
    short = pd.read_csv(Path(args.shortterm_long_csv).expanduser())
    x_all, s_all, y_all, keys = build_shortterm_samples(short)
    train_mask = keys["year"].to_numpy() <= args.train_end_year
    val_mask = keys["year"].to_numpy() == args.validation_year
    test_mask = keys["year"].to_numpy() == args.test_year
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(f"empty split: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")

    x_train, s_train, y_train = x_all[train_mask], s_all[train_mask], y_all[train_mask]
    x_val, s_val, y_val = x_all[val_mask], s_all[val_mask], y_all[val_mask]
    x_test, s_test, y_test = x_all[test_mask], s_all[test_mask], y_all[test_mask]
    val_keys = keys.loc[val_mask, ["Date", "STN_ID"]].reset_index(drop=True)
    test_keys = keys.loc[test_mask, ["Date", "STN_ID"]].reset_index(drop=True)
    print(f"[LSTM data] train={len(x_train):,} val={len(x_val):,} test={len(x_test):,} shape={x_train.shape[1:]}", flush=True)

    lstm_model, norm = train_lstm(x_train, s_train, y_train, x_val, s_val, y_val, args, output_dir)
    lstm_val = lstm_predict(lstm_model, norm, x_val, s_val, args)
    lstm_test = lstm_predict(lstm_model, norm, x_test, s_test, args)
    cat_val_df, cat_test_df = train_catboost(master, val_keys, test_keys, args, output_dir)

    val_join = val_keys.copy(); val_join[["TA", "HM"]] = y_val
    test_join = test_keys.copy(); test_join[["TA", "HM"]] = y_test
    val_join = val_join.merge(cat_val_df, on=["Date", "STN_ID"], how="left", validate="1:1")
    test_join = test_join.merge(cat_test_df, on=["Date", "STN_ID"], how="left", validate="1:1")
    if val_join[["cat_TA", "cat_HM"]].isna().any().any() or test_join[["cat_TA", "cat_HM"]].isna().any().any():
        raise ValueError("CatBoost predictions failed to align with LSTM keys")

    cat_val = val_join[["cat_TA", "cat_HM"]].to_numpy(dtype=np.float32)
    cat_test = test_join[["cat_TA", "cat_HM"]].to_numpy(dtype=np.float32)
    weights = optimize_weights(y_val, cat_val, lstm_val, args.weight_step)
    ensemble_val = blend(cat_val, lstm_val, weights)
    ensemble_test = blend(cat_test, lstm_test, weights)

    result = {
        "config": vars(args),
        "sample_counts": {"lstm_train": int(len(x_train)), "validation": int(len(x_val)), "test": int(len(x_test))},
        "weights_lstm": weights,
        "validation": {
            "catboost": metrics(y_val, cat_val), "lstm": metrics(y_val, lstm_val), "ensemble": metrics(y_val, ensemble_val),
            "cat_lstm_residual_correlation": residual_corr(y_val, cat_val, lstm_val),
        },
        "test": {
            "catboost": metrics(y_test, cat_test), "lstm": metrics(y_test, lstm_test), "ensemble": metrics(y_test, ensemble_test),
            "cat_lstm_residual_correlation": residual_corr(y_test, cat_test, lstm_test),
        },
    }

    def prediction_frame(keys_df, actual, cat, lstm, ens):
        out = keys_df.copy(); out["TA"] = actual[:, 0]; out["HM"] = actual[:, 1]
        for i, target in enumerate(TARGETS):
            out[f"cat_{target}"] = cat[:, i]; out[f"lstm_{target}"] = lstm[:, i]; out[f"ensemble_{target}"] = ens[:, i]
        return out

    prediction_frame(val_keys, y_val, cat_val, lstm_val, ensemble_val).to_csv(output_dir / "validation_predictions.csv", index=False)
    prediction_frame(test_keys, y_test, cat_test, lstm_test, ensemble_test).to_csv(output_dir / "test_predictions.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(output_dir / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)

    rows = []
    for split in ["validation", "test"]:
        for model_name in ["catboost", "lstm", "ensemble"]:
            m = result[split][model_name]
            rows.append({"split": split, "model": model_name, "TA_RMSE": m["TA"]["RMSE"], "HM_RMSE": m["HM"]["RMSE"], "competition_score": m["competition_score"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "comparison.csv", index=False)
    print("\n=== ENSEMBLE WEIGHTS (LSTM share, tuned on validation) ===")
    print(json.dumps(weights, indent=2))
    print("\n=== COMPARISON ===")
    print(summary.to_string(index=False))
    print(f"\noutputs -> {output_dir}")


if __name__ == "__main__":
    main()
