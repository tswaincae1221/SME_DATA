#!/usr/bin/env python3
"""Cumulative five-step feature deduplication ablation for the TA/HM ensemble.

Each effective variant is evaluated with rolling-year OOF folds (2022-2024),
then refit through 2024 and evaluated on 2025.  The model is the current best
architecture: CatBoost plus a residual LSTM using 12:00-14:00 KST GK-2A
sequences.  TA receives the same pooled + latest-year OOF offset calibration;
HM is clipped to [0, 100].

The first requested removal concerns Ridge pair features.  Ridge is not used
by the current best ensemble, so step 1 is recorded as an intentional no-op
and reuses the baseline result exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_dualbranch_ta_baseline as base  # noqa: E402


KEYS = ["Date", "STN_ID"]
CALENDAR_SOLAR_REMOVALS = [
    "month", "dayofyear", "solar_azimuth", "solar_declination",
    "local_solar_time", "equation_of_time",
]
PAIR_KEEP = [
    "fog_window_038_112",
    "cloud_phase_087_112",
    "ozone_window_096_112",
    "split_window_112_123",
    "wv_vertical_063_073",
]


@dataclass(frozen=True)
class Variant:
    step: int
    name: str
    removal: str
    tabular_features: tuple[str, ...]
    lstm_static_features: tuple[str, ...]
    effective: bool = True


def build_variants() -> list[Variant]:
    full_tabular = tuple(base.TABULAR_FEATURES)
    full_static = tuple(base.LSTM_STATIC_FEATURES)
    calendar_dedup = tuple(feature for feature in full_tabular if feature not in CALENDAR_SOLAR_REMOVALS)
    no_missing_fraction = tuple(feature for feature in full_static if feature != "sequence_missing_fraction")
    reduced_static = tuple(
        feature for feature in no_missing_fraction
        if feature not in {"doy_sin", "doy_cos", "solar_azimuth"}
    )
    pair_dedup = tuple(
        feature for feature in calendar_dedup
        if feature not in base.PAIR_FEATURES or feature in PAIR_KEEP
    )
    return [
        Variant(0, "step0_current_baseline", "none", full_tabular, full_static),
        Variant(
            1, "step1_drop_ridge_pair_features",
            "Ridge pair features removed; no effect because Ridge is absent from the best ensemble",
            full_tabular, full_static, effective=False,
        ),
        Variant(
            2, "step2_drop_redundant_calendar_solar",
            ",".join(CALENDAR_SOLAR_REMOVALS), calendar_dedup, full_static,
        ),
        Variant(
            3, "step3_drop_lstm_missing_fraction",
            "sequence_missing_fraction", calendar_dedup, no_missing_fraction,
        ),
        Variant(
            4, "step4_drop_lstm_seasonal_azimuth",
            "doy_sin,doy_cos,solar_azimuth from LSTM static inputs",
            calendar_dedup, reduced_static,
        ),
        Variant(
            5, "step5_reduce_catboost_pair_features",
            "CatBoost standardized pairs reduced from 11 to 5",
            pair_dedup, reduced_static,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oof-years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument(
        "--steps", nargs="+", type=int,
        help="Optional subset of effective step numbers, e.g. --steps 0 3 for a multi-seed confirmation run",
    )
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
    parser.add_argument("--residual-scale-step", type=float, default=0.02)
    parser.add_argument("--residual-scale-max", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_device(args):
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.threads))
    return device


def build_lstm(args, static_dim: int):
    import torch.nn as nn

    class ResidualLSTM(nn.Module):
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
                nn.Linear(args.hidden_size + static_dim, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, 1),
            )

        def forward(self, dynamic, static):
            _, (hidden, _) = self.lstm(dynamic)
            return self.head(__import__("torch").cat([hidden[-1], static], dim=1)).squeeze(1)

    return ResidualLSTM()


def predict_lstm(model, normalization, device, x_raw, static_raw) -> np.ndarray:
    import torch

    dynamic, static = base.transform_lstm_inputs(x_raw, static_raw, normalization)
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(dynamic), 512):
            d = torch.from_numpy(dynamic[start:start + 512]).to(device)
            s = torch.from_numpy(static[start:start + 512]).to(device)
            parts.append(model(d, s).cpu().numpy())
    scaled = np.concatenate(parts)
    return scaled * normalization["y_std"][0] + normalization["y_mean"][0]


def train_lstm_dev(x_train, static_train, y_train, x_val, static_val, y_val, args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch_device(args)
    normalization = base.fit_lstm_normalization(x_train, static_train, y_train)
    train_dynamic, train_static = base.transform_lstm_inputs(x_train, static_train, normalization)
    target = ((y_train - normalization["y_mean"][0]) / normalization["y_std"][0]).astype(np.float32)
    model = build_lstm(args, static_train.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_dynamic), torch.from_numpy(train_static), torch.from_numpy(target)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = torch.nn.MSELoss()
    best_state = None
    best_epoch = 1
    best_score = math.inf
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for dynamic, static, batch_target in loader:
            dynamic, static, batch_target = dynamic.to(device), static.to(device), batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(dynamic, static), batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        prediction = predict_lstm(model, normalization, device, x_val, static_val)
        score = base.rmse(y_val, prediction)
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
        raise RuntimeError("LSTM checkpoint selection failed")
    model.load_state_dict(best_state)
    return model, normalization, device, best_epoch, best_score


def train_lstm_fixed(x, static, y, epochs: int, args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch_device(args)
    normalization = base.fit_lstm_normalization(x, static, y)
    dynamic, static_scaled = base.transform_lstm_inputs(x, static, normalization)
    target = ((y - normalization["y_mean"][0]) / normalization["y_std"][0]).astype(np.float32)
    model = build_lstm(args, static.shape[1]).to(device)
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


def train_catboost_dev(train, validation, features, target, args):
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
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        train[list(features)], train[target],
        eval_set=(validation[list(features)], validation[target]),
        early_stopping_rounds=args.cat_early_stopping_rounds,
        use_best_model=True,
    )
    return model, max(1, int(model.get_best_iteration()) + 1)


def train_catboost_fixed(train, features, target, iterations, args):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=max(1, int(iterations)),
        depth=args.cat_depth,
        learning_rate=args.cat_learning_rate,
        l2_leaf_reg=args.cat_l2,
        loss_function="RMSE",
        random_seed=args.seed,
        thread_count=max(1, args.threads),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train[list(features)], train[target])
    return model


def align_master(engineered, keys, features, target):
    aligned = keys[KEYS].merge(
        engineered[[*KEYS, target, *features]], on=KEYS, how="inner", validate="1:1"
    )
    if len(aligned) != len(keys):
        raise ValueError(f"master/sequence mismatch for {target}: {len(keys)} vs {len(aligned)}")
    return aligned


def available_mask(engineered, sequence_keys, target):
    available = engineered.loc[engineered[target].notna(), KEYS].drop_duplicates().assign(ok=True)
    return sequence_keys[KEYS].merge(
        available, on=KEYS, how="left", validate="m:1"
    ).ok.eq(True).to_numpy(bool)


def sequence_labels(shortterm, sequence_keys, target):
    labels = shortterm.loc[pd.to_numeric(shortterm.TimeKST).eq(1400), [*KEYS, target]].copy()
    labels["Date"] = pd.to_numeric(labels.Date).round().astype("int64")
    labels["STN_ID"] = pd.to_numeric(labels.STN_ID).round().astype("int64")
    if labels.duplicated(KEYS).any():
        raise ValueError(f"duplicate 14:00 {target} labels")
    return sequence_keys[KEYS].merge(labels, on=KEYS, how="left", validate="1:1")[target].to_numpy(np.float32)


def one_fold(
    variant, target, year, master_solar, x_raw, static_all, labels,
    sequence_keys, missing_counts, args,
):
    target_offset = 0 if target == "TA" else 10000
    fold_args = copy.copy(args)
    fold_args.seed = args.seed + target_offset + year
    seed_everything(fold_args.seed)
    stats = base.fit_channel_stats(master_solar, year - 1)
    engineered = base.engineer_tabular(master_solar, stats)
    features = variant.tabular_features
    master_train = engineered[(engineered.year <= year - 1) & engineered[target].notna()]
    master_val = engineered[(engineered.year == year) & engineered[target].notna()]
    cat, iterations = train_catboost_dev(master_train, master_val, features, target, fold_args)

    available = available_mask(engineered, sequence_keys, target)
    years = sequence_keys.year.to_numpy(int)
    train_idx = np.flatnonzero((years <= year - 1) & np.isfinite(labels) & available)
    val_idx = np.flatnonzero((years == year) & np.isfinite(labels) & available)
    train_keys = sequence_keys.iloc[train_idx].reset_index(drop=True)
    val_keys = sequence_keys.iloc[val_idx].reset_index(drop=True)
    train_aligned = align_master(engineered, train_keys, features, target)
    val_aligned = align_master(engineered, val_keys, features, target)
    cat_train = cat.predict(train_aligned[list(features)])
    cat_val = cat.predict(val_aligned[list(features)])
    residual_train = labels[train_idx] - cat_train
    residual_val_target = labels[val_idx] - cat_val
    static_indices = [base.LSTM_STATIC_FEATURES.index(name) for name in variant.lstm_static_features]
    static = static_all[:, static_indices]
    model, normalization, device, epoch, residual_rmse = train_lstm_dev(
        x_raw[train_idx], static[train_idx], residual_train,
        x_raw[val_idx], static[val_idx], residual_val_target, fold_args,
    )
    residual_val = predict_lstm(model, normalization, device, x_raw[val_idx], static[val_idx])
    frame = val_keys.copy()
    frame[f"actual_{target}"] = labels[val_idx]
    frame[f"catboost_{target}"] = cat_val
    frame[f"residual_lstm_{target}"] = residual_val
    frame["sequence_missing_cells"] = missing_counts[val_idx]
    summary = {
        "variant": variant.name,
        "target": target,
        "fold_year": year,
        "train_sequences": len(train_idx),
        "validation_sequences": len(val_idx),
        "cat_iterations": iterations,
        "residual_lstm_epoch": epoch,
        "cat_RMSE": base.rmse(labels[val_idx], cat_val),
        "residual_target_RMSE": residual_rmse,
        "scale1_RMSE": base.rmse(labels[val_idx], cat_val + residual_val),
    }
    return frame, summary


def final_predictions(
    variant, target, master_solar, x_raw, static_all, labels,
    sequence_keys, missing_counts, fold_summary, args,
):
    train_end = args.test_year - 1
    target_offset = 0 if target == "TA" else 10000
    final_args = copy.copy(args)
    final_args.seed = args.seed + target_offset + args.test_year
    seed_everything(final_args.seed)
    stats = base.fit_channel_stats(master_solar, train_end)
    engineered = base.engineer_tabular(master_solar, stats)
    features = variant.tabular_features
    master_train = engineered[(engineered.year <= train_end) & engineered[target].notna()]
    iterations = max(1, int(round(fold_summary.cat_iterations.median())))
    epoch = max(1, int(round(fold_summary.residual_lstm_epoch.median())))
    cat = train_catboost_fixed(master_train, features, target, iterations, final_args)
    available = available_mask(engineered, sequence_keys, target)
    years = sequence_keys.year.to_numpy(int)
    train_idx = np.flatnonzero((years <= train_end) & np.isfinite(labels) & available)
    test_idx = np.flatnonzero((years == args.test_year) & np.isfinite(labels) & available)
    train_keys = sequence_keys.iloc[train_idx].reset_index(drop=True)
    test_keys = sequence_keys.iloc[test_idx].reset_index(drop=True)
    train_aligned = align_master(engineered, train_keys, features, target)
    test_aligned = align_master(engineered, test_keys, features, target)
    cat_train = cat.predict(train_aligned[list(features)])
    cat_test = cat.predict(test_aligned[list(features)])
    residual_train = labels[train_idx] - cat_train
    static_indices = [base.LSTM_STATIC_FEATURES.index(name) for name in variant.lstm_static_features]
    static = static_all[:, static_indices]
    model, normalization, device = train_lstm_fixed(
        x_raw[train_idx], static[train_idx], residual_train, epoch, final_args,
    )
    residual_test = predict_lstm(model, normalization, device, x_raw[test_idx], static[test_idx])
    frame = test_keys.copy()
    frame[f"actual_{target}"] = labels[test_idx]
    frame[f"catboost_{target}"] = cat_test
    frame[f"residual_lstm_{target}"] = residual_test
    frame["sequence_missing_cells"] = missing_counts[test_idx]
    return frame


def optimize_scale(actual, cat, residual, args):
    scales = np.arange(0.0, args.residual_scale_max + args.residual_scale_step / 2, args.residual_scale_step)
    scores = [base.rmse(actual, cat + scale * residual) for scale in scales]
    index = int(np.argmin(scores))
    return float(scales[index])


def score_target(oof, test, target, args):
    actual_oof = oof[f"actual_{target}"].to_numpy(float)
    cat_oof = oof[f"catboost_{target}"].to_numpy(float)
    residual_oof = oof[f"residual_lstm_{target}"].to_numpy(float)
    scale = optimize_scale(actual_oof, cat_oof, residual_oof, args)
    raw_oof = cat_oof + scale * residual_oof
    raw_test = test[f"catboost_{target}"].to_numpy(float) + scale * test[f"residual_lstm_{target}"].to_numpy(float)
    recipe = {"residual_scale": scale}
    if target == "TA":
        pooled_offset = float(np.mean(actual_oof - raw_oof))
        pooled_oof = raw_oof + pooled_offset
        latest_year = int(oof.year.max())
        latest_mask = oof.year.to_numpy(int) == latest_year
        latest_offset = float(np.mean(actual_oof[latest_mask] - pooled_oof[latest_mask]))
        prediction_oof = pooled_oof + latest_offset
        prediction_test = raw_test + pooled_offset + latest_offset
        recipe.update({
            "pooled_oof_offset": pooled_offset,
            "latest_oof_year": latest_year,
            "latest_oof_offset": latest_offset,
        })
    else:
        prediction_oof = np.clip(raw_oof, 0.0, 100.0)
        prediction_test = np.clip(raw_test, 0.0, 100.0)
    oof = oof.copy()
    test = test.copy()
    oof[f"prediction_{target}"] = prediction_oof
    test[f"prediction_{target}"] = prediction_test
    metrics = {
        "oof_RMSE": base.rmse(actual_oof, prediction_oof),
        "test_RMSE": base.rmse(test[f"actual_{target}"].to_numpy(float), prediction_test),
        "test_MAE": base.diagnostic_metrics(
            test[f"actual_{target}"].to_numpy(float), prediction_test
        )["MAE"],
        "test_bias": base.diagnostic_metrics(
            test[f"actual_{target}"].to_numpy(float), prediction_test
        )["bias"],
    }
    return oof, test, recipe, metrics


def run_variant(
    variant, master_solar, x_raw, static_all, labels_by_target,
    sequence_keys, missing_counts, args, output_dir,
):
    target_outputs = {}
    fold_rows = []
    recipes = {}
    metrics = {}
    for target in ["TA", "HM"]:
        parts = []
        summaries = []
        for year in args.oof_years:
            part, summary = one_fold(
                variant, target, year, master_solar, x_raw, static_all,
                labels_by_target[target], sequence_keys, missing_counts, args,
            )
            parts.append(part)
            summaries.append(summary)
            print(
                f"[{variant.name} {target} OOF {year}] "
                f"cat={summary['cat_RMSE']:.4f} scale1={summary['scale1_RMSE']:.4f} "
                f"epoch={summary['residual_lstm_epoch']}", flush=True,
            )
        fold_summary = pd.DataFrame(summaries)
        fold_rows.extend(summaries)
        oof = pd.concat(parts, ignore_index=True)
        test = final_predictions(
            variant, target, master_solar, x_raw, static_all,
            labels_by_target[target], sequence_keys, missing_counts,
            fold_summary, args,
        )
        oof, test, recipe, target_metrics = score_target(oof, test, target, args)
        recipes[target] = recipe
        metrics[target] = target_metrics
        target_outputs[target] = (oof, test)
        oof.to_csv(output_dir / f"{variant.name}_{target}_oof.csv", index=False)
        test.to_csv(output_dir / f"{variant.name}_{target}_test.csv", index=False)
    ta_test = target_outputs["TA"][1]
    hm_test = target_outputs["HM"][1]
    if not ta_test[KEYS].equals(hm_test[KEYS]):
        raise ValueError("TA/HM test keys differ")
    result = {
        "step": variant.step,
        "variant": variant.name,
        "removal": variant.removal,
        "tabular_feature_count": len(variant.tabular_features),
        "lstm_static_feature_count": len(variant.lstm_static_features),
        "TA_RMSE": metrics["TA"]["test_RMSE"],
        "TA_MAE": metrics["TA"]["test_MAE"],
        "TA_bias": metrics["TA"]["test_bias"],
        "HM_RMSE": metrics["HM"]["test_RMSE"],
        "HM_MAE": metrics["HM"]["test_MAE"],
        "HM_bias": metrics["HM"]["test_bias"],
        "competition_score": metrics["TA"]["test_RMSE"] + 0.1 * metrics["HM"]["test_RMSE"],
        "TA_oof_RMSE": metrics["TA"]["oof_RMSE"],
        "HM_oof_RMSE": metrics["HM"]["oof_RMSE"],
    }
    return result, fold_rows, recipes


def manifest(variants):
    rows = []
    for variant in variants:
        for feature in base.TABULAR_FEATURES:
            rows.append({
                "step": variant.step, "variant": variant.name, "branch": "CatBoost",
                "feature": feature, "included": feature in variant.tabular_features,
            })
        for feature in base.LSTM_STATIC_FEATURES:
            rows.append({
                "step": variant.step, "variant": variant.name, "branch": "LSTM_static",
                "feature": feature, "included": feature in variant.lstm_static_features,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    variants = build_variants()
    if args.steps is not None:
        requested = set(args.steps)
        unknown = requested - {variant.step for variant in variants}
        if unknown:
            raise ValueError(f"unknown --steps: {sorted(unknown)}")
        variants = [variant for variant in variants if variant.step in requested]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    master_solar = base.add_solar_features(pd.read_csv(args.master_csv))
    shortterm = pd.read_csv(args.shortterm_long_csv)
    x_raw, static_all, ta_labels, sequence_keys, missing_counts = base.build_sequences(shortterm)
    labels_by_target = {
        "TA": ta_labels,
        "HM": sequence_labels(shortterm, sequence_keys, "HM"),
    }
    rows = []
    fold_rows = []
    recipes = {}
    baseline_result = None
    for variant in variants:
        print("\n" + "=" * 96, flush=True)
        print(f"[{variant.name}] {variant.removal}", flush=True)
        if not variant.effective:
            if baseline_result is None:
                raise RuntimeError("no-op step requires completed baseline")
            result = dict(baseline_result)
            result.update({
                "step": variant.step,
                "variant": variant.name,
                "removal": variant.removal,
                "tabular_feature_count": len(variant.tabular_features),
                "lstm_static_feature_count": len(variant.lstm_static_features),
            })
            recipes[variant.name] = {"reused_from": variants[0].name, "reason": variant.removal}
        else:
            result, variant_folds, recipe = run_variant(
                variant, master_solar, x_raw, static_all, labels_by_target,
                sequence_keys, missing_counts, args, output_dir,
            )
            fold_rows.extend(variant_folds)
            recipes[variant.name] = recipe
        rows.append(result)
        if variant.step == 0:
            baseline_result = dict(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    results = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    for metric in ["TA_RMSE", "HM_RMSE", "competition_score"]:
        results[f"delta_previous_{metric}"] = results[metric].diff()
        results[f"delta_baseline_{metric}"] = results[metric] - results.loc[0, metric]
    results["improved_vs_previous"] = results.delta_previous_competition_score.lt(0)
    results.loc[0, "improved_vs_previous"] = False
    results.to_csv(output_dir / "cumulative_ablation_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_training_metrics.csv", index=False)
    manifest(variants).to_csv(output_dir / "feature_manifest.csv", index=False)
    (output_dir / "recipes.json").write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8")
    best = results.sort_values("competition_score").iloc[0]
    summary = {
        "design": "cumulative removal; rolling OOF 2022-2024; retrospective test 2025",
        "score_formula": "RMSE_TA + 0.1 * RMSE_HM",
        "best_variant": best.to_dict(),
        "results": results.to_dict(orient="records"),
        "rules_note": "Only LE1B channels, official station metadata, and date/solar calculations are model inputs.",
        "caveat": "2025 has been inspected repeatedly and is not a pristine leaderboard estimate.",
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[FINAL CUMULATIVE ABLATION]", flush=True)
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
