#!/usr/bin/env python3
"""Run TA experiments 1-3 with rolling-year OOF selection.

Experiments
-----------
1. Bias-calibrated full hierarchy: direct LSTM + (CatBoost + Ridge).
2. CatBoost + residual LSTM: the LSTM target is TA - CatBoost prediction.
3. Ridge removal: bias-calibrated direct LSTM + CatBoost only.

Only 2019-2024 labels are used to choose calibration constants and ensemble
weights. 2025 is held out until the final comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_dualbranch_ta_baseline as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oof-years", nargs="+", type=int, default=[2022, 2023, 2024])
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
    parser.add_argument("--residual-scale-max", type=float, default=2.00)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(np.asarray(array, dtype=float))
    return mask


def offset(actual: np.ndarray, prediction: np.ndarray) -> float:
    mask = finite_mask(actual, prediction)
    return float(np.mean(actual[mask] - prediction[mask]))


def optimize_scale(
    actual: np.ndarray,
    baseline: np.ndarray,
    correction: np.ndarray,
    step: float,
    maximum: float,
) -> tuple[float, float]:
    scales = np.arange(0.0, maximum + step / 2.0, step)
    scores = [base.rmse(actual, baseline + scale * correction) for scale in scales]
    index = int(np.argmin(scores))
    return float(scales[index]), float(scores[index])


def predict_lstm_raw(model, normalization, device, x_raw, static_raw) -> np.ndarray:
    dynamic, static = base.transform_lstm_inputs(x_raw, static_raw, normalization)
    return base.predict_lstm(model, dynamic, static, normalization, device)


def align_sequence_master(engineered: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    aligned = base.align_master(engineered, keys)
    if len(aligned) != len(keys):
        raise ValueError(f"master/sequence key mismatch: {len(keys)} keys, {len(aligned)} matches")
    return aligned


def train_one_oof_fold(
    year: int,
    master_solar: pd.DataFrame,
    x_raw: np.ndarray,
    static_raw: np.ndarray,
    sequence_y: np.ndarray,
    sequence_keys: pd.DataFrame,
    missing_counts: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float]]:
    print("\n" + "=" * 88, flush=True)
    print(f"[OOF {year}] train <= {year - 1}, validate == {year}", flush=True)
    fold_args = copy.copy(args)
    fold_args.seed = args.seed + year
    base.seed_everything(fold_args.seed)

    stats = base.fit_channel_stats(master_solar, year - 1)
    engineered = base.engineer_tabular(master_solar, stats)
    master_train = engineered[(engineered.year <= year - 1) & engineered.TA.notna()].copy()
    master_validation = engineered[(engineered.year == year) & engineered.TA.notna()].copy()

    cat, cat_iterations = base.train_catboost_dev(master_train, master_validation, fold_args)
    ridge, ridge_alpha, _ = base.fit_ridge(master_train, master_validation)

    available_master_keys = (
        engineered.loc[engineered.TA.notna(), ["Date", "STN_ID"]]
        .drop_duplicates()
        .assign(_master_available=True)
    )
    sequence_availability = sequence_keys[["Date", "STN_ID"]].merge(
        available_master_keys, on=["Date", "STN_ID"], how="left", validate="m:1"
    )["_master_available"].fillna(False).to_numpy(dtype=bool)
    train_mask = (
        (sequence_keys.year.to_numpy() <= year - 1)
        & np.isfinite(sequence_y)
        & sequence_availability
    )
    validation_mask = (
        (sequence_keys.year.to_numpy() == year)
        & np.isfinite(sequence_y)
        & sequence_availability
    )
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError(f"fold {year} has no train or validation sequence")

    direct_model, direct_norm, direct_device, direct_epoch, direct_history = base.train_lstm_dev(
        x_raw[train_indices], static_raw[train_indices], sequence_y[train_indices],
        x_raw[validation_indices], static_raw[validation_indices], sequence_y[validation_indices],
        fold_args,
    )
    direct_validation = predict_lstm_raw(
        direct_model, direct_norm, direct_device,
        x_raw[validation_indices], static_raw[validation_indices],
    )

    train_keys = sequence_keys.iloc[train_indices].reset_index(drop=True)
    validation_keys = sequence_keys.iloc[validation_indices].reset_index(drop=True)
    train_aligned = align_sequence_master(engineered, train_keys)
    validation_aligned = align_sequence_master(engineered, validation_keys)
    if not np.allclose(train_aligned.TA.to_numpy(), sequence_y[train_indices], equal_nan=True):
        raise ValueError(f"fold {year}: train labels disagree between master and short-term table")
    if not np.allclose(validation_aligned.TA.to_numpy(), sequence_y[validation_indices], equal_nan=True):
        raise ValueError(f"fold {year}: validation labels disagree between tables")

    cat_train = cat.predict(train_aligned[base.TABULAR_FEATURES])
    cat_validation = cat.predict(validation_aligned[base.TABULAR_FEATURES])
    ridge_validation = ridge.predict(validation_aligned[base.TABULAR_FEATURES])
    residual_train = sequence_y[train_indices] - cat_train
    residual_validation_target = sequence_y[validation_indices] - cat_validation

    residual_model, residual_norm, residual_device, residual_epoch, residual_history = base.train_lstm_dev(
        x_raw[train_indices], static_raw[train_indices], residual_train,
        x_raw[validation_indices], static_raw[validation_indices], residual_validation_target,
        fold_args,
    )
    residual_validation = predict_lstm_raw(
        residual_model, residual_norm, residual_device,
        x_raw[validation_indices], static_raw[validation_indices],
    )

    frame = validation_keys[["Date", "STN_ID", "year"]].copy()
    frame["actual_TA"] = sequence_y[validation_indices]
    frame["catboost_TA"] = cat_validation
    frame["ridge_TA"] = ridge_validation
    frame["direct_lstm_TA"] = direct_validation
    frame["residual_lstm_TA"] = residual_validation
    frame["sequence_missing_cells"] = missing_counts[validation_indices]

    direct_history = direct_history.assign(fold_year=year, target="absolute_TA")
    residual_history = residual_history.assign(fold_year=year, target="catboost_residual")
    frame.attrs["histories"] = pd.concat([direct_history, residual_history], ignore_index=True)
    summary = {
        "fold_year": year,
        "train_master_rows": int(len(master_train)),
        "train_sequences": int(len(train_indices)),
        "validation_sequences": int(len(validation_indices)),
        "cat_iterations": int(cat_iterations),
        "ridge_alpha": float(ridge_alpha),
        "direct_lstm_epoch": int(direct_epoch),
        "residual_lstm_epoch": int(residual_epoch),
        "catboost_RMSE": base.rmse(frame.actual_TA.to_numpy(), cat_validation),
        "direct_lstm_RMSE": base.rmse(frame.actual_TA.to_numpy(), direct_validation),
        "cat_plus_residual_lstm_RMSE_scale1": base.rmse(
            frame.actual_TA.to_numpy(), cat_validation + residual_validation
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return frame, summary


def build_recipe(oof: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    y = oof.actual_TA.to_numpy(dtype=float)
    cat = oof.catboost_TA.to_numpy(dtype=float)
    ridge = oof.ridge_TA.to_numpy(dtype=float)
    direct = oof.direct_lstm_TA.to_numpy(dtype=float)
    residual = oof.residual_lstm_TA.to_numpy(dtype=float)

    raw_cat_weight, _ = base.optimize_weight(y, cat, ridge, args.weight_step)
    raw_tabular = raw_cat_weight * cat + (1.0 - raw_cat_weight) * ridge
    raw_lstm_weight, _ = base.optimize_weight(y, direct, raw_tabular, args.weight_step)

    offsets = {
        "catboost": offset(y, cat),
        "ridge": offset(y, ridge),
        "direct_lstm": offset(y, direct),
    }
    cat_c = cat + offsets["catboost"]
    ridge_c = ridge + offsets["ridge"]
    direct_c = direct + offsets["direct_lstm"]
    calibrated_cat_weight, _ = base.optimize_weight(y, cat_c, ridge_c, args.weight_step)
    calibrated_tabular = calibrated_cat_weight * cat_c + (1.0 - calibrated_cat_weight) * ridge_c
    calibrated_lstm_weight, _ = base.optimize_weight(y, direct_c, calibrated_tabular, args.weight_step)
    no_ridge_lstm_weight, _ = base.optimize_weight(y, direct_c, cat_c, args.weight_step)

    residual_scale, _ = optimize_scale(
        y, cat, residual, args.weight_step, args.residual_scale_max
    )
    residual_raw = cat + residual_scale * residual
    residual_offset = offset(y, residual_raw)

    return {
        "raw_cat_weight": raw_cat_weight,
        "raw_lstm_weight": raw_lstm_weight,
        "catboost_offset": offsets["catboost"],
        "ridge_offset": offsets["ridge"],
        "direct_lstm_offset": offsets["direct_lstm"],
        "calibrated_cat_weight": calibrated_cat_weight,
        "calibrated_lstm_weight": calibrated_lstm_weight,
        "no_ridge_lstm_weight": no_ridge_lstm_weight,
        "residual_scale": residual_scale,
        "residual_final_offset": residual_offset,
    }


def apply_recipe(frame: pd.DataFrame, recipe: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    cat = out.catboost_TA.to_numpy(dtype=float)
    ridge = out.ridge_TA.to_numpy(dtype=float)
    direct = out.direct_lstm_TA.to_numpy(dtype=float)
    residual = out.residual_lstm_TA.to_numpy(dtype=float)

    raw_tabular = recipe["raw_cat_weight"] * cat + (1.0 - recipe["raw_cat_weight"]) * ridge
    out["Baseline_FullRaw"] = (
        recipe["raw_lstm_weight"] * direct
        + (1.0 - recipe["raw_lstm_weight"]) * raw_tabular
    )

    cat_c = cat + recipe["catboost_offset"]
    ridge_c = ridge + recipe["ridge_offset"]
    direct_c = direct + recipe["direct_lstm_offset"]
    calibrated_tabular = (
        recipe["calibrated_cat_weight"] * cat_c
        + (1.0 - recipe["calibrated_cat_weight"]) * ridge_c
    )
    out["Exp1_FullCalibrated"] = (
        recipe["calibrated_lstm_weight"] * direct_c
        + (1.0 - recipe["calibrated_lstm_weight"]) * calibrated_tabular
    )
    out["Exp2_CatResidualLSTM"] = (
        cat
        + recipe["residual_scale"] * residual
        + recipe["residual_final_offset"]
    )
    out["Exp3_NoRidge"] = (
        recipe["no_ridge_lstm_weight"] * direct_c
        + (1.0 - recipe["no_ridge_lstm_weight"]) * cat_c
    )
    out["CatBoost_Raw"] = cat
    out["CatBoost_Calibrated"] = cat_c
    return out


def metric_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    models = [
        "CatBoost_Raw", "CatBoost_Calibrated", "Baseline_FullRaw",
        "Exp1_FullCalibrated", "Exp2_CatResidualLSTM", "Exp3_NoRidge",
    ]
    rows = []
    y = frame.actual_TA.to_numpy(dtype=float)
    for model in models:
        rows.append({"split": split, "model": model, **base.diagnostic_metrics(y, frame[model].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def subgroup_table(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["temperature_band"] = pd.cut(
        work.actual_TA,
        [-np.inf, 28.0, 30.0, 32.0, np.inf],
        labels=["<28", "28-<30", "30-<32", ">=32"],
        right=False,
    )
    work["sequence_group"] = np.where(work.sequence_missing_cells.eq(0), "complete", "incomplete")
    rows = []
    for dimension in ["temperature_band", "sequence_group"]:
        for group, part in work.groupby(dimension, observed=True):
            y = part.actual_TA.to_numpy(dtype=float)
            for model in ["Exp1_FullCalibrated", "Exp2_CatResidualLSTM", "Exp3_NoRidge"]:
                rows.append({
                    "dimension": dimension,
                    "group": str(group),
                    "model": model,
                    **base.diagnostic_metrics(y, part[model].to_numpy(dtype=float)),
                })
    return pd.DataFrame(rows)


def train_final_models(
    master_solar: pd.DataFrame,
    x_raw: np.ndarray,
    static_raw: np.ndarray,
    sequence_y: np.ndarray,
    sequence_keys: pd.DataFrame,
    missing_counts: np.ndarray,
    fold_summary: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    train_end = args.test_year - 1
    stats = base.fit_channel_stats(master_solar, train_end)
    engineered = base.engineer_tabular(master_solar, stats)
    master_train = engineered[(engineered.year <= train_end) & engineered.TA.notna()].copy()
    cat_iterations = max(1, int(round(fold_summary.cat_iterations.median())))
    ridge_alpha = float(fold_summary.ridge_alpha.median())
    direct_epoch = max(1, int(round(fold_summary.direct_lstm_epoch.median())))
    residual_epoch = max(1, int(round(fold_summary.residual_lstm_epoch.median())))

    final_args = copy.copy(args)
    final_args.seed = args.seed + args.test_year
    base.seed_everything(final_args.seed)
    cat = base.train_catboost_fixed(master_train, cat_iterations, final_args)
    ridge = base.refit_ridge(master_train, ridge_alpha)

    available_master_keys = (
        engineered.loc[engineered.TA.notna(), ["Date", "STN_ID"]]
        .drop_duplicates()
        .assign(_master_available=True)
    )
    sequence_availability = sequence_keys[["Date", "STN_ID"]].merge(
        available_master_keys, on=["Date", "STN_ID"], how="left", validate="m:1"
    )["_master_available"].fillna(False).to_numpy(dtype=bool)
    train_mask = (
        (sequence_keys.year.to_numpy() <= train_end)
        & np.isfinite(sequence_y)
        & sequence_availability
    )
    test_mask = (
        (sequence_keys.year.to_numpy() == args.test_year)
        & np.isfinite(sequence_y)
        & sequence_availability
    )
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    train_keys = sequence_keys.iloc[train_indices].reset_index(drop=True)
    test_keys = sequence_keys.iloc[test_indices].reset_index(drop=True)
    train_aligned = align_sequence_master(engineered, train_keys)
    test_aligned = align_sequence_master(engineered, test_keys)

    cat_train = cat.predict(train_aligned[base.TABULAR_FEATURES])
    cat_test = cat.predict(test_aligned[base.TABULAR_FEATURES])
    ridge_test = ridge.predict(test_aligned[base.TABULAR_FEATURES])
    residual_train = sequence_y[train_indices] - cat_train

    direct_model, direct_norm, direct_device = base.train_lstm_fixed(
        x_raw[train_indices], static_raw[train_indices], sequence_y[train_indices], direct_epoch, final_args
    )
    residual_model, residual_norm, residual_device = base.train_lstm_fixed(
        x_raw[train_indices], static_raw[train_indices], residual_train, residual_epoch, final_args
    )
    direct_test = predict_lstm_raw(
        direct_model, direct_norm, direct_device, x_raw[test_indices], static_raw[test_indices]
    )
    residual_test = predict_lstm_raw(
        residual_model, residual_norm, residual_device, x_raw[test_indices], static_raw[test_indices]
    )

    model_dir = output_dir / "evaluation_models"
    model_dir.mkdir(exist_ok=True)
    cat.save_model(model_dir / "catboost_TA.cbm")
    import joblib

    joblib.dump(ridge, model_dir / "ridge_TA.joblib")
    base.save_lstm(model_dir / "direct_lstm_TA.pt", direct_model, direct_norm, direct_epoch, final_args)
    base.save_lstm(model_dir / "residual_lstm_TA.pt", residual_model, residual_norm, residual_epoch, final_args)
    (model_dir / "tabular_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    frame = test_keys[["Date", "STN_ID", "year"]].copy()
    frame["actual_TA"] = sequence_y[test_indices]
    frame["catboost_TA"] = cat_test
    frame["ridge_TA"] = ridge_test
    frame["direct_lstm_TA"] = direct_test
    frame["residual_lstm_TA"] = residual_test
    frame["sequence_missing_cells"] = missing_counts[test_indices]
    return frame


def main() -> None:
    args = parse_args()
    if sorted(set(args.oof_years)) != args.oof_years:
        raise ValueError("--oof-years must be unique and ascending")
    if max(args.oof_years) >= args.test_year:
        raise ValueError("OOF years must be earlier than the test year")
    base.seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    master_raw = pd.read_csv(args.master_csv)
    master_solar = base.add_solar_features(master_raw)
    if master_solar.duplicated(["Date", "STN_ID"]).any():
        raise ValueError("master CSV has duplicate Date/STN_ID rows")
    shortterm = pd.read_csv(args.shortterm_long_csv)
    x_raw, static_raw, sequence_y, sequence_keys, missing_counts = base.build_sequences(shortterm)

    oof_parts = []
    histories = []
    fold_summaries = []
    for year in args.oof_years:
        part, summary = train_one_oof_fold(
            year, master_solar, x_raw, static_raw, sequence_y, sequence_keys, missing_counts, args
        )
        histories.append(part.attrs.pop("histories"))
        oof_parts.append(part)
        fold_summaries.append(summary)

    oof = pd.concat(oof_parts, ignore_index=True)
    fold_summary = pd.DataFrame(fold_summaries)
    history = pd.concat(histories, ignore_index=True)
    recipe = build_recipe(oof, args)
    scored_oof = apply_recipe(oof, recipe)
    test_raw = train_final_models(
        master_solar, x_raw, static_raw, sequence_y, sequence_keys, missing_counts,
        fold_summary, args, output_dir,
    )
    scored_test = apply_recipe(test_raw, recipe)

    metrics = pd.concat(
        [metric_table(scored_oof, "rolling_oof"), metric_table(scored_test, f"test_{args.test_year}")],
        ignore_index=True,
    )
    subgroup = subgroup_table(scored_test)
    oof.to_csv(output_dir / "rolling_oof_base_predictions.csv", index=False)
    scored_oof.to_csv(output_dir / "rolling_oof_scored_predictions.csv", index=False)
    scored_test.to_csv(output_dir / f"test_{args.test_year}_predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    subgroup.to_csv(output_dir / f"subgroup_metrics_{args.test_year}.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_training_summary.csv", index=False)
    history.to_csv(output_dir / "lstm_training_history.csv", index=False)
    (output_dir / "calibration_and_weights.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test_metrics = metrics[metrics.split.eq(f"test_{args.test_year}")].sort_values("RMSE")
    summary = {
        "oof_years": args.oof_years,
        "test_year": args.test_year,
        "oof_rows": int(len(scored_oof)),
        "test_rows": int(len(scored_test)),
        "recipe": recipe,
        "best_test_model": test_metrics.iloc[0].to_dict(),
        "test_metrics": test_metrics.to_dict(orient="records"),
        "rules_note": "Inputs are limited to LE1B channels, official station coordinates/elevation, and date/solar calculations.",
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[FINAL METRICS]", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print("\n[CALIBRATION / WEIGHTS]", flush=True)
    print(json.dumps(recipe, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
