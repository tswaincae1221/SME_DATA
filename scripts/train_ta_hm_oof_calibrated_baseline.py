#!/usr/bin/env python3
"""Calibrate the TA residual ensemble and train a rules-safe HM baseline.

The script consumes the rolling OOF/test predictions produced by
``experiment_ta_calibration_residual_ablation.py``.  The TA prediction is
calibrated with the mean residual of the most recent OOF year.  Humidity is
then modelled as CatBoost plus an LSTM prediction of the CatBoost residual.

The optional TA input to the HM calibration layer is always a model prediction:
rolling OOF TA for HM OOF rows and the frozen TA model prediction for the test
rows.  Observed ASOS TA/HM are used only as training/evaluation labels.  No
ASOS lag, station climatology, external weather product, or external static
geography is used.
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

import experiment_ta_calibration_residual_ablation as taexp  # noqa: E402
import train_dualbranch_ta_baseline as base  # noqa: E402


KEYS = ["Date", "STN_ID"]
TA_BASE_COLUMN = "Exp2_CatResidualLSTM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--ta-oof-csv", required=True)
    parser.add_argument("--ta-test-csv", required=True)
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
    parser.add_argument("--residual-scale-step", type=float, default=0.02)
    parser.add_argument("--residual-scale-max", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_prediction_frame(path: str | Path, expected_years: set[int]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [*KEYS, "year", "actual_TA", TA_BASE_COLUMN]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing TA columns: {missing}")
    frame["Date"] = pd.to_numeric(frame["Date"], errors="raise").round().astype("int64")
    frame["STN_ID"] = pd.to_numeric(frame["STN_ID"], errors="raise").round().astype("int64")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype("int16")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} contains duplicate Date/STN_ID keys")
    observed_years = set(frame.year.unique().tolist())
    if observed_years != expected_years:
        raise ValueError(f"{path}: expected years {sorted(expected_years)}, got {sorted(observed_years)}")
    return frame


def calibrate_ta(
    oof: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    latest_year = int(oof.year.max())
    latest = oof[oof.year.eq(latest_year)]
    correction = taexp.offset(
        latest.actual_TA.to_numpy(dtype=float),
        latest[TA_BASE_COLUMN].to_numpy(dtype=float),
    )

    oof = oof.copy()
    test = test.copy()
    oof["TA_PooledOOF"] = oof[TA_BASE_COLUMN]
    test["TA_PooledOOF"] = test[TA_BASE_COLUMN]
    oof["TA_LatestOOFCalibrated"] = oof[TA_BASE_COLUMN] + correction
    test["TA_LatestOOFCalibrated"] = test[TA_BASE_COLUMN] + correction

    rows: list[dict[str, object]] = []
    for split, frame in [("rolling_oof", oof), (f"test_{int(test.year.iloc[0])}", test)]:
        for model in ["TA_PooledOOF", "TA_LatestOOFCalibrated"]:
            rows.append({"split": split, "model": model, **base.diagnostic_metrics(
                frame.actual_TA.to_numpy(dtype=float), frame[model].to_numpy(dtype=float)
            )})
    for year, frame in oof.groupby("year"):
        for model in ["TA_PooledOOF", "TA_LatestOOFCalibrated"]:
            rows.append({"split": f"oof_{int(year)}", "model": model, **base.diagnostic_metrics(
                frame.actual_TA.to_numpy(dtype=float), frame[model].to_numpy(dtype=float)
            )})
    metrics = pd.DataFrame(rows)
    test_base = metrics.query("split.str.startswith('test_') and model == 'TA_PooledOOF'", engine="python").iloc[0]
    test_cal = metrics.query("split.str.startswith('test_') and model == 'TA_LatestOOFCalibrated'", engine="python").iloc[0]
    recipe: dict[str, object] = {
        "base_column": TA_BASE_COLUMN,
        "calibration_method": "mean(actual_minus_prediction) from latest OOF year",
        "latest_oof_year": latest_year,
        "additive_correction": correction,
        "test_base_RMSE": float(test_base.RMSE),
        "test_calibrated_RMSE": float(test_cal.RMSE),
        "test_improved": bool(test_cal.RMSE < test_base.RMSE),
        "test_RMSE_delta": float(test_cal.RMSE - test_base.RMSE),
    }
    return oof, test, recipe, metrics


def hm_labels_for_sequences(shortterm: pd.DataFrame, sequence_keys: pd.DataFrame) -> np.ndarray:
    required = ["Date", "TimeKST", "STN_ID", "HM"]
    missing = [column for column in required if column not in shortterm]
    if missing:
        raise ValueError(f"short-term CSV is missing HM label columns: {missing}")
    labels = shortterm.loc[pd.to_numeric(shortterm.TimeKST).eq(1400), ["Date", "STN_ID", "HM"]].copy()
    labels["Date"] = pd.to_numeric(labels.Date, errors="raise").round().astype("int64")
    labels["STN_ID"] = pd.to_numeric(labels.STN_ID, errors="raise").round().astype("int64")
    if labels.duplicated(KEYS).any():
        raise ValueError("short-term table contains duplicate 14:00 HM labels")
    aligned = sequence_keys[KEYS].merge(labels, on=KEYS, how="left", validate="1:1")
    if len(aligned) != len(sequence_keys):
        raise AssertionError("HM label alignment changed sequence row count")
    return pd.to_numeric(aligned.HM, errors="coerce").to_numpy(dtype=np.float32)


def align_hm_master(engineered: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    aligned = keys[KEYS].merge(
        engineered[[*KEYS, "HM", *base.TABULAR_FEATURES]],
        on=KEYS, how="inner", validate="1:1",
    )
    if len(aligned) != len(keys):
        raise ValueError(f"master/sequence key mismatch: {len(keys)} keys, {len(aligned)} matches")
    return aligned


def train_catboost_hm_dev(train: pd.DataFrame, validation: pd.DataFrame, args):
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
        train[base.TABULAR_FEATURES], train.HM,
        eval_set=(validation[base.TABULAR_FEATURES], validation.HM),
        early_stopping_rounds=args.cat_early_stopping_rounds,
        use_best_model=True,
    )
    return model, max(1, int(model.get_best_iteration()) + 1)


def train_catboost_hm_fixed(train: pd.DataFrame, iterations: int, args):
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
    model.fit(train[base.TABULAR_FEATURES], train.HM)
    return model


def sequence_availability(engineered: pd.DataFrame, sequence_keys: pd.DataFrame) -> np.ndarray:
    available = (
        engineered.loc[engineered.HM.notna(), KEYS]
        .drop_duplicates()
        .assign(_available=True)
    )
    return sequence_keys[KEYS].merge(
        available, on=KEYS, how="left", validate="m:1"
    )._available.fillna(False).to_numpy(dtype=bool)


def train_hm_oof_fold(
    year: int,
    master_solar: pd.DataFrame,
    x_raw: np.ndarray,
    static_raw: np.ndarray,
    hm_y: np.ndarray,
    sequence_keys: pd.DataFrame,
    missing_counts: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    print("\n" + "=" * 88, flush=True)
    print(f"[HM OOF {year}] train <= {year - 1}, validate == {year}", flush=True)
    fold_args = copy.copy(args)
    fold_args.seed = args.seed + 10000 + year
    base.seed_everything(fold_args.seed)

    stats = base.fit_channel_stats(master_solar, year - 1)
    engineered = base.engineer_tabular(master_solar, stats)
    master_train = engineered[(engineered.year <= year - 1) & engineered.HM.notna()].copy()
    master_validation = engineered[(engineered.year == year) & engineered.HM.notna()].copy()
    cat, cat_iterations = train_catboost_hm_dev(master_train, master_validation, fold_args)

    available = sequence_availability(engineered, sequence_keys)
    train_mask = (
        sequence_keys.year.to_numpy().astype(int) <= year - 1
    ) & np.isfinite(hm_y) & available
    validation_mask = (
        sequence_keys.year.to_numpy().astype(int) == year
    ) & np.isfinite(hm_y) & available
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError(f"HM fold {year} has no train or validation sequence")

    train_keys = sequence_keys.iloc[train_indices].reset_index(drop=True)
    validation_keys = sequence_keys.iloc[validation_indices].reset_index(drop=True)
    train_aligned = align_hm_master(engineered, train_keys)
    validation_aligned = align_hm_master(engineered, validation_keys)
    if not np.allclose(train_aligned.HM.to_numpy(), hm_y[train_indices], equal_nan=True):
        raise ValueError(f"HM fold {year}: train labels disagree between tables")
    if not np.allclose(validation_aligned.HM.to_numpy(), hm_y[validation_indices], equal_nan=True):
        raise ValueError(f"HM fold {year}: validation labels disagree between tables")

    cat_train = cat.predict(train_aligned[base.TABULAR_FEATURES])
    cat_validation = cat.predict(validation_aligned[base.TABULAR_FEATURES])
    residual_train = hm_y[train_indices] - cat_train
    residual_validation_target = hm_y[validation_indices] - cat_validation

    direct_model, direct_norm, direct_device, direct_epoch, direct_history = base.train_lstm_dev(
        x_raw[train_indices], static_raw[train_indices], hm_y[train_indices],
        x_raw[validation_indices], static_raw[validation_indices], hm_y[validation_indices],
        fold_args,
    )
    direct_validation = taexp.predict_lstm_raw(
        direct_model, direct_norm, direct_device,
        x_raw[validation_indices], static_raw[validation_indices],
    )
    residual_model, residual_norm, residual_device, residual_epoch, residual_history = base.train_lstm_dev(
        x_raw[train_indices], static_raw[train_indices], residual_train,
        x_raw[validation_indices], static_raw[validation_indices], residual_validation_target,
        fold_args,
    )
    residual_validation = taexp.predict_lstm_raw(
        residual_model, residual_norm, residual_device,
        x_raw[validation_indices], static_raw[validation_indices],
    )

    frame = validation_keys[[*KEYS, "year"]].copy()
    frame["actual_HM"] = hm_y[validation_indices]
    frame["catboost_HM"] = cat_validation
    frame["direct_lstm_HM"] = direct_validation
    frame["residual_lstm_HM"] = residual_validation
    frame["sequence_missing_cells"] = missing_counts[validation_indices]
    histories = pd.concat([
        direct_history.assign(fold_year=year, target="absolute_HM"),
        residual_history.assign(fold_year=year, target="catboost_HM_residual"),
    ], ignore_index=True)
    summary = {
        "fold_year": year,
        "train_master_rows": int(len(master_train)),
        "validation_master_rows": int(len(master_validation)),
        "train_sequences": int(len(train_indices)),
        "validation_sequences": int(len(validation_indices)),
        "cat_iterations": int(cat_iterations),
        "direct_lstm_epoch": int(direct_epoch),
        "residual_lstm_epoch": int(residual_epoch),
        "catboost_RMSE": base.rmse(frame.actual_HM.to_numpy(), cat_validation),
        "direct_lstm_RMSE": base.rmse(frame.actual_HM.to_numpy(), direct_validation),
        "cat_plus_residual_RMSE_scale1": base.rmse(
            frame.actual_HM.to_numpy(), cat_validation + residual_validation
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return frame, summary, histories


def train_hm_final(
    master_solar: pd.DataFrame,
    x_raw: np.ndarray,
    static_raw: np.ndarray,
    hm_y: np.ndarray,
    sequence_keys: pd.DataFrame,
    missing_counts: np.ndarray,
    fold_summary: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    train_end = args.test_year - 1
    stats = base.fit_channel_stats(master_solar, train_end)
    engineered = base.engineer_tabular(master_solar, stats)
    master_train = engineered[(engineered.year <= train_end) & engineered.HM.notna()].copy()
    cat_iterations = max(1, int(round(fold_summary.cat_iterations.median())))
    direct_epoch = max(1, int(round(fold_summary.direct_lstm_epoch.median())))
    residual_epoch = max(1, int(round(fold_summary.residual_lstm_epoch.median())))

    final_args = copy.copy(args)
    final_args.seed = args.seed + 10000 + args.test_year
    base.seed_everything(final_args.seed)
    cat = train_catboost_hm_fixed(master_train, cat_iterations, final_args)

    available = sequence_availability(engineered, sequence_keys)
    years = sequence_keys.year.to_numpy().astype(int)
    train_indices = np.flatnonzero((years <= train_end) & np.isfinite(hm_y) & available)
    test_indices = np.flatnonzero((years == args.test_year) & np.isfinite(hm_y) & available)
    train_keys = sequence_keys.iloc[train_indices].reset_index(drop=True)
    test_keys = sequence_keys.iloc[test_indices].reset_index(drop=True)
    train_aligned = align_hm_master(engineered, train_keys)
    test_aligned = align_hm_master(engineered, test_keys)
    cat_train = cat.predict(train_aligned[base.TABULAR_FEATURES])
    cat_test = cat.predict(test_aligned[base.TABULAR_FEATURES])
    residual_train = hm_y[train_indices] - cat_train

    direct_model, direct_norm, direct_device = base.train_lstm_fixed(
        x_raw[train_indices], static_raw[train_indices], hm_y[train_indices], direct_epoch, final_args
    )
    residual_model, residual_norm, residual_device = base.train_lstm_fixed(
        x_raw[train_indices], static_raw[train_indices], residual_train, residual_epoch, final_args
    )
    direct_test = taexp.predict_lstm_raw(
        direct_model, direct_norm, direct_device, x_raw[test_indices], static_raw[test_indices]
    )
    residual_test = taexp.predict_lstm_raw(
        residual_model, residual_norm, residual_device, x_raw[test_indices], static_raw[test_indices]
    )

    model_dir = output_dir / "evaluation_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    cat.save_model(model_dir / "catboost_HM.cbm")
    base.save_lstm(model_dir / "direct_lstm_HM.pt", direct_model, direct_norm, direct_epoch, final_args)
    base.save_lstm(model_dir / "residual_lstm_HM.pt", residual_model, residual_norm, residual_epoch, final_args)
    (model_dir / "tabular_stats_HM.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    frame = test_keys[[*KEYS, "year"]].copy()
    frame["actual_HM"] = hm_y[test_indices]
    frame["catboost_HM"] = cat_test
    frame["direct_lstm_HM"] = direct_test
    frame["residual_lstm_HM"] = residual_test
    frame["sequence_missing_cells"] = missing_counts[test_indices]
    return frame


def fit_hm_calibrator(method: str, frame: pd.DataFrame) -> object:
    y = frame.actual_HM.to_numpy(dtype=float)
    raw = frame.HM_CatResidualRaw.to_numpy(dtype=float)
    if method == "raw":
        return None
    if method == "offset":
        return float(np.mean(y - raw))
    if method == "affine":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(raw.reshape(-1, 1), y)
        return model
    if method == "affine_with_oof_TA":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(frame[["HM_CatResidualRaw", "TA_LatestOOFCalibrated"]], y)
        return model
    raise ValueError(f"unknown HM calibration method: {method}")


def apply_hm_calibrator(method: str, calibrator: object, frame: pd.DataFrame) -> np.ndarray:
    raw = frame.HM_CatResidualRaw.to_numpy(dtype=float)
    if method == "raw":
        prediction = raw
    elif method == "offset":
        prediction = raw + float(calibrator)
    elif method == "affine":
        prediction = calibrator.predict(raw.reshape(-1, 1))
    elif method == "affine_with_oof_TA":
        prediction = calibrator.predict(frame[["HM_CatResidualRaw", "TA_LatestOOFCalibrated"]])
    else:
        raise ValueError(f"unknown HM calibration method: {method}")
    return np.clip(np.asarray(prediction, dtype=float), 0.0, 100.0)


def select_hm_calibration(oof: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select calibration by forward-year meta-validation, not in-sample OOF fit."""
    methods = ["raw", "offset", "affine", "affine_with_oof_TA"]
    years = sorted(int(year) for year in oof.year.unique())
    parts = []
    for validation_year in years[1:]:
        train = oof[oof.year.lt(validation_year)]
        validation = oof[oof.year.eq(validation_year)].copy()
        for method in methods:
            calibrator = fit_hm_calibrator(method, train)
            part = validation[[*KEYS, "year", "actual_HM"]].copy()
            part["method"] = method
            part["prediction_HM"] = apply_hm_calibrator(method, calibrator, validation)
            parts.append(part)
    predictions = pd.concat(parts, ignore_index=True)
    rows = []
    for method, part in predictions.groupby("method", sort=False):
        rows.append({"split": "forward_meta_oof", "method": method, **base.diagnostic_metrics(
            part.actual_HM.to_numpy(dtype=float), part.prediction_HM.to_numpy(dtype=float)
        )})
        for year, year_part in part.groupby("year"):
            rows.append({"split": f"meta_oof_{int(year)}", "method": method, **base.diagnostic_metrics(
                year_part.actual_HM.to_numpy(dtype=float), year_part.prediction_HM.to_numpy(dtype=float)
            )})
    metrics = pd.DataFrame(rows)
    selected = str(metrics[metrics.split.eq("forward_meta_oof")].sort_values("RMSE").iloc[0].method)
    return selected, metrics


def add_hm_predictions(
    oof: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame, object]:
    scale, _ = taexp.optimize_scale(
        oof.actual_HM.to_numpy(dtype=float),
        oof.catboost_HM.to_numpy(dtype=float),
        oof.residual_lstm_HM.to_numpy(dtype=float),
        args.residual_scale_step,
        args.residual_scale_max,
    )
    oof = oof.copy()
    test = test.copy()
    for frame in [oof, test]:
        frame["HM_CatBoost"] = np.clip(frame.catboost_HM.to_numpy(dtype=float), 0.0, 100.0)
        frame["HM_DirectLSTM"] = np.clip(frame.direct_lstm_HM.to_numpy(dtype=float), 0.0, 100.0)
        frame["HM_CatResidualRaw"] = np.clip(
            frame.catboost_HM.to_numpy(dtype=float)
            + scale * frame.residual_lstm_HM.to_numpy(dtype=float),
            0.0, 100.0,
        )

    selected_method, selection_metrics = select_hm_calibration(oof)
    final_calibrator = fit_hm_calibrator(selected_method, oof)
    oof["HM_SelectedCalibrated"] = apply_hm_calibrator(selected_method, final_calibrator, oof)
    test["HM_SelectedCalibrated"] = apply_hm_calibrator(selected_method, final_calibrator, test)

    rows = []
    for split, frame in [("rolling_oof_refit", oof), (f"test_{args.test_year}", test)]:
        for model in ["HM_CatBoost", "HM_DirectLSTM", "HM_CatResidualRaw", "HM_SelectedCalibrated"]:
            rows.append({"split": split, "model": model, **base.diagnostic_metrics(
                frame.actual_HM.to_numpy(dtype=float), frame[model].to_numpy(dtype=float)
            )})
    metrics = pd.concat([selection_metrics.rename(columns={"method": "model"}), pd.DataFrame(rows)], ignore_index=True)
    recipe: dict[str, object] = {
        "residual_scale": scale,
        "calibration_candidates": ["raw", "offset", "affine", "affine_with_oof_TA"],
        "selection_split": "forward meta-OOF: fit earlier OOF years, validate next OOF year",
        "selected_calibration": selected_method,
        "TA_meta_feature": "TA_LatestOOFCalibrated (model prediction only)",
        "output_clip": [0.0, 100.0],
    }
    return oof, test, recipe, metrics, final_calibrator


def subgroup_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["humidity_band"] = pd.cut(
        work.actual_HM, [-np.inf, 50.0, 70.0, 85.0, np.inf],
        labels=["<50", "50-<70", "70-<85", ">=85"], right=False,
    )
    work["sequence_group"] = np.where(work.sequence_missing_cells.eq(0), "complete", "incomplete")
    rows = []
    for dimension in ["humidity_band", "sequence_group"]:
        for group, part in work.groupby(dimension, observed=True):
            for model in ["HM_CatBoost", "HM_CatResidualRaw", "HM_SelectedCalibrated"]:
                rows.append({"dimension": dimension, "group": str(group), "model": model,
                             **base.diagnostic_metrics(part.actual_HM.to_numpy(dtype=float),
                                                       part[model].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def competition_scores(ta_test: pd.DataFrame, hm_test: pd.DataFrame) -> pd.DataFrame:
    merged = hm_test.merge(
        ta_test[[*KEYS, "actual_TA"]],
        on=KEYS, how="inner", validate="1:1",
    )
    rows = []
    definitions = [
        ("before_TA_calibration_plus_HM_raw", "TA_PooledOOF", "HM_CatResidualRaw"),
        ("TA_calibrated_plus_HM_raw", "TA_LatestOOFCalibrated", "HM_CatResidualRaw"),
        ("final_selected", "TA_LatestOOFCalibrated", "HM_SelectedCalibrated"),
    ]
    for name, ta_column, hm_column in definitions:
        ta_score = base.rmse(merged.actual_TA.to_numpy(dtype=float), merged[ta_column].to_numpy(dtype=float))
        hm_score = base.rmse(merged.actual_HM.to_numpy(dtype=float), merged[hm_column].to_numpy(dtype=float))
        rows.append({
            "system": name,
            "TA_model": ta_column,
            "HM_model": hm_column,
            "RMSE_TA": ta_score,
            "RMSE_HM": hm_score,
            "competition_score": ta_score + 0.1 * hm_score,
            "formula": "RMSE_TA + 0.1 * RMSE_HM",
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if sorted(set(args.oof_years)) != args.oof_years:
        raise ValueError("--oof-years must be unique and ascending")
    if max(args.oof_years) >= args.test_year:
        raise ValueError("OOF years must precede --test-year")
    base.seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ta_oof_raw = clean_prediction_frame(args.ta_oof_csv, set(args.oof_years))
    ta_test_raw = clean_prediction_frame(args.ta_test_csv, {args.test_year})
    ta_oof, ta_test, ta_recipe, ta_metrics = calibrate_ta(ta_oof_raw, ta_test_raw)
    print("\n[TA OOF CALIBRATION]", flush=True)
    print(json.dumps(ta_recipe, ensure_ascii=False, indent=2), flush=True)
    if not ta_recipe["test_improved"]:
        raise RuntimeError("latest-year OOF TA calibration did not improve the held-out test RMSE; HM training skipped")

    master_raw = pd.read_csv(args.master_csv)
    master_solar = base.add_solar_features(master_raw)
    if master_solar.duplicated(KEYS).any():
        raise ValueError("master CSV has duplicate Date/STN_ID rows")
    shortterm = pd.read_csv(args.shortterm_long_csv)
    x_raw, static_raw, _ta_sequence_y, sequence_keys, missing_counts = base.build_sequences(shortterm)
    hm_y = hm_labels_for_sequences(shortterm, sequence_keys)

    oof_parts = []
    summaries = []
    histories = []
    for year in args.oof_years:
        part, summary, history = train_hm_oof_fold(
            year, master_solar, x_raw, static_raw, hm_y, sequence_keys, missing_counts, args
        )
        oof_parts.append(part)
        summaries.append(summary)
        histories.append(history)
    hm_oof = pd.concat(oof_parts, ignore_index=True)
    fold_summary = pd.DataFrame(summaries)
    history = pd.concat(histories, ignore_index=True)
    hm_test = train_hm_final(
        master_solar, x_raw, static_raw, hm_y, sequence_keys, missing_counts,
        fold_summary, args, output_dir,
    )

    hm_oof = hm_oof.merge(
        ta_oof[[*KEYS, "TA_PooledOOF", "TA_LatestOOFCalibrated"]],
        on=KEYS, how="inner", validate="1:1",
    )
    hm_test = hm_test.merge(
        ta_test[[*KEYS, "TA_PooledOOF", "TA_LatestOOFCalibrated"]],
        on=KEYS, how="inner", validate="1:1",
    )
    expected_oof = len(pd.concat(oof_parts, ignore_index=True))
    if len(hm_oof) != expected_oof or len(hm_test) != len(ta_test):
        raise ValueError("TA/HM prediction keys do not fully align")

    hm_oof, hm_test, hm_recipe, hm_metrics, hm_calibrator = add_hm_predictions(
        hm_oof, hm_test, args
    )
    scores = competition_scores(ta_test, hm_test)
    subgroups = subgroup_metrics(hm_test)

    import joblib

    joblib.dump(
        {"method": hm_recipe["selected_calibration"], "model": hm_calibrator},
        output_dir / "evaluation_models" / "hm_meta_calibrator.joblib",
    )
    ta_oof.to_csv(output_dir / "ta_oof_calibrated_predictions.csv", index=False)
    ta_test.to_csv(output_dir / f"ta_test_{args.test_year}_calibrated_predictions.csv", index=False)
    ta_metrics.to_csv(output_dir / "ta_calibration_metrics.csv", index=False)
    hm_oof.to_csv(output_dir / "hm_rolling_oof_predictions.csv", index=False)
    hm_test.to_csv(output_dir / f"hm_test_{args.test_year}_predictions.csv", index=False)
    hm_metrics.to_csv(output_dir / "hm_metrics.csv", index=False)
    scores.to_csv(output_dir / "competition_scores.csv", index=False)
    subgroups.to_csv(output_dir / f"hm_subgroup_metrics_{args.test_year}.csv", index=False)
    fold_summary.to_csv(output_dir / "hm_fold_training_summary.csv", index=False)
    history.to_csv(output_dir / "hm_lstm_training_history.csv", index=False)
    recipe = {"TA": ta_recipe, "HM": hm_recipe}
    (output_dir / "calibration_recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compliance = {
        "allowed_inputs": [
            "GK-2A LE1B KO 16 channels",
            "official station LAT/LON/ALT",
            "date/time/coordinate-derived calendar and solar values",
        ],
        "ASOS_usage": "TA and HM are labels only",
        "ASOS_lag_or_recent_observation_input": False,
        "HM_uses_observed_TA_as_input": False,
        "HM_TA_meta_feature": "rolling OOF/frozen model-predicted TA only",
        "external_weather_or_static_geography": False,
        "HM_output_clipped_to_0_100": True,
        "score_formula": "RMSE_TA + 0.1 * RMSE_HM",
        "evaluation_note": "2025 is a retrospective holdout used repeatedly during development, not a pristine leaderboard estimate.",
    }
    (output_dir / "rule_compliance.json").write_text(
        json.dumps(compliance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best = scores.loc[scores.system.eq("final_selected")].iloc[0].to_dict()
    summary = {
        "oof_years": args.oof_years,
        "test_year": args.test_year,
        "TA_calibration": ta_recipe,
        "HM_recipe": hm_recipe,
        "final_test_metrics": best,
        "all_competition_scores": scores.to_dict(orient="records"),
        "rule_compliance": compliance,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[HM METRICS]", flush=True)
    print(hm_metrics.to_string(index=False), flush=True)
    print("\n[COMPETITION SCORE]", flush=True)
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
