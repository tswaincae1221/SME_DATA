#!/usr/bin/env python3
"""Refit the frozen 2026 TA/HM ensemble and create an inference-only bundle.

This script is training-only and must be run before the Kaggle submission.
The generated bundle contains no ASOS rows; it contains only fitted models,
normalisation statistics, frozen recipes, and inference source files.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import confirm_team_catboost_feature_ensemble_multiseed as frozen_team  # noqa: E402
import experiment_feature_dedup_ablation as dedup  # noqa: E402
import experiment_feature_engineering_optimizer as optimizer  # noqa: E402
import experiment_ta_literature_features as literature  # noqa: E402
import experiment_team_catboost_feature_ensemble as team  # noqa: E402
import train_dualbranch_ta_baseline as base  # noqa: E402


KEYS = ["Date", "STN_ID"]
SEEDS = [42, 43, 44]
TA_RIDGE_FEATURES = [
    "VI004_last", "VI006_last", "VI008_last", "NR013_last", "NR016_last",
    "SW038_last", "WV063_last", "WV069_last", "WV073_last", "IR096_last",
    "SW038_mean_z", "WV069_mean_z", "WV073_mean_z", "IR123_mean_z",
    "clean_window_105_112_mean", "split_window_112_123_mean",
    "window_105_123_mean", "ozone_window_096_112_mean",
    "cloud_phase_087_112_mean", "low_wv_window_073_112_mean",
    "mid_wv_window_069_112_mean", "fog_window_038_112_mean",
    "co2_window_133_112_mean",
]

# All values were selected on rolling 2022-2024 OOF predictions.  They are
# frozen here; 2025 and 2026 are never used to retune a blend weight.
RECIPES = {
    "42": {
        "TA": {"residual_scale": 1.58, "pooled_offset": 0.59916548457637,
               "latest_offset": 1.6051019415143006, "ridge_weight": 0.16,
               "cat_iterations": 494, "lstm_epochs": 7},
        "HM": {"residual_scale": 1.22, "team_weight": 0.12,
               "cat_iterations": 496, "lstm_epochs": 9},
    },
    "43": {
        "TA": {"residual_scale": 1.58, "pooled_offset": 0.6663080883469205,
               "latest_offset": 1.601419387940055, "ridge_weight": 0.18,
               "cat_iterations": 466, "lstm_epochs": 4},
        "HM": {"residual_scale": 1.30, "team_weight": 0.14,
               "cat_iterations": 500, "lstm_epochs": 15},
    },
    "44": {
        "TA": {"residual_scale": 1.80, "pooled_offset": 0.661822504766966,
               "latest_offset": 1.6347237333174751, "ridge_weight": 0.16,
               "cat_iterations": 474, "lstm_epochs": 3},
        "HM": {"residual_scale": 1.30, "team_weight": 0.18,
               "cat_iterations": 499, "lstm_epochs": 9},
    },
}

JUNE_RECIPE = {
    "selection_years": [2022, 2023, 2024, 2025],
    "selection_window_mmdd": [624, 630],
    "ta_catboost_weight": 0.78,
    "ta_ridge_weight": 0.22,
    "ta_offset": 0.5421463057934491,
    "hm_base_catboost_weight": 0.36,
    "hm_spatial_catboost_weight": 0.64,
    "selection_oof_TA_RMSE": 2.504229875760419,
    "selection_oof_HM_RMSE": 13.317086783634018,
    "selection_oof_score": 3.8359385541238207,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--station-list", required=True)
    parser.add_argument(
        "--baseline-oof-dirs", nargs=3, required=True,
        help="Seed 42/43/44 feature-dedup output directories containing step0 TA OOF CSVs",
    )
    parser.add_argument("--output-dir", required=True)
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


def model_args(seed: int, threads: int, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=48, lstm_layers=1, dropout=0.15, batch_size=64,
        learning_rate=5e-4, weight_decay=1e-3,
        cat_depth=7, cat_learning_rate=0.04, cat_l2=8.0,
        threads=threads, device=device, seed=seed,
    )


def save_residual_lstm(path: Path, model, normalization: dict[str, np.ndarray],
                       epochs: int, target: str, args: SimpleNamespace) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(), "target": target,
            "input_size": 32, "static_size": len(base.LSTM_STATIC_FEATURES),
            "hidden_size": args.hidden_size, "lstm_layers": args.lstm_layers,
            "dropout": args.dropout, "epochs": int(epochs),
            "channels": base.CHANNELS,
            "static_features": base.LSTM_STATIC_FEATURES,
        },
        path,
    )
    np.savez_compressed(path.with_name(path.stem + "_normalization.npz"), **normalization)


def sequence_labels(long_frame: pd.DataFrame, keys: pd.DataFrame, target: str) -> np.ndarray:
    labels = long_frame.loc[
        pd.to_numeric(long_frame["TimeKST"]).eq(1400), [*KEYS, target]
    ].copy()
    labels["Date"] = pd.to_numeric(labels["Date"]).round().astype("int64")
    labels["STN_ID"] = pd.to_numeric(labels["STN_ID"]).round().astype("int64")
    return keys[KEYS].merge(labels, on=KEYS, how="left", validate="1:1")[target].to_numpy(float)


def fit_sequence_stats(long_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    year = pd.to_numeric(long_frame["Date"]).round().astype("int64") // 10000
    train = long_frame[year.le(2025)]
    means = train[base.CHANNELS].mean(skipna=True)
    stds = train[base.CHANNELS].std(skipna=True).replace(0.0, 1.0).fillna(1.0)
    return {
        "mean": {name: float(means[name]) for name in base.CHANNELS},
        "std": {name: float(stds[name]) for name in base.CHANNELS},
    }


def prepare_training(master_path: str, shortterm_path: str, station_path: str):
    stations = team.clean_station_list(station_path)
    master_raw = pd.read_csv(master_path)
    master_official = team.apply_official_coordinates(master_raw, stations)
    master_solar = base.add_solar_features(master_official)
    tabular_stats = base.fit_channel_stats(master_solar, 2025)
    master_engineered = base.engineer_tabular(master_solar, tabular_stats)

    shortterm = optimizer.validate_long(pd.read_csv(shortterm_path))
    shortterm = shortterm.drop(columns=[c for c in ["LAT", "LON", "ALT"] if c in shortterm]).merge(
        stations, on="STN_ID", how="left", validate="m:1"
    )
    x_raw, static_raw, _, sequence_keys, _ = base.build_sequences(shortterm)
    labels = {target: sequence_labels(shortterm, sequence_keys, target) for target in ["TA", "HM"]}
    return (
        stations, master_raw, master_engineered, tabular_stats, shortterm,
        x_raw, static_raw, sequence_keys, labels,
    )


def train_base_seed(
    seed: int, seed_dir: Path, master: pd.DataFrame, shortterm: pd.DataFrame,
    x_raw: np.ndarray, static_raw: np.ndarray, sequence_keys: pd.DataFrame,
    labels: dict[str, np.ndarray], threads: int, device: str,
) -> None:
    for target in ["TA", "HM"]:
        recipe = RECIPES[str(seed)][target]
        target_offset = 0 if target == "TA" else 10000
        args = model_args(seed + target_offset + 2026, threads, device)
        seed_everything(args.seed)
        train = master[master[target].notna()].copy()
        cat = dedup.train_catboost_fixed(
            train, tuple(base.TABULAR_FEATURES), target, recipe["cat_iterations"], args
        )
        cat.save_model(seed_dir / f"base_catboost_{target}.cbm")

        candidates = sequence_keys[KEYS].copy()
        candidates["_sequence_index"] = np.arange(len(candidates), dtype=int)
        candidates = candidates[np.isfinite(labels[target])]
        aligned = candidates.merge(
            master[[*KEYS, *base.TABULAR_FEATURES]], on=KEYS,
            how="inner", validate="1:1",
        )
        if aligned.empty:
            raise ValueError(f"seed {seed}/{target}: no aligned sequence/master rows")
        indices = aligned["_sequence_index"].to_numpy(dtype=int)
        cat_sequence = np.asarray(cat.predict(aligned[base.TABULAR_FEATURES]), dtype=float)
        residual = labels[target][indices] - cat_sequence
        lstm, normalization, _ = dedup.train_lstm_fixed(
            x_raw[indices], static_raw[indices], residual,
            int(recipe["lstm_epochs"]), args,
        )
        save_residual_lstm(
            seed_dir / f"residual_lstm_{target}.pt", lstm, normalization,
            int(recipe["lstm_epochs"]), target, args,
        )


def train_ta_residual_ridge(
    seed: int, seed_dir: Path, long_frame: pd.DataFrame, oof_dir: Path,
) -> None:
    table = optimizer.build_augmented_table(long_frame, 2025, None)
    oof = pd.read_csv(oof_dir / "step0_current_baseline_TA_oof.csv")
    required = [*KEYS, "year", "actual_TA", "prediction_TA"]
    missing = [column for column in required if column not in oof]
    if missing:
        raise ValueError(f"seed {seed}: baseline OOF missing {missing}")
    train = table.merge(oof[required], on=[*KEYS, "year"], how="inner", validate="1:1")
    train["_residual_target"] = train["actual_TA"] - train["prediction_TA"]
    args = SimpleNamespace(estimator="ridge", threads=1, seed=seed)
    model = optimizer.fit_estimator(
        train, TA_RIDGE_FEATURES, "_residual_target", "ridge", 1000.0, args
    )
    joblib.dump(model, seed_dir / "ta_residual_ridge.joblib")


def train_june_direct_ridge(seed_dir: Path, master: pd.DataFrame) -> None:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train = master[master["TA"].notna()].copy()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=100.0)),
    ])
    model.fit(train[base.TABULAR_FEATURES], train["TA"])
    joblib.dump(model, seed_dir / "june_direct_ridge_TA.joblib")


def train_team_hm_seed(
    seed: int, seed_dir: Path, master_raw: pd.DataFrame,
    stations: pd.DataFrame, threads: int,
) -> None:
    from catboost import CatBoostRegressor

    frame = team.engineer_team_features(master_raw, stations)
    stats = base.fit_channel_stats(frame, 2025)
    engineered = base.engineer_tabular(frame, stats)
    train = engineered[engineered["HM"].notna()].copy()
    model = CatBoostRegressor(
        iterations=500, depth=7, learning_rate=0.04, l2_leaf_reg=8.0,
        loss_function="RMSE", random_seed=seed + 2026,
        thread_count=max(1, threads), verbose=False, allow_writing_files=False,
    )
    model.fit(train[frozen_team.FROZEN_HM_FEATURES], train["HM"])
    model.save_model(seed_dir / "team_spatial_catboost_HM.cbm")


def copy_inference_sources(output_dir: Path) -> None:
    names = [
        "submission_2026_inference.py", "submission_predictor.py",
        "train_dualbranch_ta_baseline.py", "experiment_ta_literature_features.py",
        "experiment_feature_engineering_optimizer.py",
        "experiment_team_catboost_feature_ensemble.py",
        "confirm_team_catboost_feature_ensemble_multiseed.py",
    ]
    for name in names:
        source = SCRIPT_DIR / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        stations, master_raw, master, tabular_stats, shortterm,
        x_raw, static_raw, sequence_keys, labels,
    ) = prepare_training(args.master_csv, args.shortterm_long_csv, args.station_list)

    for seed, oof_value in zip(SEEDS, args.baseline_oof_dirs):
        print(f"[BUNDLE] seed={seed}", flush=True)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_base_seed(
            seed, seed_dir, master, shortterm, x_raw, static_raw,
            sequence_keys, labels, args.threads, args.device,
        )
        train_ta_residual_ridge(seed, seed_dir, shortterm, Path(oof_value))
        train_june_direct_ridge(seed_dir, master)
        train_team_hm_seed(seed, seed_dir, master_raw, stations, args.threads)

    sequence_stats = fit_sequence_stats(shortterm)
    manifest = {
        "bundle_version": "2026.08.21-v1",
        "training_end_year": 2025,
        "seeds": SEEDS,
        "channels": base.CHANNELS,
        "times_kst": base.EXPECTED_TIMES.astype(int).tolist(),
        "tabular_features": base.TABULAR_FEATURES,
        "lstm_static_features": base.LSTM_STATIC_FEATURES,
        "ta_ridge_features": TA_RIDGE_FEATURES,
        "team_hm_features": frozen_team.FROZEN_HM_FEATURES,
        "tabular_channel_stats": tabular_stats,
        "sequence_channel_stats": sequence_stats,
        "recipes": RECIPES,
        "sequence_supported_mmdd": [824, 830],
        "june_recipe": JUNE_RECIPE,
        "rules_contract": {
            "satellite_path": "/GK2A/LE1B/{channel}/KO/data",
            "asos_inference_inputs": False,
            "direct_year_feature": False,
            "official_station_coordinates": True,
        },
        "historical_report": {
            "period": "2025-08-24..2025-08-30",
            "TA_RMSE": 1.6684495016692684,
            "HM_RMSE": 7.147950918288395,
            "competition_score": 2.383244593498108,
            "note": "retrospective report only; not a guaranteed 2026 leaderboard score",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    copy_inference_sources(output_dir)
    print(f"[DONE] inference-only bundle: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
