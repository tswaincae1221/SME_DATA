#!/usr/bin/env python3
"""Select a June-safe 14:00 ensemble and report once on June 2025.

The 12:00-14:00 LSTM was trained only on Aug 24-30 sequences.  This experiment
therefore selects a separate 14:00 branch on rolling June 24-30 OOF years and
keeps June 2025 locked until all weights are frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import confirm_team_catboost_feature_ensemble_multiseed as frozen  # noqa: E402
import experiment_team_catboost_feature_ensemble as team  # noqa: E402
import train_dualbranch_ta_baseline as base  # noqa: E402


SEEDS = [42, 43, 44]
BASE_ITERATIONS = {
    42: {"TA": 494, "HM": 496},
    43: {"TA": 466, "HM": 500},
    44: {"TA": 474, "HM": 499},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--station-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oof-years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--report-year", type=int, default=2025)
    parser.add_argument("--start-mmdd", type=int, default=624)
    parser.add_argument("--end-mmdd", type=int, default=630)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weight-step", type=float, default=0.02)
    return parser.parse_args()


def ridge_model(alpha: float = 100.0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def cat_model(seed: int, target: str, iterations: int, threads: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations, depth=7, learning_rate=0.04, l2_leaf_reg=8.0,
        loss_function="RMSE", random_seed=seed,
        thread_count=max(1, threads), verbose=False, allow_writing_files=False,
    )


def mask(frame: pd.DataFrame, year: int, start_mmdd: int, end_mmdd: int) -> pd.Series:
    mmdd = frame.Date.astype("int64") % 10000
    return frame.year.eq(year) & mmdd.between(start_mmdd, end_mmdd)


def rmse(actual, prediction) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean((prediction - actual) ** 2)))


def optimize_weight(actual, first, second, step: float) -> tuple[float, float]:
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    scores = [rmse(actual, weight * first + (1.0 - weight) * second) for weight in weights]
    index = int(np.argmin(scores))
    return float(weights[index]), float(scores[index])


def fit_fold(frame: pd.DataFrame, seed: int, year: int, args: argparse.Namespace) -> pd.DataFrame:
    engineered = base.engineer_tabular(frame, base.fit_channel_stats(frame, year - 1))
    validation = engineered[mask(engineered, year, args.start_mmdd, args.end_mmdd)].copy()
    output = validation[["Date", "STN_ID", "year", "TA", "HM"]].copy()
    for target in ["TA", "HM"]:
        train = engineered[engineered.year.le(year - 1) & engineered[target].notna()].copy()
        valid = validation[validation[target].notna()].copy()
        cat = cat_model(
            seed + (0 if target == "TA" else 10000) + year,
            target, BASE_ITERATIONS[seed][target], args.threads,
        )
        cat.fit(train[base.TABULAR_FEATURES], train[target])
        ridge = ridge_model(100.0)
        ridge.fit(train[base.TABULAR_FEATURES], train[target])
        prediction = valid[["Date", "STN_ID"]].copy()
        prediction[f"cat_{target}"] = cat.predict(valid[base.TABULAR_FEATURES])
        prediction[f"ridge_{target}"] = ridge.predict(valid[base.TABULAR_FEATURES])
        output = output.merge(prediction, on=["Date", "STN_ID"], how="left", validate="1:1")

    train_hm = engineered[engineered.year.le(year - 1) & engineered.HM.notna()].copy()
    valid_hm = validation[validation.HM.notna()].copy()
    spatial = cat_model(seed + year, "HM", 500, args.threads)
    spatial.fit(train_hm[frozen.FROZEN_HM_FEATURES], train_hm.HM)
    spatial_prediction = valid_hm[["Date", "STN_ID"]].copy()
    spatial_prediction["spatial_HM"] = spatial.predict(valid_hm[frozen.FROZEN_HM_FEATURES])
    return output.merge(spatial_prediction, on=["Date", "STN_ID"], how="left", validate="1:1")


def select_recipe(oof: pd.DataFrame, step: float) -> dict[str, float]:
    ta = oof.dropna(subset=["TA", "cat_TA", "ridge_TA"])
    cat_weight, ta_rmse = optimize_weight(ta.TA, ta.cat_TA, ta.ridge_TA, step)
    ta_blend = cat_weight * ta.cat_TA.to_numpy(float) + (1.0 - cat_weight) * ta.ridge_TA.to_numpy(float)
    ta_offset = float(np.mean(ta.TA.to_numpy(float) - ta_blend))
    ta_calibrated_rmse = rmse(ta.TA, ta_blend + ta_offset)

    hm = oof.dropna(subset=["HM", "cat_HM", "spatial_HM"])
    current_weight, hm_rmse = optimize_weight(hm.HM, hm.cat_HM, hm.spatial_HM, step)
    return {
        "ta_catboost_weight": cat_weight,
        "ta_ridge_weight": 1.0 - cat_weight,
        "ta_offset": ta_offset,
        "ta_oof_RMSE_uncalibrated": ta_rmse,
        "ta_oof_RMSE": ta_calibrated_rmse,
        "hm_base_catboost_weight": current_weight,
        "hm_spatial_catboost_weight": 1.0 - current_weight,
        "hm_oof_RMSE": hm_rmse,
    }


def apply_recipe(frame: pd.DataFrame, recipe: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    out["prediction_TA"] = (
        recipe["ta_catboost_weight"] * out.cat_TA
        + recipe["ta_ridge_weight"] * out.ridge_TA
        + recipe["ta_offset"]
    )
    out["prediction_HM"] = np.clip(
        recipe["hm_base_catboost_weight"] * out.cat_HM
        + recipe["hm_spatial_catboost_weight"] * out.spatial_HM,
        0.0, 100.0,
    )
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stations = team.clean_station_list(args.station_list)
    frame = team.engineer_team_features(pd.read_csv(args.master_csv), stations)

    recipes, report_parts, fold_parts = {}, [], []
    for seed in SEEDS:
        print(f"[SEED {seed}] rolling June OOF", flush=True)
        oof = pd.concat(
            [fit_fold(frame, seed, year, args) for year in args.oof_years], ignore_index=True
        )
        recipe = select_recipe(oof, args.weight_step)
        recipes[str(seed)] = recipe
        oof = apply_recipe(oof, recipe)
        oof["seed"] = seed
        fold_parts.append(oof)

        report = fit_fold(frame, seed, args.report_year, args)
        report = apply_recipe(report, recipe)
        report["seed"] = seed
        report_parts.append(report)
        print(json.dumps(recipe, ensure_ascii=False, indent=2), flush=True)

    by_seed = pd.concat(report_parts, ignore_index=True)
    final = by_seed.groupby(["Date", "STN_ID"], as_index=False).agg(
        TA=("TA", "first"), HM=("HM", "first"),
        prediction_TA=("prediction_TA", "mean"),
        prediction_HM=("prediction_HM", "mean"),
    ).dropna()
    metrics = {
        "report_period": f"{args.report_year}-{args.start_mmdd:04d}..{args.end_mmdd:04d}",
        "report_year_used_for_selection": False,
        "n": len(final),
        "TA_RMSE": rmse(final.TA, final.prediction_TA),
        "HM_RMSE": rmse(final.HM, final.prediction_HM),
    }
    metrics["competition_score"] = metrics["TA_RMSE"] + 0.1 * metrics["HM_RMSE"]
    summary = {
        "selection_window_mmdd": [args.start_mmdd, args.end_mmdd],
        "oof_years": args.oof_years,
        "seeds": SEEDS,
        "recipes": recipes,
        "locked_report": metrics,
        "lstm_gate": {
            "enabled_mmdd": [824, 830],
            "reason": "historical 13-step LSTM training/validation sequences exist only for Aug 24-30",
        },
    }
    pd.concat(fold_parts, ignore_index=True).to_csv(output_dir / "june_oof_predictions.csv", index=False)
    by_seed.to_csv(output_dir / "june_2025_by_seed.csv", index=False)
    final.to_csv(output_dir / "june_2025_multiseed.csv", index=False)
    (output_dir / "june_submission_recipe.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
