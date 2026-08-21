#!/usr/bin/env python3
"""Confirm the frozen team-feature HM branch across multiple seeds.

The discovery run showed that same-date spatial features were accepted in all
three seeds and physical channel relations in two of three.  This confirmation
therefore freezes one common 54-feature HM CatBoost branch:

    existing 40 + physical channel relations 6 + same-date spatial context 8

Station identity and the predicted-TA chain are not included.  No feature is
reselected per seed.  Each new branch is blended with the matching current HM
OOF prediction, and the final prediction averages the seed-specific accepted
ensembles.  TA is the unchanged mean of the current seed predictions.
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

import experiment_team_catboost_feature_ensemble as experiment  # noqa: E402
import train_dualbranch_ta_baseline as base  # noqa: E402


FROZEN_HM_FEATURES = [
    *base.TABULAR_FEATURES,
    *experiment.PHYSICAL_FEATURES,
    *experiment.SPATIAL_FEATURES,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--station-list", required=True)
    parser.add_argument("--current-multiseed-predictions", required=True)
    parser.add_argument("--baseline-dirs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oof-years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--report-year", type=int, default=2025)
    parser.add_argument("--validation-start-mmdd", type=int, default=824)
    parser.add_argument("--validation-end-mmdd", type=int, default=830)
    parser.add_argument("--cat-iterations", type=int, default=500)
    parser.add_argument("--cat-depth", type=int, default=7)
    parser.add_argument("--cat-learning-rate", type=float, default=0.04)
    parser.add_argument("--cat-l2", type=float, default=8.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weight-step", type=float, default=0.02)
    parser.add_argument("--min-improvement-ta", type=float, default=0.005)
    parser.add_argument("--min-improvement-hm", type=float, default=0.05)
    parser.add_argument("--min-blend-improvement-ta", type=float, default=0.005)
    parser.add_argument("--min-blend-improvement-hm", type=float, default=0.02)
    parser.add_argument("--latest-tolerance-ta", type=float, default=0.01)
    parser.add_argument("--latest-tolerance-hm", type=float, default=0.10)
    return parser.parse_args()


def current_hm_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "step0_current_baseline_HM_oof.csv")
    required = [*experiment.KEYS, "year", "actual_HM", "prediction_HM"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path}: current HM OOF is missing {missing}")
    return frame[required].rename(columns={"prediction_HM": "current_best"})


def current_seed_test(multiseed: pd.DataFrame, seed: int, target: str) -> pd.DataFrame:
    frame = multiseed[(multiseed.seed.eq(seed)) & (multiseed.target.eq(target))].copy()
    actual = f"actual_{target}"
    prediction = f"ensemble_{target}"
    required = [*experiment.KEYS, "year", actual, prediction]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"seed {seed}/{target}: current predictions are missing {missing}")
    return frame[required].rename(columns={prediction: "current_best"})


def metric(actual: pd.Series, prediction: pd.Series) -> float:
    return experiment.rmse(actual.to_numpy(float), prediction.to_numpy(float))


def main() -> None:
    args = parse_args()
    if len(args.baseline_dirs) != len(args.seeds):
        raise ValueError("--baseline-dirs must align one-for-one with --seeds")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_predictions = pd.read_csv(args.current_multiseed_predictions)
    stations = experiment.clean_station_list(args.station_list)
    frame = experiment.engineer_team_features(pd.read_csv(args.master_csv), stations)
    experiment.validate_feature_contract(FROZEN_HM_FEATURES, "HM")

    test_parts = []
    metric_rows = []
    recipes = {}
    for seed, baseline_path in zip(args.seeds, args.baseline_dirs):
        print("\n" + "=" * 100, flush=True)
        print(f"[SEED {seed}] frozen HM branch", flush=True)
        args.seed = int(seed)
        experiment.seed_everything(args.seed)
        candidate = experiment.rolling_oof(
            frame, "HM", "frozen_physical_spatial", FROZEN_HM_FEATURES, [],
            args.oof_years, args,
        )
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        report, importance = experiment.final_fit_predict(
            frame, candidate, args.report_year, args, seed_dir
        )
        current_oof = current_hm_oof(Path(baseline_path).expanduser().resolve())
        current_test = current_seed_test(current_predictions, seed, "HM")
        ensemble_oof, ensemble_test, recipe = experiment.blend_with_current(
            "HM", current_oof, current_test, candidate.predictions, report, args
        )
        candidate.predictions.to_csv(seed_dir / "frozen_HM_oof.csv", index=False)
        candidate.fold_metrics.to_csv(seed_dir / "frozen_HM_fold_metrics.csv", index=False)
        report.to_csv(seed_dir / f"frozen_HM_{args.report_year}.csv", index=False)
        importance.to_csv(seed_dir / "frozen_HM_importance.csv", index=False)
        ensemble_oof.to_csv(seed_dir / "ensemble_HM_oof.csv", index=False)
        ensemble_test.to_csv(seed_dir / f"ensemble_HM_{args.report_year}.csv", index=False)
        ensemble_test["seed"] = seed
        test_parts.append(ensemble_test)
        recipes[str(seed)] = recipe
        metric_rows.append({
            "seed": seed,
            "new_branch_weight": recipe["team_feature_catboost_weight"],
            "accepted": recipe["accepted"],
            "OOF_current_RMSE": recipe["OOF_current_RMSE"],
            "OOF_ensemble_RMSE": recipe["OOF_selected_RMSE"],
            "report_current_RMSE": recipe["report_current_RMSE"],
            "report_ensemble_RMSE": recipe["report_ensemble_RMSE"],
            "report_improvement": recipe["report_improvement"],
        })

    by_seed = pd.concat(test_parts, ignore_index=True)
    averaged_hm = by_seed.groupby(experiment.KEYS, as_index=False).agg(
        actual_HM=("actual_HM", "first"),
        current_HM=("current_best", "mean"),
        enhanced_HM=("ensemble_HM", "mean"),
    )
    ta_parts = [current_seed_test(current_predictions, seed, "TA") for seed in args.seeds]
    averaged_ta = pd.concat(ta_parts, ignore_index=True).groupby(experiment.KEYS, as_index=False).agg(
        actual_TA=("actual_TA", "first"),
        enhanced_TA=("current_best", "mean"),
    )
    final = averaged_ta.merge(averaged_hm, on=experiment.KEYS, how="inner", validate="1:1")
    ta_rmse = metric(final.actual_TA, final.enhanced_TA)
    current_hm_rmse = metric(final.actual_HM, final.current_HM)
    enhanced_hm_rmse = metric(final.actual_HM, final.enhanced_HM)
    current_score = ta_rmse + 0.1 * current_hm_rmse
    enhanced_score = ta_rmse + 0.1 * enhanced_hm_rmse

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "frozen_multiseed_metrics.csv", index=False)
    by_seed.to_csv(output_dir / f"frozen_multiseed_HM_{args.report_year}_by_seed.csv", index=False)
    final.to_csv(output_dir / f"frozen_multiseed_final_{args.report_year}.csv", index=False)
    summary = {
        "frozen_feature_count": len(FROZEN_HM_FEATURES),
        "frozen_features": FROZEN_HM_FEATURES,
        "seeds": args.seeds,
        "accepted_seed_count": int(metrics.accepted.sum()),
        "recipes": recipes,
        "report_metrics": {
            "TA_RMSE_unchanged_multiseed_average": ta_rmse,
            "HM_RMSE_current_multiseed_average": current_hm_rmse,
            "HM_RMSE_enhanced_multiseed_average": enhanced_hm_rmse,
            "HM_RMSE_improvement": current_hm_rmse - enhanced_hm_rmse,
            "competition_score_current": current_score,
            "competition_score_enhanced": enhanced_score,
            "competition_score_improvement": current_score - enhanced_score,
        },
        "selection": {
            "feature_groups_reselected_per_seed": False,
            "blend_weight_selected_from": "matching 2022-2024 rolling OOF",
            "report_year_used_for_selection": False,
            "catboost_hyperparameters_tuned": False,
        },
    }
    (output_dir / "frozen_multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nOutputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
