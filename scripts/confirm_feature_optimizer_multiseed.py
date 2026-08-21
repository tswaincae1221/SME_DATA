#!/usr/bin/env python3
"""Confirm a frozen engineered residual branch against multiple baseline seeds.

Feature selection is read from a completed optimizer summary and is not
repeated per seed.  Only the residual Ridge is refit to each seed's rolling
OOF baseline predictions, then its OOF-selected correction weight is applied
once to the report year.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import experiment_feature_engineering_optimizer as optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--optimizer-summary", required=True)
    parser.add_argument("--baseline-dirs", nargs="+", required=True)
    parser.add_argument("--seed-labels", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--report-year", type=int, default=2025)
    parser.add_argument("--ensemble-step", type=float, default=0.02)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.baseline_dirs) != len(args.seed_labels):
        raise ValueError("--baseline-dirs and --seed-labels must have the same length")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.optimizer_summary).read_text(encoding="utf-8"))
    long_frame = optimizer.validate_long(pd.read_csv(args.shortterm_long_csv))
    selection_years = sorted(set(args.selection_years))
    tables = {
        year: optimizer.build_augmented_table(long_frame, year - 1, None)
        for year in selection_years
    }
    report_table = optimizer.build_augmented_table(long_frame, args.report_year - 1, None)

    rows = []
    prediction_parts = []
    for seed_label, baseline_dir_value in zip(args.seed_labels, args.baseline_dirs):
        baseline_dir = Path(baseline_dir_value).expanduser().resolve()
        for target in ["TA", "HM"]:
            target_summary = summary["targets"][target]
            features = list(target_summary["selected_features"])
            alpha = float(target_summary["selection_oof"]["hyperparameter"])
            oof = optimizer.load_baseline(
                str(baseline_dir / f"step0_current_baseline_{target}_oof.csv"), target, "oof"
            )
            test = optimizer.load_baseline(
                str(baseline_dir / f"step0_current_baseline_{target}_test.csv"), target, "test"
            )
            effective_years = [year for year in selection_years if year > int(oof.year.min())]
            eval_args = argparse.Namespace(
                estimator="ridge", ridge_alphas=[alpha], cat_l2=8.0,
                cat_iterations=600, cat_depth=7, cat_learning_rate=0.035,
                threads=args.threads, seed=args.seed,
            )
            evaluation = optimizer.evaluate(
                tables, effective_years, features, target, eval_args,
                hyperparameter=alpha, baseline_oof=oof,
            )
            final, engineered_metrics = optimizer.final_fit_predict(
                report_table, target, features, alpha, args.report_year, eval_args,
                oof, test,
            )
            ensemble, ensemble_metrics = optimizer.ensemble_with_baseline(
                evaluation, final, target, oof, test, args.ensemble_step
            )
            rows.append({
                "seed": seed_label,
                "target": target,
                "feature_count": len(features),
                "residual_alpha": alpha,
                "residual_oof_RMSE": evaluation.pooled_rmse,
                "baseline_weight": ensemble_metrics["baseline_weight"],
                "engineered_weight": ensemble_metrics["engineered_weight"],
                "ensemble_oof_RMSE": ensemble_metrics["selection_OOF_RMSE"],
                "engineered_report_RMSE": engineered_metrics["RMSE"],
                "ensemble_report_RMSE": ensemble_metrics["report_RMSE"],
                "ensemble_report_MAE": ensemble_metrics["report_MAE"],
                "ensemble_report_bias": ensemble_metrics["report_bias"],
            })
            part = ensemble.copy()
            part["seed"] = seed_label
            part["target"] = target
            prediction_parts.append(part)

    metrics = pd.DataFrame(rows)
    pivot = metrics.pivot(index="seed", columns="target", values="ensemble_report_RMSE").reset_index()
    pivot["competition_score"] = pivot["TA"] + 0.1 * pivot["HM"]
    summary_metrics = pd.DataFrame([
        {
            "metric": column,
            "mean": float(pivot[column].mean()),
            "std": float(pivot[column].std(ddof=1)),
            "min": float(pivot[column].min()),
            "max": float(pivot[column].max()),
        }
        for column in ["TA", "HM", "competition_score"]
    ])
    metrics.to_csv(output_dir / "multiseed_target_metrics.csv", index=False)
    pivot.to_csv(output_dir / "multiseed_competition_scores.csv", index=False)
    summary_metrics.to_csv(output_dir / "multiseed_summary.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_csv(
        output_dir / "multiseed_report_predictions.csv", index=False
    )
    print(metrics.to_string(index=False), flush=True)
    print("\n", pivot.to_string(index=False), flush=True)
    print("\n", summary_metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
