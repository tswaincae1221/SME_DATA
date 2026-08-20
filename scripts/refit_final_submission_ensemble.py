#!/usr/bin/env python3
"""Refit 2019--2024 models and select target-specific 2026 components on 2025.

This is the final chronological fit after the earlier 2019--2023 / 2024 / 2025
model-development experiment.  The most recent fully labelled year, 2025, is
used as the final pre-2026 validation period.  TA and HM components are selected
independently because the competition loss is additive by target and the
historical experiments show different model complementarity for each target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from train_submission_safe_ensemble import (
    CHANNELS,
    NONVISIBLE_CHANNELS,
    STATIC_DATE_FEATURES,
    TARGETS,
    add_date_features,
    align_lstm_predictions,
    blend,
    optimize_weights,
    rmse,
    train_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--lstm-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-end-year", type=int, default=2024)
    parser.add_argument("--validation-year", type=int, default=2025)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lstm_dir = Path(args.lstm_artifact_dir).expanduser().resolve()

    master = add_date_features(pd.read_csv(args.master_csv))
    shortterm = pd.read_csv(args.shortterm_long_csv)
    shortterm = shortterm[pd.to_numeric(shortterm["TimeKST"], errors="coerce").eq(1400)].copy()
    shortterm = add_date_features(shortterm)
    validation_base = shortterm[shortterm["year"].eq(args.validation_year)].copy()
    validation = align_lstm_predictions(validation_base, lstm_dir / "predictions_2025.csv")
    actual = validation[TARGETS].to_numpy(dtype=float)
    lstm_prediction = validation[["lstm_TA", "lstm_HM"]].to_numpy(dtype=float)

    variants = {
        "master_nonvisible": {
            "training": master[master["year"].le(args.train_end_year)].copy(),
            "features": [*NONVISIBLE_CHANNELS, *STATIC_DATE_FEATURES],
        },
        "shortterm_all16": {
            "training": shortterm[shortterm["year"].le(args.train_end_year)].copy(),
            "features": [*CHANNELS, *STATIC_DATE_FEATURES],
        },
    }
    trained: dict[str, dict[str, object]] = {}
    metric_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(TARGETS):
        metric_rows.append(
            {
                "target": target,
                "candidate": "LSTM",
                "variant": "none",
                "lstm_weight": 1.0,
                "catboost_weight": 0.0,
                "RMSE_2025": rmse(actual[:, target_index], lstm_prediction[:, target_index]),
            }
        )

    for name, spec in variants.items():
        print(f"\n[FINAL VARIANT] {name}", flush=True)
        models, cat_prediction, _, importance = train_variant(
            name=name,
            training=spec["training"],
            validation=validation,
            test=validation,
            features=spec["features"],
            args=args,
            output_dir=output_dir,
        )
        weights = optimize_weights(actual, lstm_prediction, cat_prediction, args.weight_step)
        ensemble_prediction = blend(lstm_prediction, cat_prediction, weights)
        trained[name] = {
            "models": models,
            "features": spec["features"],
            "weights": weights,
            "catboost": cat_prediction,
            "ensemble": ensemble_prediction,
            "importance": importance,
        }
        for target_index, target in enumerate(TARGETS):
            metric_rows.extend(
                [
                    {
                        "target": target,
                        "candidate": "CatBoost",
                        "variant": name,
                        "lstm_weight": 0.0,
                        "catboost_weight": 1.0,
                        "RMSE_2025": rmse(actual[:, target_index], cat_prediction[:, target_index]),
                    },
                    {
                        "target": target,
                        "candidate": "Ensemble",
                        "variant": name,
                        "lstm_weight": weights[target],
                        "catboost_weight": 1.0 - weights[target],
                        "RMSE_2025": rmse(actual[:, target_index], ensemble_prediction[:, target_index]),
                    },
                ]
            )

    metrics = pd.DataFrame(metric_rows).sort_values(["target", "RMSE_2025"]).reset_index(drop=True)
    selected_rows = metrics.groupby("target", sort=False).head(1).copy()
    selected_rows["rank_within_target"] = 1
    selected_rows.to_csv(output_dir / "final_component_selection.csv", index=False)
    metrics["rank_within_target"] = metrics.groupby("target")["RMSE_2025"].rank(
        method="first", ascending=True
    ).astype(int)
    metrics.to_csv(output_dir / "final_component_candidates.csv", index=False)

    catboost_features_by_target: dict[str, list[str]] = {}
    weights_rows: list[dict[str, object]] = []
    final_prediction = np.empty_like(actual, dtype=float)
    for target_index, target in enumerate(TARGETS):
        selected = selected_rows[selected_rows["target"].eq(target)].iloc[0]
        candidate = str(selected["candidate"])
        variant = str(selected["variant"])
        lstm_weight = float(selected["lstm_weight"])
        catboost_weight = float(selected["catboost_weight"])

        # A zero-weight placeholder CatBoost is still packaged so the inference
        # artifact has one stable file contract for both targets.
        if variant == "none":
            best_cat_row = metrics[
                metrics["target"].eq(target) & metrics["candidate"].eq("CatBoost")
            ].iloc[0]
            model_variant = str(best_cat_row["variant"])
        else:
            model_variant = variant
        shutil.copy2(
            output_dir / "variants" / model_variant / f"catboost_{target}.cbm",
            output_dir / f"catboost_{target}.cbm",
        )
        catboost_features_by_target[target] = list(trained[model_variant]["features"])
        if candidate == "LSTM":
            final_prediction[:, target_index] = lstm_prediction[:, target_index]
        elif candidate == "CatBoost":
            final_prediction[:, target_index] = trained[variant]["catboost"][:, target_index]
        else:
            final_prediction[:, target_index] = trained[variant]["ensemble"][:, target_index]
        weights_rows.append(
            {
                "target": target,
                "component": candidate,
                "catboost_variant": model_variant,
                "lstm_weight": lstm_weight,
                "catboost_weight": catboost_weight,
                "RMSE_2025": float(selected["RMSE_2025"]),
            }
        )

    final_prediction[:, 1] = np.clip(final_prediction[:, 1], 0.0, 100.0)
    for name in ("lstm_model.pt", "lstm_normalization.npz"):
        shutil.copy2(lstm_dir / name, output_dir / name)
    weights_frame = pd.DataFrame(weights_rows)
    weights_frame.to_csv(output_dir / "ensemble_component_weights.csv", index=False)
    feature_spec = {
        "selection_period": args.validation_year,
        "training_end_year": args.train_end_year,
        "catboost_features_by_target": catboost_features_by_target,
        "lstm_channels": CHANNELS,
        "lstm_static_features": ["LAT", "LON", "ALT", "doy_sin", "doy_cos"],
        "observation_times_kst": [
            1200, 1210, 1220, 1230, 1240, 1250, 1300,
            1310, 1320, 1330, 1340, 1350, 1400,
        ],
    }
    (output_dir / "feature_spec.json").write_text(
        json.dumps(feature_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    prediction_frame = validation[["Date", "STN_ID"]].copy()
    for target_index, target in enumerate(TARGETS):
        prediction_frame[f"actual_{target}"] = actual[:, target_index]
        prediction_frame[f"final_{target}"] = final_prediction[:, target_index]
    prediction_frame.to_csv(output_dir / "final_predictions_2025.csv", index=False)
    preview = prediction_frame[["Date", "STN_ID"]].copy()
    preview["ID"] = preview["Date"].astype(str) + "_" + preview["STN_ID"].astype(str)
    preview["TA"] = np.clip(final_prediction[:, 0], -50.0, 50.0).round(2)
    preview["HM"] = np.clip(final_prediction[:, 1], 0.0, 100.0).round(2)
    preview[["ID", "TA", "HM"]].sort_values("ID").to_csv(
        output_dir / "submission_preview_2025.csv", index=False
    )
    ta_rmse = rmse(actual[:, 0], final_prediction[:, 0])
    hm_rmse = rmse(actual[:, 1], final_prediction[:, 1])
    summary = {
        "chronology": {
            "model_development_train": "2019-2023",
            "model_development_validation": 2024,
            "model_development_test": 2025,
            "final_refit_train": f"2019-{args.train_end_year}",
            "final_selection": args.validation_year,
            "competition_evaluation": 2026,
        },
        "selected_components": weights_rows,
        "final_validation_2025": {
            "TA_RMSE": ta_rmse,
            "HM_RMSE": hm_rmse,
            "competition_score": ta_rmse + 0.1 * hm_rmse,
            "rows": len(validation),
        },
    }
    (output_dir / "final_submission_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.sha256.json"
    }
    (output_dir / "manifest.sha256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n[FINAL TARGET SELECTION]", flush=True)
    print(weights_frame.to_string(index=False), flush=True)
    print(
        f"\n2025 validation: TA_RMSE={ta_rmse:.6f}, HM_RMSE={hm_rmse:.6f}, "
        f"score={ta_rmse + 0.1 * hm_rmse:.6f}",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
