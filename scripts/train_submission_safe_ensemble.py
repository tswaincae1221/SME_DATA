#!/usr/bin/env python3
"""Select and package a deployment-safe CatBoost + LSTM submission ensemble.

The historical 14:00 master table and the short-term table were produced by
different preprocessing runs.  This script evaluates CatBoost variants on the
same official-coordinate short-term features that will be available at
inference, chooses the variant using 2024 only, and reports 2025 once.

No evaluation-period ASOS values are used as model inputs.  ASOS TA/HM are read
only from the historical training/validation/test tables as labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = [
    "VI004", "VI005", "VI006", "VI008", "NR013", "NR016", "SW038",
    "WV063", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112",
    "IR123", "IR133",
]
VISIBLE_CHANNELS = ["VI004", "VI005", "VI006", "VI008"]
NONVISIBLE_CHANNELS = [channel for channel in CHANNELS if channel not in VISIBLE_CHANNELS]
STATIC_DATE_FEATURES = [
    "LAT", "LON", "ALT", "month", "day", "dayofyear", "doy_sin", "doy_cos",
]
TARGETS = ["TA", "HM"]
HM_SCORE_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--lstm-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = pd.to_numeric(out["Date"], errors="raise").round().astype("int64")
    out["STN_ID"] = pd.to_numeric(out["STN_ID"], errors="raise").astype("int64")
    date = pd.to_datetime(out["Date"].astype(str), format="%Y%m%d", errors="raise")
    out["year"] = date.dt.year.astype("int16")
    out["month"] = date.dt.month.astype("int8")
    out["day"] = date.dt.day.astype("int8")
    out["dayofyear"] = date.dt.dayofyear.astype("int16")
    angle = 2.0 * np.pi * out["dayofyear"].to_numpy(dtype=float) / 365.25
    out["doy_sin"] = np.sin(angle)
    out["doy_cos"] = np.cos(angle)
    return out


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return float("nan")
    error = predicted[mask] - actual[mask]
    return float(np.sqrt(np.mean(error * error)))


def metric_row(split: str, model: str, actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    ta_rmse = rmse(actual[:, 0], predicted[:, 0])
    hm_rmse = rmse(actual[:, 1], predicted[:, 1])
    return {
        "split": split,
        "model": model,
        "TA_RMSE": ta_rmse,
        "HM_RMSE": hm_rmse,
        "competition_score": ta_rmse + HM_SCORE_WEIGHT * hm_rmse,
    }


def optimize_weights(actual: np.ndarray, lstm: np.ndarray, catboost: np.ndarray, step: float) -> dict[str, float]:
    candidates = np.arange(0.0, 1.0 + step / 2.0, step)
    result: dict[str, float] = {}
    for index, target in enumerate(TARGETS):
        scores = [
            rmse(actual[:, index], weight * lstm[:, index] + (1.0 - weight) * catboost[:, index])
            for weight in candidates
        ]
        result[target] = float(candidates[int(np.nanargmin(scores))])
    return result


def blend(lstm: np.ndarray, catboost: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    result = np.empty_like(lstm, dtype=float)
    for index, target in enumerate(TARGETS):
        weight = weights[target]
        result[:, index] = weight * lstm[:, index] + (1.0 - weight) * catboost[:, index]
    result[:, 1] = np.clip(result[:, 1], 0.0, 100.0)
    return result


def align_lstm_predictions(frame: pd.DataFrame, prediction_path: Path) -> pd.DataFrame:
    prediction = pd.read_csv(prediction_path)
    required = ["Date", "STN_ID", "actual_TA", "actual_HM", "lstm_TA", "lstm_HM"]
    missing = [column for column in required if column not in prediction]
    if missing:
        raise ValueError(f"{prediction_path.name} missing columns: {missing}")
    merged = frame.merge(
        prediction[required], on=["Date", "STN_ID"], how="inner", validate="1:1",
    ).sort_values(["Date", "STN_ID"]).reset_index(drop=True)
    if len(merged) != len(prediction):
        raise ValueError(
            f"short-term/prediction key mismatch: table={len(frame)}, "
            f"prediction={len(prediction)}, common={len(merged)}"
        )
    for target in TARGETS:
        if not np.allclose(merged[target], merged[f"actual_{target}"], equal_nan=True):
            raise ValueError(f"historical {target} labels disagree for {prediction_path.name}")
    return merged


def train_variant(
    *,
    name: str,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, pd.DataFrame]:
    from catboost import CatBoostRegressor

    variant_dir = output_dir / "variants" / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, object] = {}
    validation_prediction = np.empty((len(validation), len(TARGETS)), dtype=float)
    test_prediction = np.empty((len(test), len(TARGETS)), dtype=float)
    importance_parts: list[pd.DataFrame] = []
    for target_index, target in enumerate(TARGETS):
        train_rows = training[np.isfinite(training[target])]
        validation_rows = validation[np.isfinite(validation[target])]
        model = CatBoostRegressor(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=args.seed,
            thread_count=max(1, args.threads),
            verbose=100,
            allow_writing_files=False,
        )
        model.fit(
            train_rows[features], train_rows[target],
            eval_set=(validation_rows[features], validation_rows[target]),
            early_stopping_rounds=args.early_stopping_rounds,
            use_best_model=True,
        )
        model.save_model(variant_dir / f"catboost_{target}.cbm")
        models[target] = model
        validation_prediction[:, target_index] = model.predict(validation[features])
        test_prediction[:, target_index] = model.predict(test[features])
        importance_parts.append(
            pd.DataFrame(
                {
                    "variant": name,
                    "target": target,
                    "feature": features,
                    "importance": model.get_feature_importance(type="PredictionValuesChange"),
                    "best_iteration": model.get_best_iteration(),
                }
            )
        )
    validation_prediction[:, 1] = np.clip(validation_prediction[:, 1], 0.0, 100.0)
    test_prediction[:, 1] = np.clip(test_prediction[:, 1], 0.0, 100.0)
    importance = pd.concat(importance_parts, ignore_index=True)
    importance["rank"] = importance.groupby("target")["importance"].rank(
        method="first", ascending=False
    ).astype(int)
    importance.sort_values(["target", "rank"]).to_csv(
        variant_dir / "catboost_feature_importance.csv", index=False
    )
    (variant_dir / "feature_spec.json").write_text(
        json.dumps({"variant": name, "features": features}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return models, validation_prediction, test_prediction, importance


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
    artifact_dir = Path(args.lstm_artifact_dir).expanduser().resolve()

    master = add_date_features(pd.read_csv(args.master_csv))
    shortterm = pd.read_csv(args.shortterm_long_csv)
    shortterm = shortterm[pd.to_numeric(shortterm["TimeKST"], errors="coerce").eq(1400)].copy()
    shortterm = add_date_features(shortterm)
    if shortterm.duplicated(["Date", "STN_ID"]).any():
        raise ValueError("short-term 14:00 table has duplicate Date/STN_ID keys")

    validation_base = shortterm[shortterm["year"].eq(args.validation_year)].copy()
    test_base = shortterm[shortterm["year"].eq(args.test_year)].copy()
    validation = align_lstm_predictions(validation_base, artifact_dir / "predictions_2024.csv")
    test = align_lstm_predictions(test_base, artifact_dir / "predictions_2025.csv")
    validation_actual = validation[TARGETS].to_numpy(dtype=float)
    test_actual = test[TARGETS].to_numpy(dtype=float)
    validation_lstm = validation[["lstm_TA", "lstm_HM"]].to_numpy(dtype=float)
    test_lstm = test[["lstm_TA", "lstm_HM"]].to_numpy(dtype=float)

    variants = {
        # Full historical summer coverage, while dropping the four channels whose
        # old and official-coordinate preprocessing disagree materially.
        "master_nonvisible": {
            "training": master[master["year"].le(args.train_end_year)].copy(),
            "features": [*NONVISIBLE_CHANNELS, *STATIC_DATE_FEATURES],
        },
        # Smaller, exact preprocessing match.  This is included as a controlled
        # comparison rather than assumed to be better.
        "shortterm_all16": {
            "training": shortterm[shortterm["year"].le(args.train_end_year)].copy(),
            "features": [*CHANNELS, *STATIC_DATE_FEATURES],
        },
    }

    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    trained: dict[str, dict[str, object]] = {}
    for name, spec in variants.items():
        print(f"\n[VARIANT] {name}", flush=True)
        models, cat_validation, cat_test, importance = train_variant(
            name=name,
            training=spec["training"],
            validation=validation,
            test=test,
            features=spec["features"],
            args=args,
            output_dir=output_dir,
        )
        weights = optimize_weights(
            validation_actual, validation_lstm, cat_validation, args.weight_step
        )
        ensemble_validation = blend(validation_lstm, cat_validation, weights)
        ensemble_test = blend(test_lstm, cat_test, weights)
        for split, actual, cat_pred, ensemble_pred in (
            ("validation_2024", validation_actual, cat_validation, ensemble_validation),
            ("test_2025", test_actual, cat_test, ensemble_test),
        ):
            metric_rows.append(metric_row(split, f"{name}:CatBoost", actual, cat_pred))
            metric_rows.append(metric_row(split, f"{name}:Ensemble", actual, ensemble_pred))
        validation_metric = metric_row(
            "validation_2024", f"{name}:Ensemble", validation_actual, ensemble_validation
        )
        selection_rows.append(
            {
                "variant": name,
                "feature_count": len(spec["features"]),
                "train_rows": len(spec["training"]),
                "lstm_weight_TA": weights["TA"],
                "catboost_weight_TA": 1.0 - weights["TA"],
                "lstm_weight_HM": weights["HM"],
                "catboost_weight_HM": 1.0 - weights["HM"],
                "validation_score": validation_metric["competition_score"],
            }
        )
        trained[name] = {
            "models": models,
            "features": spec["features"],
            "weights": weights,
            "cat_validation": cat_validation,
            "cat_test": cat_test,
            "ensemble_validation": ensemble_validation,
            "ensemble_test": ensemble_test,
            "importance": importance,
        }

    # LSTM is identical across variants; include it once in the comparison.
    metric_rows.extend(
        [
            metric_row("validation_2024", "LSTM", validation_actual, validation_lstm),
            metric_row("test_2025", "LSTM", test_actual, test_lstm),
        ]
    )
    metrics = pd.DataFrame(metric_rows).sort_values(["split", "competition_score"])
    selection = pd.DataFrame(selection_rows).sort_values("validation_score").reset_index(drop=True)
    selected_name = str(selection.iloc[0]["variant"])
    selected = trained[selected_name]

    # Root-level package contains only the 2024-selected deployment artifacts.
    for target in TARGETS:
        shutil.copy2(
            output_dir / "variants" / selected_name / f"catboost_{target}.cbm",
            output_dir / f"catboost_{target}.cbm",
        )
    for name in ("lstm_model.pt", "lstm_normalization.npz"):
        source = artifact_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)
    shutil.copy2(
        output_dir / "variants" / selected_name / "catboost_feature_importance.csv",
        output_dir / "catboost_feature_importance.csv",
    )
    feature_spec = {
        "selected_variant": selected_name,
        "catboost_features": selected["features"],
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
    weights_frame = pd.DataFrame(
        [
            {
                "target": target,
                "lstm_weight": selected["weights"][target],
                "catboost_weight": 1.0 - selected["weights"][target],
            }
            for target in TARGETS
        ]
    )
    weights_frame.to_csv(output_dir / "ensemble_component_weights.csv", index=False)
    metrics.to_csv(output_dir / "submission_variant_metrics.csv", index=False)
    selection.to_csv(output_dir / "submission_variant_selection.csv", index=False)

    for split_name, aligned, actual, lstm_pred, cat_pred, ensemble_pred in (
        (
            "validation_2024", validation, validation_actual, validation_lstm,
            selected["cat_validation"], selected["ensemble_validation"],
        ),
        (
            "test_2025", test, test_actual, test_lstm,
            selected["cat_test"], selected["ensemble_test"],
        ),
    ):
        frame = aligned[["Date", "STN_ID"]].copy()
        for index, target in enumerate(TARGETS):
            frame[f"actual_{target}"] = actual[:, index]
            frame[f"lstm_{target}"] = lstm_pred[:, index]
            frame[f"catboost_{target}"] = cat_pred[:, index]
            frame[f"ensemble_{target}"] = ensemble_pred[:, index]
        frame.to_csv(output_dir / f"submission_predictions_{split_name}.csv", index=False)

    preview = test[["Date", "STN_ID"]].copy()
    preview["ID"] = preview["Date"].astype(str) + "_" + preview["STN_ID"].astype(str)
    preview["TA"] = np.clip(selected["ensemble_test"][:, 0], -50.0, 50.0).round(2)
    preview["HM"] = np.clip(selected["ensemble_test"][:, 1], 0.0, 100.0).round(2)
    preview[["ID", "TA", "HM"]].sort_values("ID").to_csv(
        output_dir / "submission_preview_2025.csv", index=False
    )

    summary = {
        "selection_rule": "lowest 2024 validation competition score",
        "selected_variant": selected_name,
        "ensemble_weights": weights_frame.to_dict(orient="records"),
        "validation_metrics": metrics[metrics["split"].eq("validation_2024")].to_dict(orient="records"),
        "test_metrics": metrics[metrics["split"].eq("test_2025")].to_dict(orient="records"),
        "counts": {
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
    }
    (output_dir / "submission_experiment_summary.json").write_text(
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

    print("\n[SELECTION]", flush=True)
    print(selection.to_string(index=False), flush=True)
    print("\n[METRICS]", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(f"\nSelected: {selected_name}\nOutput: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
