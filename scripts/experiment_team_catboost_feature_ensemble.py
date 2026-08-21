#!/usr/bin/env python3
"""Evaluate rules-safe team features and blend them with the current model.

This experiment deliberately keeps the repository's existing CatBoost training
configuration fixed.  It audits only two things from the supplied team code:

1. feature engineering at the current 14:00 observation, and
2. target-specific OOF ensembling.

No feature may contain an ASOS observation, an ASOS lag, climatology, a direct
year term, external geography, or external weather.  Spatial features use the
official station list and same-date GK-2A LE1B channel values only.  The
optional TA hint for HM is a rolling OOF/frozen model prediction; observed TA
is never used as an HM input.

Feature groups are tested cumulatively with rolling-year OOF.  A group is kept
only when it improves pooled OOF RMSE and does not materially degrade the most
recent OOF year.  The report year is scored only after all feature and blend
choices have been frozen.
"""

from __future__ import annotations

import argparse
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
TARGETS = ["TA", "HM"]

PHYSICAL_FEATURES = [
    "team_diff_ir105_ir087",
    "team_diff_ir105_ir123",
    "team_diff_wv063_wv073",
    "team_diff_ir105_wv069",
    "team_diff_ir096_wv063",
    "team_diff_sw038_ir096",
]
SPATIAL_SOURCE_FEATURES = [
    "IR105",
    "WV069",
    "team_diff_ir096_wv063",
    "team_ndvi_safe",
]
SPATIAL_FEATURES = [
    *[f"team_daily_anomaly_{feature.lower()}" for feature in SPATIAL_SOURCE_FEATURES],
    "team_nearest3_mean_ir105",
    "team_nearest3_delta_ir105",
    "team_nearest3_mean_wv069",
    "team_nearest3_delta_wv069",
]
NORMALIZED_VISIBLE_TA = ["team_ndvi_safe", "team_signed_log_sw_vi"]
NORMALIZED_VISIBLE_HM = ["team_ndvi_safe"]
TA_HINT_FEATURES = ["team_oof_ta_hint", "team_oof_ta_hint_missing"]

FORBIDDEN_FEATURE_MARKERS = (
    "lag", "roll", "climat", "normal_", "평년", "previous", "recent_asos",
    "actual_ta", "actual_hm", "observed_ta", "observed_hm",
)


@dataclass
class VariantResult:
    target: str
    name: str
    features: tuple[str, ...]
    categorical: tuple[str, ...]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    pooled_rmse: float
    latest_rmse: float
    worst_rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--station-list", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--optimizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oof-years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--report-year", type=int, default=2025)
    parser.add_argument("--validation-start-mmdd", type=int, default=824)
    parser.add_argument("--validation-end-mmdd", type=int, default=830)
    # These defaults are the existing repository baseline, not values copied
    # from either supplied team file.  The experiment never tunes them.
    parser.add_argument("--cat-iterations", type=int, default=500)
    parser.add_argument("--cat-depth", type=int, default=7)
    parser.add_argument("--cat-learning-rate", type=float, default=0.04)
    parser.add_argument("--cat-l2", type=float, default=8.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-step", type=float, default=0.02)
    parser.add_argument("--min-improvement-ta", type=float, default=0.005)
    parser.add_argument("--min-improvement-hm", type=float, default=0.05)
    parser.add_argument("--min-blend-improvement-ta", type=float, default=0.005)
    parser.add_argument("--min-blend-improvement-hm", type=float, default=0.02)
    parser.add_argument("--latest-tolerance-ta", type=float, default=0.01)
    parser.add_argument("--latest-tolerance-hm", type=float, default=0.10)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(prediction)
    if not mask.any():
        return math.nan
    return float(np.sqrt(np.mean((prediction[mask] - actual[mask]) ** 2)))


def clean_station_list(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["STN_ID", "LAT", "LON", "ALT"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"official station list is missing: {missing}")
    frame = frame[required].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["STN_ID"] = frame.STN_ID.round().astype("int64")
    if frame.STN_ID.duplicated().any():
        raise ValueError("official station list has duplicate STN_ID")
    return frame.sort_values("STN_ID").reset_index(drop=True)


def _haversine_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(latitude.astype(float))
    lon = np.radians(longitude.astype(float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def nearest_station_ids(stations: pd.DataFrame, count: int = 3) -> dict[int, list[int]]:
    if len(stations) <= count:
        raise ValueError(f"at least {count + 1} official stations are required")
    distance = _haversine_matrix(stations.LAT.to_numpy(), stations.LON.to_numpy())
    order = np.argsort(distance, axis=1)[:, 1 : count + 1]
    identifiers = stations.STN_ID.to_numpy(dtype=int)
    return {
        int(identifiers[index]): [int(value) for value in identifiers[order[index]]]
        for index in range(len(stations))
    }


def apply_official_coordinates(master: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    out = base.clean_keys(master)
    official = stations.rename(columns={"LAT": "official_LAT", "LON": "official_LON", "ALT": "official_ALT"})
    out = out.merge(official, on="STN_ID", how="left", validate="m:1")
    if out[["official_LAT", "official_LON", "official_ALT"]].isna().any().any():
        unknown = sorted(out.loc[out.official_LAT.isna(), "STN_ID"].unique().tolist())
        raise ValueError(f"master contains stations absent from the official station list: {unknown}")
    for column in ["LAT", "LON", "ALT"]:
        source = f"official_{column}"
        if column in out:
            difference = np.nanmax(np.abs(pd.to_numeric(out[column], errors="coerce") - out[source]))
            if np.isfinite(difference) and difference > 1e-4:
                print(f"[OFFICIAL] replacing {column}; max supplied difference={difference:.6g}", flush=True)
        out[column] = out[source]
    return out.drop(columns=["official_LAT", "official_LON", "official_ALT"])


def add_same_date_neighbour_features(
    frame: pd.DataFrame,
    neighbours: dict[int, list[int]],
    source_columns: list[str],
) -> pd.DataFrame:
    out = frame.copy()
    key_index = pd.MultiIndex.from_frame(out[KEYS])
    for source in source_columns:
        wide = out.pivot(index="Date", columns="STN_ID", values=source)
        means = pd.DataFrame(index=wide.index)
        for station, station_neighbours in neighbours.items():
            available = [value for value in station_neighbours if value in wide.columns]
            means[station] = wide[available].mean(axis=1, skipna=True) if available else np.nan
        long_mean = (
            means.rename_axis(index="Date", columns="STN_ID")
            .reset_index()
            .melt(id_vars="Date", var_name="STN_ID", value_name="_nearest_mean")
            .set_index(KEYS)["_nearest_mean"]
        )
        values = long_mean.reindex(key_index).to_numpy(dtype=float)
        short = source.lower()
        out[f"team_nearest3_mean_{short}"] = values
        out[f"team_nearest3_delta_{short}"] = out[source].to_numpy(dtype=float) - values
    return out


def engineer_team_features(master: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    required = [*base.CHANNELS, "Date", "STN_ID", "TA", "HM"]
    missing = [column for column in required if column not in master]
    if missing:
        raise ValueError(f"master CSV is missing: {missing}")
    out = base.add_solar_features(apply_official_coordinates(master, stations))
    out["team_diff_ir105_ir087"] = out.IR105 - out.IR087
    out["team_diff_ir105_ir123"] = out.IR105 - out.IR123
    out["team_diff_wv063_wv073"] = out.WV063 - out.WV073
    out["team_diff_ir105_wv069"] = out.IR105 - out.WV069
    out["team_diff_ir096_wv063"] = out.IR096 - out.WV063
    out["team_diff_sw038_ir096"] = out.SW038 - out.IR096

    # Bounded normalized difference: no dataset-wide minimum, no future bfill,
    # and no explosion when the original NDVI denominator approaches zero.
    visible_scale = np.abs(out.VI008) + np.abs(out.VI006)
    out["team_ndvi_safe"] = (out.VI008 - out.VI006) / visible_scale.where(visible_scale > 1e-8)
    out["team_ndvi_safe"] = out.team_ndvi_safe.clip(-1.0, 1.0)
    out["team_signed_log_sw_vi"] = (
        np.sign(out.SW038) * np.log1p(np.abs(out.SW038))
        - np.sign(out.VI006) * np.log1p(np.abs(out.VI006))
    )
    out["STN_ID_cat"] = out.STN_ID.astype(str)

    for source in SPATIAL_SOURCE_FEATURES:
        out[f"team_daily_anomaly_{source.lower()}"] = out[source] - out.groupby("Date")[source].transform("mean")
    out = add_same_date_neighbour_features(
        out, nearest_station_ids(stations), ["IR105", "WV069"]
    )
    out["team_oof_ta_hint"] = np.nan
    out["team_oof_ta_hint_missing"] = 1.0
    return out


def feature_groups(target: str) -> list[tuple[str, list[str], list[str]]]:
    groups = [
        ("physical_channel_relations", PHYSICAL_FEATURES, []),
        ("same_date_station_context", SPATIAL_FEATURES, []),
        ("station_identity", ["STN_ID_cat"], ["STN_ID_cat"]),
        (
            "bounded_visible_relations",
            NORMALIZED_VISIBLE_TA if target == "TA" else NORMALIZED_VISIBLE_HM,
            [],
        ),
    ]
    if target == "HM":
        groups.append(("oof_predicted_ta_hint", TA_HINT_FEATURES, []))
    return groups


def validate_feature_contract(features: list[str], target: str) -> None:
    if len(features) != len(set(features)):
        raise ValueError(f"{target}: duplicate feature names")
    forbidden = [
        feature for feature in features
        if feature.lower() in {"year", "date", "time", "ta", "hm"}
        or any(marker in feature.lower() for marker in FORBIDDEN_FEATURE_MARKERS)
    ]
    if forbidden:
        raise ValueError(f"{target}: prohibited features: {forbidden}")
    if target == "HM" and "TA" in features:
        raise ValueError("observed TA cannot be an HM feature")


def fold_engineered_table(frame: pd.DataFrame, train_end_year: int) -> pd.DataFrame:
    stats = base.fit_channel_stats(frame, train_end_year)
    return base.engineer_tabular(frame, stats)


def validation_mask(
    frame: pd.DataFrame,
    year: int,
    args: argparse.Namespace,
    window_only: bool = True,
) -> pd.Series:
    mask = frame.year.eq(year)
    if window_only:
        mmdd = frame.Date.astype("int64") % 10000
        mask &= mmdd.between(args.validation_start_mmdd, args.validation_end_mmdd)
    return mask


def fit_catboost(
    train: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    target: str,
    args: argparse.Namespace,
    seed_offset: int,
):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=args.cat_iterations,
        depth=args.cat_depth,
        learning_rate=args.cat_learning_rate,
        l2_leaf_reg=args.cat_l2,
        loss_function="RMSE",
        random_seed=args.seed + seed_offset,
        thread_count=max(1, args.threads),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train[features], train[target], cat_features=categorical)
    return model


def rolling_oof(
    frame: pd.DataFrame,
    target: str,
    name: str,
    features: list[str],
    categorical: list[str],
    years: list[int],
    args: argparse.Namespace,
    window_only: bool = True,
) -> VariantResult:
    validate_feature_contract(features, target)
    parts = []
    metrics = []
    for year in years:
        engineered = fold_engineered_table(frame, year - 1)
        train = engineered[engineered.year.le(year - 1) & engineered[target].notna()].copy()
        validation = engineered[validation_mask(engineered, year, args, window_only) & engineered[target].notna()].copy()
        if train.empty or validation.empty:
            raise ValueError(f"{target}/{name}: empty rolling fold {year}")
        for column in categorical:
            train[column] = train[column].astype(str)
            validation[column] = validation[column].astype(str)
        model = fit_catboost(train, features, categorical, target, args, year)
        prediction = np.asarray(model.predict(validation[features]), dtype=float)
        if target == "HM":
            prediction = np.clip(prediction, 0.0, 100.0)
        actual = validation[target].to_numpy(dtype=float)
        score = rmse(actual, prediction)
        metrics.append({"target": target, "variant": name, "year": year, "RMSE": score, "n": len(validation)})
        part = validation[KEYS + ["year", target]].copy()
        part["prediction"] = prediction
        parts.append(part)
        print(f"[{target} {name}] OOF {year}: RMSE={score:.6f}", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    fold_metrics = pd.DataFrame(metrics)
    pooled = rmse(predictions[target].to_numpy(float), predictions.prediction.to_numpy(float))
    return VariantResult(
        target=target,
        name=name,
        features=tuple(features),
        categorical=tuple(categorical),
        predictions=predictions,
        fold_metrics=fold_metrics,
        pooled_rmse=pooled,
        latest_rmse=float(fold_metrics.loc[fold_metrics.year.idxmax(), "RMSE"]),
        worst_rmse=float(fold_metrics.RMSE.max()),
    )


def select_feature_groups(
    frame: pd.DataFrame,
    target: str,
    years: list[int],
    args: argparse.Namespace,
) -> tuple[VariantResult, pd.DataFrame]:
    features = list(base.TABULAR_FEATURES)
    categorical: list[str] = []
    current = rolling_oof(frame, target, "base40", features, categorical, years, args)
    rows = [{
        "target": target, "candidate_group": "base40", "accepted": True,
        "feature_count": len(features), "pooled_OOF_RMSE": current.pooled_rmse,
        "latest_OOF_RMSE": current.latest_rmse, "delta_pooled_RMSE": 0.0,
    }]
    minimum = args.min_improvement_ta if target == "TA" else args.min_improvement_hm
    tolerance = args.latest_tolerance_ta if target == "TA" else args.latest_tolerance_hm
    for group_name, additions, categorical_additions in feature_groups(target):
        candidate_features = list(dict.fromkeys([*features, *additions]))
        candidate_categorical = list(dict.fromkeys([*categorical, *categorical_additions]))
        candidate = rolling_oof(
            frame, target, group_name, candidate_features, candidate_categorical, years, args
        )
        improvement = current.pooled_rmse - candidate.pooled_rmse
        accepted = improvement >= minimum and candidate.latest_rmse <= current.latest_rmse + tolerance
        rows.append({
            "target": target, "candidate_group": group_name, "accepted": accepted,
            "feature_count": len(candidate_features), "pooled_OOF_RMSE": candidate.pooled_rmse,
            "latest_OOF_RMSE": candidate.latest_rmse,
            "delta_pooled_RMSE": candidate.pooled_rmse - current.pooled_rmse,
        })
        if accepted:
            features = candidate_features
            categorical = candidate_categorical
            current = candidate
    return current, pd.DataFrame(rows)


def final_fit_predict(
    frame: pd.DataFrame,
    result: VariantResult,
    report_year: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    engineered = fold_engineered_table(frame, report_year - 1)
    train = engineered[engineered.year.le(report_year - 1) & engineered[result.target].notna()].copy()
    test = engineered[
        validation_mask(engineered, report_year, args, True) & engineered[result.target].notna()
    ].copy()
    features = list(result.features)
    categorical = list(result.categorical)
    for column in categorical:
        train[column] = train[column].astype(str)
        test[column] = test[column].astype(str)
    model = fit_catboost(train, features, categorical, result.target, args, report_year)
    prediction = np.asarray(model.predict(test[features]), dtype=float)
    if result.target == "HM":
        prediction = np.clip(prediction, 0.0, 100.0)
    output = test[KEYS + ["year", result.target]].copy()
    output[f"team_catboost_{result.target}"] = prediction
    importance = pd.DataFrame({
        "target": result.target,
        "feature": features,
        "importance": model.get_feature_importance(type="PredictionValuesChange"),
    }).sort_values("importance", ascending=False)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / f"team_feature_catboost_{result.target}.cbm")
    return output, importance


def add_oof_ta_hint(
    frame: pd.DataFrame,
    ta_result: VariantResult,
    ta_report: pd.DataFrame,
    early_years: list[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    earliest = int(frame.loc[frame.TA.notna(), "year"].min())
    years = sorted(set(year for year in early_years if year > earliest))
    ta_all_oof = rolling_oof(
        frame, "TA", "selected_for_hm_hint", list(ta_result.features),
        list(ta_result.categorical), years, args, window_only=False,
    ).predictions
    hints = ta_all_oof[KEYS + ["prediction"]].rename(columns={"prediction": "team_oof_ta_hint"})
    out = frame.drop(columns=TA_HINT_FEATURES).merge(hints, on=KEYS, how="left", validate="1:1")
    out["team_oof_ta_hint_missing"] = out.team_oof_ta_hint.isna().astype(float)
    report_hints = ta_report[KEYS + ["team_catboost_TA"]].rename(
        columns={"team_catboost_TA": "team_oof_ta_hint"}
    )
    return out, report_hints


def load_current_best(
    target: str,
    baseline_dir: Path,
    optimizer_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_oof = pd.read_csv(baseline_dir / f"step0_current_baseline_{target}_oof.csv")
    baseline_test = pd.read_csv(baseline_dir / f"step0_current_baseline_{target}_test.csv")
    with (optimizer_dir / "feature_engineering_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    target_summary = summary["targets"][target]
    weights = target_summary[f"report_2025_ensemble"]
    baseline_weight = float(weights["baseline_weight"])
    engineered_weight = float(weights["engineered_weight"])

    oof = baseline_oof[KEYS + ["year", f"actual_{target}", f"prediction_{target}"]].copy()
    engineered = pd.read_csv(optimizer_dir / f"selected_oof_predictions_{target}.csv")
    engineered = engineered[KEYS + ["year", "prediction"]].rename(columns={"prediction": "engineered_prediction"})
    oof = oof.merge(engineered, on=KEYS + ["year"], how="left", validate="1:1")
    has_engineered = oof.engineered_prediction.notna()
    oof["current_best"] = oof[f"prediction_{target}"]
    oof.loc[has_engineered, "current_best"] = (
        baseline_weight * oof.loc[has_engineered, f"prediction_{target}"]
        + engineered_weight * oof.loc[has_engineered, "engineered_prediction"]
    )

    optimized_test = pd.read_csv(optimizer_dir / f"report_2025_ensemble_{target}.csv")
    test = baseline_test[KEYS + ["year", f"actual_{target}"]].merge(
        optimized_test[KEYS + [f"ensemble_{target}"]], on=KEYS, how="inner", validate="1:1"
    )
    test = test.rename(columns={f"ensemble_{target}": "current_best"})
    return oof, test


def optimize_blend_weight(
    actual: np.ndarray,
    current: np.ndarray,
    candidate: np.ndarray,
    step: float,
) -> tuple[float, float]:
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    scores = [rmse(actual, (1.0 - weight) * current + weight * candidate) for weight in weights]
    index = int(np.nanargmin(scores))
    return float(weights[index]), float(scores[index])


def blend_with_current(
    target: str,
    current_oof: pd.DataFrame,
    current_test: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    candidate_test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    actual_column = f"actual_{target}"
    candidate_column = f"team_catboost_{target}"
    oof = current_oof.merge(
        candidate_oof[KEYS + ["year", "prediction"]].rename(columns={"prediction": candidate_column}),
        on=KEYS + ["year"], how="inner", validate="1:1",
    )
    actual = oof[actual_column].to_numpy(float)
    current = oof.current_best.to_numpy(float)
    candidate = oof[candidate_column].to_numpy(float)
    candidate_weight, selected_rmse = optimize_blend_weight(
        actual, current, candidate, args.weight_step
    )
    baseline_rmse = rmse(actual, current)
    minimum = (
        getattr(args, "min_blend_improvement_ta", args.min_improvement_ta)
        if target == "TA"
        else getattr(args, "min_blend_improvement_hm", args.min_improvement_hm)
    )
    latest_year = int(oof.year.max())
    latest = oof.year.eq(latest_year).to_numpy()
    latest_baseline = rmse(actual[latest], current[latest])
    latest_blend = rmse(
        actual[latest],
        (1.0 - candidate_weight) * current[latest] + candidate_weight * candidate[latest],
    )
    tolerance = args.latest_tolerance_ta if target == "TA" else args.latest_tolerance_hm
    accepted = baseline_rmse - selected_rmse >= minimum and latest_blend <= latest_baseline + tolerance
    if not accepted:
        candidate_weight = 0.0
        selected_rmse = baseline_rmse
        latest_blend = latest_baseline
    oof[f"ensemble_{target}"] = (
        (1.0 - candidate_weight) * current + candidate_weight * candidate
    )

    test = current_test.merge(
        candidate_test[KEYS + [candidate_column]], on=KEYS, how="inner", validate="1:1"
    )
    test[f"ensemble_{target}"] = (
        (1.0 - candidate_weight) * test.current_best.to_numpy(float)
        + candidate_weight * test[candidate_column].to_numpy(float)
    )
    if target == "HM":
        test[f"ensemble_{target}"] = test[f"ensemble_{target}"].clip(0.0, 100.0)

    # Forward meta-OOF is diagnostic only: each validation-year blend weight is
    # learned from earlier OOF years, never from that year's labels.
    forward_parts = []
    for year in sorted(int(value) for value in oof.year.unique())[1:]:
        train = oof[oof.year.lt(year)]
        validation = oof[oof.year.eq(year)].copy()
        forward_weight, _ = optimize_blend_weight(
            train[actual_column].to_numpy(float), train.current_best.to_numpy(float),
            train[candidate_column].to_numpy(float), args.weight_step,
        )
        validation["forward_prediction"] = (
            (1.0 - forward_weight) * validation.current_best
            + forward_weight * validation[candidate_column]
        )
        validation["forward_candidate_weight"] = forward_weight
        forward_parts.append(validation)
    forward = pd.concat(forward_parts, ignore_index=True)
    test_rmse_current = rmse(test[actual_column].to_numpy(float), test.current_best.to_numpy(float))
    test_rmse_ensemble = rmse(test[actual_column].to_numpy(float), test[f"ensemble_{target}"].to_numpy(float))
    recipe = {
        "target": target,
        "current_best_weight": 1.0 - candidate_weight,
        "team_feature_catboost_weight": candidate_weight,
        "weight_selected_from": "2022-2024 rolling OOF only",
        "accepted": accepted,
        "OOF_current_RMSE": baseline_rmse,
        "OOF_selected_RMSE": selected_rmse,
        "OOF_improvement": baseline_rmse - selected_rmse,
        "latest_OOF_year": latest_year,
        "latest_OOF_current_RMSE": latest_baseline,
        "latest_OOF_selected_RMSE": latest_blend,
        "forward_meta_OOF_RMSE": rmse(
            forward[actual_column].to_numpy(float), forward.forward_prediction.to_numpy(float)
        ),
        "report_current_RMSE": test_rmse_current,
        "report_ensemble_RMSE": test_rmse_ensemble,
        "report_improvement": test_rmse_current - test_rmse_ensemble,
        "report_year_used_for_weight_selection": False,
    }
    return oof, test, recipe


def audit_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"source_idea": "raw channel differences", "decision": "candidate", "reason": "same-time LE1B-only physical relations"},
        {"source_idea": "NDVI proxy", "decision": "modified candidate", "reason": "bounded denominator; no dataset-wide minimum"},
        {"source_idea": "same-date all-station anomaly", "decision": "candidate", "reason": "contemporaneous LE1B spatial context"},
        {"source_idea": "nearest-three station mean", "decision": "modified candidate", "reason": "official coordinates + haversine distance"},
        {"source_idea": "categorical station ID", "decision": "candidate", "reason": "station number is in the official station list"},
        {"source_idea": "predicted TA hint for HM", "decision": "modified candidate", "reason": "rolling OOF/frozen TA prediction only"},
        {"source_idea": "Year", "decision": "excluded", "reason": "prior year-dependence instability; direct year term"},
        {"source_idea": "Hour", "decision": "excluded", "reason": "master Time is a constant 05 UTC / 14 KST"},
        {"source_idea": "Lag1/Roll3/Delta1", "decision": "excluded", "reason": "past-snapshot dependency and first-row future bfill risk"},
        {"source_idea": "bfill validation features", "decision": "excluded", "reason": "future-row leakage"},
        {"source_idea": "validation-wide min log transform", "decision": "excluded", "reason": "transductive transform and train/test mismatch"},
        {"source_idea": "validation-tuned three-model weights", "decision": "replaced", "reason": "target-specific rolling OOF blend"},
    ])


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    oof_years = sorted(set(args.oof_years))
    if oof_years != args.oof_years or max(oof_years) >= args.report_year:
        raise ValueError("--oof-years must be unique, ascending, and earlier than --report-year")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    optimizer_dir = Path(args.optimizer_dir).expanduser().resolve()

    stations = clean_station_list(args.station_list)
    master = pd.read_csv(args.master_csv)
    frame = engineer_team_features(master, stations)
    if frame.duplicated(KEYS).any():
        raise ValueError("master has duplicate Date/STN_ID keys")
    audit_rows().to_csv(output_dir / "team_source_feature_audit.csv", index=False)

    print("\n" + "=" * 100, flush=True)
    print("[TA] groupwise rolling OOF selection", flush=True)
    ta_result, ta_trials = select_feature_groups(frame, "TA", oof_years, args)
    ta_report, ta_importance = final_fit_predict(frame, ta_result, args.report_year, args, output_dir)

    # Generate strictly prior-year TA predictions for the HM chain candidate.
    hm_frame, report_hints = add_oof_ta_hint(
        frame, ta_result, ta_report, list(range(int(frame.year.min()) + 1, max(oof_years) + 1)), args
    )
    report_mask = validation_mask(hm_frame, args.report_year, args, True)
    report_hint_lookup = report_hints.set_index(KEYS).team_oof_ta_hint
    report_index = pd.MultiIndex.from_frame(hm_frame.loc[report_mask, KEYS])
    hm_frame.loc[report_mask, "team_oof_ta_hint"] = report_hint_lookup.reindex(report_index).to_numpy(float)
    hm_frame.loc[report_mask, "team_oof_ta_hint_missing"] = hm_frame.loc[report_mask, "team_oof_ta_hint"].isna().astype(float)

    print("\n" + "=" * 100, flush=True)
    print("[HM] groupwise rolling OOF selection", flush=True)
    hm_result, hm_trials = select_feature_groups(hm_frame, "HM", oof_years, args)
    hm_report, hm_importance = final_fit_predict(hm_frame, hm_result, args.report_year, args, output_dir)

    trial_table = pd.concat([ta_trials, hm_trials], ignore_index=True)
    trial_table.to_csv(output_dir / "feature_group_oof_trials.csv", index=False)
    importance = pd.concat([ta_importance, hm_importance], ignore_index=True)
    importance.to_csv(output_dir / "selected_catboost_feature_importance.csv", index=False)
    ta_result.predictions.to_csv(output_dir / "team_catboost_TA_oof.csv", index=False)
    hm_result.predictions.to_csv(output_dir / "team_catboost_HM_oof.csv", index=False)
    ta_result.fold_metrics.to_csv(output_dir / "team_catboost_TA_fold_metrics.csv", index=False)
    hm_result.fold_metrics.to_csv(output_dir / "team_catboost_HM_fold_metrics.csv", index=False)
    ta_report.to_csv(output_dir / f"team_catboost_TA_{args.report_year}.csv", index=False)
    hm_report.to_csv(output_dir / f"team_catboost_HM_{args.report_year}.csv", index=False)

    recipes = {}
    final_tests = {}
    for target, result, report in [("TA", ta_result, ta_report), ("HM", hm_result, hm_report)]:
        current_oof, current_test = load_current_best(target, baseline_dir, optimizer_dir)
        ensemble_oof, ensemble_test, recipe = blend_with_current(
            target, current_oof, current_test, result.predictions, report, args
        )
        ensemble_oof.to_csv(output_dir / f"ensemble_{target}_oof.csv", index=False)
        ensemble_test.to_csv(output_dir / f"ensemble_{target}_{args.report_year}.csv", index=False)
        recipes[target] = recipe
        final_tests[target] = ensemble_test

    combined = final_tests["TA"][KEYS + ["actual_TA", "ensemble_TA"]].merge(
        final_tests["HM"][KEYS + ["actual_HM", "ensemble_HM"]],
        on=KEYS, how="inner", validate="1:1",
    )
    ta_rmse = rmse(combined.actual_TA.to_numpy(float), combined.ensemble_TA.to_numpy(float))
    hm_rmse = rmse(combined.actual_HM.to_numpy(float), combined.ensemble_HM.to_numpy(float))
    combined.to_csv(output_dir / f"final_ensemble_{args.report_year}_predictions.csv", index=False)
    summary = {
        "design": {
            "catboost_hyperparameters_tuned_in_this_experiment": False,
            "feature_selection_years": oof_years,
            "report_year": args.report_year,
            "report_year_used_for_selection": False,
            "allowed_inputs": "GK-2A LE1B KO + official station list + date/time/coordinate calculations",
            "ASOS_usage": "TA/HM labels only; HM TA hint is model-predicted",
        },
        "selected_features": {
            "TA": list(ta_result.features),
            "HM": list(hm_result.features),
        },
        "selected_feature_counts": {"TA": len(ta_result.features), "HM": len(hm_result.features)},
        "team_catboost_OOF": {"TA_RMSE": ta_result.pooled_rmse, "HM_RMSE": hm_result.pooled_rmse},
        "ensemble": recipes,
        "report_metrics": {
            "TA_RMSE": ta_rmse,
            "HM_RMSE": hm_rmse,
            "competition_score": ta_rmse + 0.1 * hm_rmse,
        },
    }
    (output_dir / "team_catboost_feature_ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nOutputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
