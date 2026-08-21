#!/usr/bin/env python3
"""Target-specific, chronological feature engineering optimizer.

The optimizer starts from the current 40-feature 14:00 tabular baseline, adds
one rules-safe feature family at a time, then removes redundant original/new
features when their removal is non-inferior on rolling OOF years.  TA and HM
are selected independently.

Only GK-2A LE1B channel values, official station LAT/LON/ALT, date/time/solar
geometry, and input missingness are used.  ASOS TA/HM are labels only.  Direct
year terms, ASOS lags, climatology, external weather, and external geography
are deliberately absent.

Selection uses rolling OOF years (default 2022--2024).  The report year
(default 2025) is loaded only after feature selection and is never used to
choose feature families, individual features, alpha, or ensemble weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import experiment_ta_literature_features as literature  # noqa: E402


KEYS = ["Date", "STN_ID"]
CHANNELS = literature.CHANNELS
PAIRS = literature.STANDARDIZED_PAIRS
PAIR_NAMES = [name for _, _, name in PAIRS]
EXPECTED_TIMES = literature.EXPECTED_TIMES
MINUTES = literature.MINUTES_FROM_START

TA_CHANNELS = ["SW038", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112", "IR123", "IR133"]
HM_CHANNELS = ["VI006", "VI008", "SW038", "WV063", "WV069", "WV073", "IR087", "IR112", "IR123", "IR133"]
TA_PAIRS = [
    "clean_window_105_112", "split_window_112_123", "window_105_123",
    "ozone_window_096_112", "cloud_phase_087_112", "low_wv_window_073_112",
    "mid_wv_window_069_112", "fog_window_038_112", "co2_window_133_112",
]
HM_PAIRS = [
    "split_window_112_123", "cloud_phase_087_112", "low_wv_window_073_112",
    "mid_wv_window_069_112", "upper_wv_window_063_112", "fog_window_038_112",
    "co2_window_133_112", "wv_vertical_063_073",
]

BASE_GROUPS = {
    "original_raw_14h": [f"{channel}_last" for channel in CHANNELS],
    "original_pair_14h": [f"{name}_last" for name in PAIR_NAMES],
    "original_station": ["LAT", "LON", "ALT"],
    "original_calendar_solar": [
        "doy_sin", "doy_cos", "cos_solar_zenith", "month", "day", "dayofyear",
        "solar_azimuth", "solar_declination", "local_solar_time", "equation_of_time",
    ],
}


@dataclass
class Evaluation:
    target: str
    features: tuple[str, ...]
    estimator: str
    hyperparameter: float
    pooled_rmse: float
    mean_fold_rmse: float
    std_fold_rmse: float
    latest_fold_rmse: float
    worst_fold_rmse: float
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    mean_abs_coefficients: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortterm-long-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spatial-features-csv", default="")
    parser.add_argument("--targets", nargs="+", choices=["TA", "HM"], default=["TA", "HM"])
    parser.add_argument("--selection-years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--report-year", type=int, default=2025)
    parser.add_argument("--estimator", choices=["ridge", "catboost"], default="ridge")
    parser.add_argument(
        "--selection-mode", choices=["auto", "direct", "baseline_residual"], default="auto",
        help=(
            "auto uses baseline_residual when matching baseline OOF/test files are supplied; "
            "otherwise it predicts the target directly"
        ),
    )
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--cat-iterations", type=int, default=600)
    parser.add_argument("--cat-depth", type=int, default=7)
    parser.add_argument("--cat-learning-rate", type=float, default=0.035)
    parser.add_argument("--cat-l2", type=float, default=8.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward-min-improvement-ta", type=float, default=0.005)
    parser.add_argument("--forward-min-improvement-hm", type=float, default=0.05)
    parser.add_argument("--prune-tolerance-ta", type=float, default=0.003)
    parser.add_argument("--prune-tolerance-hm", type=float, default=0.03)
    parser.add_argument("--correlation-threshold", type=float, default=0.985)
    parser.add_argument("--max-prune-steps", type=int, default=20)
    parser.add_argument("--baseline-ta-oof", default="")
    parser.add_argument("--baseline-ta-test", default="")
    parser.add_argument("--baseline-hm-oof", default="")
    parser.add_argument("--baseline-hm-test", default="")
    parser.add_argument("--ensemble-step", type=float, default=0.02)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_long(frame: pd.DataFrame) -> pd.DataFrame:
    out = literature.validate_input(frame)
    if "HM" not in out:
        raise ValueError("short-term CSV is missing HM label column")
    allowed_label_times = out.loc[out[["TA", "HM"]].notna().any(axis=1), "TimeKST"]
    if len(allowed_label_times) and not allowed_label_times.eq(1400).all():
        raise ValueError("TA/HM must occur only at 14:00; labels cannot be sequence inputs")
    return out


def _finite_slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return 0.0 if valid.sum() == 1 else np.nan
    return float(np.polyfit(MINUTES[valid], values[valid], 1)[0])


def _max_abs_jump(values: np.ndarray) -> float:
    differences = np.diff(values)
    return float(np.nanmax(np.abs(differences))) if np.isfinite(differences).any() else np.nan


def _curvature(values: np.ndarray) -> float:
    early = np.diff(values[:4])
    late = np.diff(values[-4:])
    if not np.isfinite(early).any() or not np.isfinite(late).any():
        return np.nan
    return float(np.nanmean(late) - np.nanmean(early))


def _sign_changes(values: np.ndarray) -> float:
    differences = np.diff(values)
    signs = np.sign(differences[np.isfinite(differences) & (differences != 0)])
    return float(np.sum(signs[1:] != signs[:-1])) if len(signs) >= 2 else 0.0


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask.astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _profile_summaries(prefix: str, values: np.ndarray, row: dict[str, float | int]) -> None:
    valid_step = np.isfinite(values).any(axis=1)
    mean = np.full(len(values), np.nan)
    spread = np.full(len(values), np.nan)
    mean[valid_step] = np.nanmean(values[valid_step], axis=1)
    spread[valid_step] = np.nanmax(values[valid_step], axis=1) - np.nanmin(values[valid_step], axis=1)
    for series_name, series in [("mean", mean), ("spread", spread)]:
        summary = literature.finite_summary(series)
        for statistic in ["mean", "std", "slope", "delta_2h", "delta_30m"]:
            row[f"profile_{prefix}_{series_name}_{statistic}"] = summary[statistic]
        row[f"profile_{prefix}_{series_name}_last_minus_mean"] = (
            float(series[-1] - np.nanmean(series)) if np.isfinite(series[-1]) and np.isfinite(series).any() else np.nan
        )
        row[f"profile_{prefix}_{series_name}_max_abs_jump"] = _max_abs_jump(series)


def advanced_sequence_features(long_frame: pd.DataFrame, scaling_end_year: int) -> pd.DataFrame:
    means, stds = literature.fit_channel_standardisation(long_frame, scaling_end_year)
    rows: list[dict[str, float | int]] = []
    for (date_value, station), group in long_frame.groupby(KEYS, sort=True):
        group = group.sort_values("TimeKST")
        if len(group) != len(EXPECTED_TIMES) or not np.array_equal(group.TimeKST.to_numpy(), EXPECTED_TIMES):
            continue
        z = {
            channel: (group[channel].to_numpy(float) - means[channel]) / stds[channel]
            for channel in CHANNELS
        }
        row: dict[str, float | int] = {"Date": int(date_value), "STN_ID": int(station)}
        for channel, values in z.items():
            valid = values[np.isfinite(values)]
            row[f"adv_{channel}_last_minus_mean_z"] = (
                float(values[-1] - np.mean(valid)) if np.isfinite(values[-1]) and len(valid) else np.nan
            )
            row[f"adv_{channel}_early_std_z"] = float(np.nanstd(values[:7])) if np.isfinite(values[:7]).any() else np.nan
            row[f"adv_{channel}_late_std_z"] = float(np.nanstd(values[-4:])) if np.isfinite(values[-4:]).any() else np.nan
            row[f"adv_{channel}_max_abs_jump_z"] = _max_abs_jump(values)
            row[f"adv_{channel}_curvature_z"] = _curvature(values)
            row[f"adv_{channel}_sign_changes"] = _sign_changes(values)
            missing = ~np.isfinite(values)
            valid_indices = np.flatnonzero(~missing)
            row[f"quality_{channel}_missing_1400"] = int(missing[-1])
            row[f"quality_{channel}_last_valid_lag"] = int(12 - valid_indices[-1]) if len(valid_indices) else 13
            row[f"quality_{channel}_longest_gap"] = _longest_true_run(missing)

        pair_series = {name: z[left] - z[right] for left, right, name in PAIRS}
        profiles = {
            "visible": np.column_stack([z[c] for c in ["VI004", "VI005", "VI006", "VI008"]]),
            "wv": np.column_stack([z[c] for c in ["WV063", "WV069", "WV073"]]),
            "window": np.column_stack([z[c] for c in ["IR105", "IR112", "IR123"]]),
        }
        for name, values in profiles.items():
            _profile_summaries(name, values, row)

        complete_steps = np.array([all(np.isfinite(z[c][i]) for c in CHANNELS) for i in range(13)])
        row["quality_complete_timestep_count"] = int(complete_steps.sum())
        row["quality_longest_incomplete_run"] = _longest_true_run(~complete_steps)
        row["quality_available_channels_1400"] = int(sum(np.isfinite(z[c][-1]) for c in CHANNELS))

        # Explicit interactions are useful to Ridge; CatBoost can decide to ignore them.
        ir112_mean = float(np.nanmean(z["IR112"]))
        visible_mean = float(np.nanmean(profiles["visible"]))
        wv_valid = np.isfinite(profiles["wv"]).any(axis=1)
        wv_spread = np.full(13, np.nan)
        wv_spread[wv_valid] = (
            np.nanmax(profiles["wv"][wv_valid], axis=1)
            - np.nanmin(profiles["wv"][wv_valid], axis=1)
        )
        row["interaction_ir112_mean_x_alt"] = ir112_mean * float(group.ALT.iloc[0])
        row["interaction_ir112_mean_x_coszen"] = ir112_mean * float(literature.solar_features(
            int(date_value), float(group.LON.iloc[0]), float(group.LAT.iloc[0])
        )["cos_solar_zenith"])
        row["interaction_visible_mean_x_coszen"] = visible_mean * float(literature.solar_features(
            int(date_value), float(group.LON.iloc[0]), float(group.LAT.iloc[0])
        )["cos_solar_zenith"])
        row["interaction_wv_spread_x_alt"] = float(np.nanmean(wv_spread)) * float(group.ALT.iloc[0])
        for name in ["fog_window_038_112", "cloud_phase_087_112", "wv_vertical_063_073"]:
            row[f"interaction_{name}_mean_x_coszen"] = float(np.nanmean(pair_series[name])) * float(
                literature.solar_features(int(date_value), float(group.LON.iloc[0]), float(group.LAT.iloc[0]))[
                    "cos_solar_zenith"
                ]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_spatial_features(path: str) -> pd.DataFrame | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    missing = [key for key in KEYS if key not in frame]
    if missing:
        raise ValueError(f"spatial feature CSV is missing keys: {missing}")
    feature_columns = [column for column in frame if column.startswith("spatial_")]
    if not feature_columns:
        raise ValueError("spatial feature CSV needs columns beginning with spatial_")
    if frame.duplicated(KEYS).any():
        raise ValueError("spatial feature CSV has duplicate Date/STN_ID keys")
    frame["Date"] = pd.to_numeric(frame.Date, errors="raise").round().astype("int64")
    frame["STN_ID"] = pd.to_numeric(frame.STN_ID, errors="raise").round().astype("int64")
    return frame[[*KEYS, *feature_columns]]


def build_augmented_table(
    long_frame: pd.DataFrame,
    scaling_end_year: int,
    spatial: pd.DataFrame | None,
) -> pd.DataFrame:
    base = literature.build_feature_table(long_frame, scaling_end_year).frame
    advanced = advanced_sequence_features(long_frame, scaling_end_year)
    labels = long_frame.loc[long_frame.TimeKST.eq(1400), [*KEYS, "HM"]]
    frame = base.merge(advanced, on=KEYS, how="left", validate="1:1").merge(
        labels, on=KEYS, how="left", validate="1:1"
    )
    if spatial is not None:
        frame = frame.merge(spatial, on=KEYS, how="left", validate="1:1")
    return frame


def original_feature_groups() -> dict[str, list[str]]:
    return {name: list(features) for name, features in BASE_GROUPS.items()}


def candidate_groups(frame: pd.DataFrame, target: str) -> dict[str, list[str]]:
    channels = TA_CHANNELS if target == "TA" else HM_CHANNELS
    pairs = TA_PAIRS if target == "TA" else HM_PAIRS
    groups: dict[str, list[str]] = {
        "new_temporal_level": [
            *[f"{channel}_mean_z" for channel in channels],
            *[f"{name}_mean" for name in pairs],
        ],
        "new_temporal_variability": [
            *[f"{channel}_{stat}_z" for channel in channels for stat in ["std", "range"]],
            *[f"{name}_{stat}" for name in pairs for stat in ["std", "range"]],
            *[f"adv_{channel}_{stat}_z" for channel in channels for stat in ["early_std", "late_std", "max_abs_jump"]],
        ],
        "new_temporal_trend": [
            *[f"{channel}_{stat}_z" for channel in channels for stat in ["slope", "delta_2h"]],
            *[f"{name}_{stat}" for name in pairs for stat in ["slope", "delta_2h"]],
        ],
        "new_recent_change": [
            *[f"{channel}_{stat}_z" for channel in channels for stat in ["delta_1h", "delta_30m", "warming_sum"]],
            *[f"{name}_{stat}" for name in pairs for stat in ["delta_1h", "delta_30m", "warming_sum"]],
            *[f"adv_{channel}_{stat}" for channel in channels for stat in ["last_minus_mean_z", "curvature_z", "sign_changes"]],
        ],
        "new_profile_regime": [
            column for column in frame.columns if column.startswith("profile_")
        ],
        "new_physical_interactions": [
            column for column in frame.columns if column.startswith("interaction_")
        ],
        "new_missing_structure": [
            column for column in frame.columns if column.startswith("quality_")
        ],
        "new_spatial_patch": [
            column for column in frame.columns if column.startswith("spatial_")
        ],
    }
    result = {}
    for name, features in groups.items():
        unique = list(dict.fromkeys(feature for feature in features if feature in frame.columns))
        if unique:
            result[name] = unique
    return result


def flatten(groups: dict[str, list[str]], names: Iterable[str] | None = None) -> list[str]:
    chosen = groups.keys() if names is None else names
    return list(dict.fromkeys(feature for name in chosen for feature in groups[name]))


def fit_estimator(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    estimator: str,
    hyperparameter: float,
    args: argparse.Namespace,
):
    if estimator == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(hyperparameter))),
        ])
    else:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("CatBoost mode requires `pip install catboost==1.2.8`") from exc
        model = CatBoostRegressor(
            iterations=args.cat_iterations,
            depth=args.cat_depth,
            learning_rate=args.cat_learning_rate,
            l2_leaf_reg=args.cat_l2,
            loss_function="RMSE",
            random_seed=args.seed,
            thread_count=max(1, args.threads),
            verbose=False,
            allow_writing_files=False,
        )
    model.fit(train[features], train[target])
    return model


def coefficient_importance(model, features: list[str], estimator: str) -> pd.Series:
    if estimator == "ridge":
        values = np.abs(np.asarray(model.named_steps["ridge"].coef_, dtype=float))
    else:
        values = np.asarray(model.get_feature_importance(type="PredictionValuesChange"), dtype=float)
    return pd.Series(values, index=features)


def evaluate(
    tables: dict[int, pd.DataFrame],
    fold_years: list[int],
    features: list[str],
    target: str,
    args: argparse.Namespace,
    hyperparameter: float | None = None,
    baseline_oof: pd.DataFrame | None = None,
) -> Evaluation:
    missing = [feature for feature in features if any(feature not in tables[year] for year in fold_years)]
    if missing:
        raise ValueError(f"features missing from fold tables: {missing[:10]}")
    candidates = (
        [float(hyperparameter)] if hyperparameter is not None
        else ([float(value) for value in args.ridge_alphas] if args.estimator == "ridge" else [float(args.cat_l2)])
    )
    evaluations = []
    for candidate in candidates:
        predictions = []
        metrics = []
        coefficients = []
        for year in fold_years:
            frame = tables[year]
            if baseline_oof is None:
                train = frame[(frame.year <= year - 1) & frame[target].notna()].copy()
                validation = frame[(frame.year == year) & frame[target].notna()].copy()
                model_target = target
            else:
                baseline_column = f"prediction_{target}"
                aligned = frame.merge(
                    baseline_oof[[*KEYS, "year", baseline_column]],
                    on=[*KEYS, "year"], how="inner", validate="1:1",
                )
                aligned["_residual_target"] = aligned[target] - aligned[baseline_column]
                train = aligned[(aligned.year < year) & aligned["_residual_target"].notna()].copy()
                validation = aligned[(aligned.year == year) & aligned["_residual_target"].notna()].copy()
                model_target = "_residual_target"
            if train.empty or validation.empty:
                raise RuntimeError(f"empty {target} fold for {year}")
            model = fit_estimator(train, features, model_target, args.estimator, candidate, args)
            model_prediction = np.asarray(model.predict(validation[features]), dtype=float)
            prediction = (
                model_prediction if baseline_oof is None
                else validation[f"prediction_{target}"].to_numpy(float) + model_prediction
            )
            if target == "HM":
                prediction = np.clip(prediction, 0.0, 100.0)
            actual = validation[target].to_numpy(float)
            score = literature.diagnostic_metrics(actual, prediction)
            metrics.append({"year": year, **score})
            part = validation[KEYS + ["year", target]].copy()
            part["prediction"] = prediction
            predictions.append(part)
            coefficients.append(coefficient_importance(model, features, args.estimator).rename(year))
        prediction_frame = pd.concat(predictions, ignore_index=True)
        metric_frame = pd.DataFrame(metrics)
        pooled = literature.diagnostic_metrics(
            prediction_frame[target].to_numpy(float), prediction_frame.prediction.to_numpy(float)
        )["RMSE"]
        evaluations.append(Evaluation(
            target=target,
            features=tuple(features),
            estimator=args.estimator,
            hyperparameter=candidate,
            pooled_rmse=float(pooled),
            mean_fold_rmse=float(metric_frame.RMSE.mean()),
            std_fold_rmse=float(metric_frame.RMSE.std(ddof=1)),
            latest_fold_rmse=float(metric_frame.loc[metric_frame.year.idxmax(), "RMSE"]),
            worst_fold_rmse=float(metric_frame.RMSE.max()),
            predictions=prediction_frame,
            fold_metrics=metric_frame,
            mean_abs_coefficients=pd.concat(coefficients, axis=1).mean(axis=1),
        ))
    return min(evaluations, key=lambda item: (item.pooled_rmse, item.worst_fold_rmse, item.hyperparameter))


def evaluation_row(stage: str, name: str, result: Evaluation, **extra) -> dict[str, object]:
    return {
        "stage": stage,
        "candidate": name,
        "target": result.target,
        "feature_count": len(result.features),
        "estimator": result.estimator,
        "hyperparameter": result.hyperparameter,
        "pooled_RMSE": result.pooled_rmse,
        "mean_fold_RMSE": result.mean_fold_rmse,
        "std_fold_RMSE": result.std_fold_rmse,
        "latest_fold_RMSE": result.latest_fold_rmse,
        "worst_fold_RMSE": result.worst_fold_rmse,
        **extra,
    }


def forward_select(
    tables: dict[int, pd.DataFrame],
    fold_years: list[int],
    target: str,
    groups: dict[str, list[str]],
    args: argparse.Namespace,
    baseline_oof: pd.DataFrame | None = None,
) -> tuple[list[str], list[str], Evaluation, pd.DataFrame]:
    original_names = list(BASE_GROUPS)
    current_features = flatten(groups, original_names)
    current = evaluate(tables, fold_years, current_features, target, args, baseline_oof=baseline_oof)
    rows = [evaluation_row("forward_baseline", "current_40", current, accepted=True, delta_RMSE=0.0)]
    selected_new: list[str] = []
    remaining = [name for name in groups if name.startswith("new_")]
    minimum = args.forward_min_improvement_ta if target == "TA" else args.forward_min_improvement_hm
    while remaining:
        trials = []
        for name in remaining:
            features = list(dict.fromkeys([*current_features, *groups[name]]))
            result = evaluate(tables, fold_years, features, target, args, baseline_oof=baseline_oof)
            trials.append((name, result))
        best_name, best = min(trials, key=lambda item: (item[1].pooled_rmse, item[1].worst_fold_rmse))
        improvement = current.pooled_rmse - best.pooled_rmse
        for name, result in trials:
            accepted = name == best_name and improvement >= minimum
            rows.append(evaluation_row(
                "forward_add", name, result, accepted=accepted,
                delta_RMSE=result.pooled_rmse - current.pooled_rmse,
            ))
        if improvement < minimum:
            break
        selected_new.append(best_name)
        current_features = list(best.features)
        current = best
        remaining.remove(best_name)
    return current_features, selected_new, current, pd.DataFrame(rows)


def backward_group_prune(
    tables: dict[int, pd.DataFrame],
    fold_years: list[int],
    target: str,
    current_features: list[str],
    active_groups: list[str],
    groups: dict[str, list[str]],
    current: Evaluation,
    args: argparse.Namespace,
    baseline_oof: pd.DataFrame | None = None,
) -> tuple[list[str], list[str], Evaluation, pd.DataFrame]:
    tolerance = args.prune_tolerance_ta if target == "TA" else args.prune_tolerance_hm
    rows = []
    active = list(active_groups)
    while len(active) > 1:
        trials = []
        for name in active:
            removal = set(groups[name])
            features = [feature for feature in current_features if feature not in removal]
            if not features:
                continue
            result = evaluate(
                tables, fold_years, features, target, args, current.hyperparameter,
                baseline_oof=baseline_oof,
            )
            trials.append((name, result))
        if not trials:
            break
        best_name, best = min(trials, key=lambda item: (item[1].pooled_rmse, item[1].worst_fold_rmse))
        acceptable = best.pooled_rmse <= current.pooled_rmse + tolerance
        for name, result in trials:
            rows.append(evaluation_row(
                "backward_group", name, result,
                accepted=bool(name == best_name and acceptable),
                delta_RMSE=result.pooled_rmse - current.pooled_rmse,
            ))
        if not acceptable:
            break
        active.remove(best_name)
        current_features = list(best.features)
        current = best
    return current_features, active, current, pd.DataFrame(rows)


def correlation_components(frame: pd.DataFrame, features: list[str], threshold: float) -> list[list[str]]:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    variable = [feature for feature in features if numeric[feature].nunique(dropna=True) > 1]
    if len(variable) < 2:
        return []
    correlation = numeric[variable].corr().abs()
    adjacency = {feature: set() for feature in variable}
    for left_index, left in enumerate(variable):
        for right in variable[left_index + 1:]:
            value = correlation.at[left, right]
            if np.isfinite(value) and value >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
    components = []
    visited: set[str] = set()
    for feature in variable:
        if feature in visited or not adjacency[feature]:
            continue
        stack = [feature]
        component = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency[node] - visited)
        if len(component) > 1:
            components.append(sorted(component))
    return components


def correlation_prune(
    tables: dict[int, pd.DataFrame],
    fold_years: list[int],
    target: str,
    current_features: list[str],
    current: Evaluation,
    args: argparse.Namespace,
    baseline_oof: pd.DataFrame | None = None,
) -> tuple[list[str], Evaluation, pd.DataFrame, pd.DataFrame]:
    tolerance = args.prune_tolerance_ta if target == "TA" else args.prune_tolerance_hm
    selection_train = tables[max(fold_years)]
    selection_train = selection_train[(selection_train.year <= max(fold_years) - 1) & selection_train[target].notna()]
    trial_rows = []
    cluster_rows = []
    for step in range(1, args.max_prune_steps + 1):
        components = correlation_components(selection_train, current_features, args.correlation_threshold)
        if not components:
            break
        candidates = []
        for cluster_id, component in enumerate(components, start=1):
            ranked = sorted(component, key=lambda feature: (current.mean_abs_coefficients.get(feature, 0.0), feature))
            candidate = ranked[0]
            candidates.append(candidate)
            for feature in component:
                cluster_rows.append({
                    "step": step, "cluster_id": cluster_id, "target": target,
                    "feature": feature, "candidate_for_removal": feature == candidate,
                    "mean_abs_model_importance": current.mean_abs_coefficients.get(feature, np.nan),
                })
        trials = []
        for feature in sorted(set(candidates)):
            remaining = [name for name in current_features if name != feature]
            result = evaluate(
                tables, fold_years, remaining, target, args, current.hyperparameter,
                baseline_oof=baseline_oof,
            )
            trials.append((feature, result))
        best_feature, best = min(trials, key=lambda item: (item[1].pooled_rmse, item[1].worst_fold_rmse))
        acceptable = best.pooled_rmse <= current.pooled_rmse + tolerance
        for feature, result in trials:
            trial_rows.append(evaluation_row(
                "correlation_prune", feature, result,
                prune_step=step, accepted=bool(feature == best_feature and acceptable),
                delta_RMSE=result.pooled_rmse - current.pooled_rmse,
            ))
        if not acceptable:
            break
        current_features = list(best.features)
        current = best
    return current_features, current, pd.DataFrame(trial_rows), pd.DataFrame(cluster_rows)


def final_fit_predict(
    table: pd.DataFrame,
    target: str,
    features: list[str],
    hyperparameter: float,
    report_year: int,
    args: argparse.Namespace,
    baseline_oof: pd.DataFrame | None = None,
    baseline_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if baseline_oof is None:
        train = table[(table.year <= report_year - 1) & table[target].notna()].copy()
        test = table[(table.year == report_year) & table[target].notna()].copy()
        model_target = target
    else:
        if baseline_test is None:
            raise ValueError("baseline residual final fit needs baseline test predictions")
        baseline_column = f"prediction_{target}"
        train = table.merge(
            baseline_oof[[*KEYS, "year", baseline_column]],
            on=[*KEYS, "year"], how="inner", validate="1:1",
        )
        train["_residual_target"] = train[target] - train[baseline_column]
        train = train[train["_residual_target"].notna()].copy()
        test = table[table.year.eq(report_year) & table[target].notna()].merge(
            baseline_test[[*KEYS, baseline_column]], on=KEYS, how="inner", validate="1:1"
        )
        model_target = "_residual_target"
    model = fit_estimator(train, features, model_target, args.estimator, hyperparameter, args)
    model_prediction = np.asarray(model.predict(test[features]), dtype=float)
    prediction = (
        model_prediction if baseline_oof is None
        else test[f"prediction_{target}"].to_numpy(float) + model_prediction
    )
    if target == "HM":
        prediction = np.clip(prediction, 0.0, 100.0)
    result = test[KEYS + ["year", target]].copy()
    result[f"engineered_{target}"] = prediction
    metrics = literature.diagnostic_metrics(test[target].to_numpy(float), prediction)
    return result, metrics


def load_baseline(path: str, target: str, split: str) -> pd.DataFrame | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    prediction = f"prediction_{target}"
    actual = f"actual_{target}"
    required = [*KEYS, prediction]
    if split == "oof":
        required.append("year")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"baseline {target} {split} is missing: {missing}")
    columns = list(dict.fromkeys([*required, actual] if actual in frame else required))
    return frame[columns]


def ensemble_with_baseline(
    oof: Evaluation,
    test: pd.DataFrame,
    target: str,
    baseline_oof: pd.DataFrame | None,
    baseline_test: pd.DataFrame | None,
    step: float,
) -> tuple[pd.DataFrame, dict[str, float]] | tuple[None, None]:
    if baseline_oof is None or baseline_test is None:
        return None, None
    oof_join = oof.predictions.merge(baseline_oof, on=KEYS + ["year"], how="inner", validate="1:1")
    engineered_column = f"engineered_{target}"
    test_join = test.merge(baseline_test, on=KEYS, how="inner", validate="1:1")
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    actual_oof = oof_join[target].to_numpy(float)
    base_oof = oof_join[f"prediction_{target}"].to_numpy(float)
    engineered_oof = oof_join.prediction.to_numpy(float)
    scores = [
        literature.diagnostic_metrics(actual_oof, weight * base_oof + (1.0 - weight) * engineered_oof)["RMSE"]
        for weight in weights
    ]
    index = int(np.argmin(scores))
    weight = float(weights[index])
    final_prediction = (
        weight * test_join[f"prediction_{target}"].to_numpy(float)
        + (1.0 - weight) * test_join[engineered_column].to_numpy(float)
    )
    if target == "HM":
        final_prediction = np.clip(final_prediction, 0.0, 100.0)
    test_join[f"ensemble_{target}"] = final_prediction
    actual = test_join[target].to_numpy(float)
    metrics = {
        "baseline_weight": weight,
        "engineered_weight": 1.0 - weight,
        "selection_OOF_RMSE": float(scores[index]),
        **{f"report_{key}": value for key, value in literature.diagnostic_metrics(actual, final_prediction).items()},
    }
    return test_join, metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    long_frame = validate_long(pd.read_csv(args.shortterm_long_csv))
    spatial = load_spatial_features(args.spatial_features_csv)
    selection_years = sorted(set(args.selection_years))
    if not selection_years or max(selection_years) >= args.report_year:
        raise ValueError("selection years must be earlier than report year")

    print("[BUILD] leakage-safe feature tables", flush=True)
    tables = {
        year: build_augmented_table(long_frame, year - 1, spatial)
        for year in selection_years
    }
    report_table = build_augmented_table(long_frame, args.report_year - 1, spatial)
    reference = tables[max(selection_years)]

    all_forward = []
    all_group_pruning = []
    all_feature_pruning = []
    all_clusters = []
    target_summary = {}
    final_predictions = {}

    for target in args.targets:
        print("\n" + "=" * 96, flush=True)
        print(f"[{target}] forward addition", flush=True)
        baseline_oof = load_baseline(getattr(args, f"baseline_{target.lower()}_oof"), target, "oof")
        baseline_test = load_baseline(getattr(args, f"baseline_{target.lower()}_test"), target, "test")
        use_residual = args.selection_mode == "baseline_residual" or (
            args.selection_mode == "auto" and baseline_oof is not None and baseline_test is not None
        )
        if args.selection_mode == "baseline_residual" and (baseline_oof is None or baseline_test is None):
            raise ValueError(f"{target}: baseline_residual mode requires both OOF and test baseline files")
        selection_baseline = baseline_oof if use_residual else None
        target_selection_years = selection_years
        if use_residual:
            first_baseline_year = int(baseline_oof.year.min())
            target_selection_years = [year for year in selection_years if year > first_baseline_year]
            if not target_selection_years:
                raise ValueError(f"{target}: no residual selection year has an earlier OOF training year")
        groups = {**original_feature_groups(), **candidate_groups(reference, target)}
        features, selected_new, current, forward = forward_select(
            tables, target_selection_years, target, groups, args, selection_baseline
        )
        active_groups = [*BASE_GROUPS.keys(), *selected_new]
        print(f"[{target}] group backward elimination", flush=True)
        features, active_groups, current, group_pruning = backward_group_prune(
            tables, target_selection_years, target, features, active_groups, groups, current, args,
            selection_baseline,
        )
        print(f"[{target}] correlation pruning", flush=True)
        features, current, feature_pruning, clusters = correlation_prune(
            tables, target_selection_years, target, features, current, args, selection_baseline
        )

        final, final_metrics = final_fit_predict(
            report_table, target, features, current.hyperparameter, args.report_year, args,
            selection_baseline, baseline_test if use_residual else None,
        )
        ensemble, ensemble_metrics = ensemble_with_baseline(
            current, final, target, baseline_oof, baseline_test, args.ensemble_step
        )

        forward["target"] = target
        group_pruning["target"] = target
        feature_pruning["target"] = target
        clusters["target"] = target
        all_forward.append(forward)
        all_group_pruning.append(group_pruning)
        all_feature_pruning.append(feature_pruning)
        all_clusters.append(clusters)

        manifest_rows = []
        for group_name, group_features in groups.items():
            for feature in group_features:
                manifest_rows.append({
                    "target": target, "group": group_name,
                    "origin": "original" if group_name.startswith("original_") else "new",
                    "feature": feature, "selected": feature in features,
                })
        pd.DataFrame(manifest_rows).to_csv(output_dir / f"selected_feature_manifest_{target}.csv", index=False)
        current.predictions.to_csv(output_dir / f"selected_oof_predictions_{target}.csv", index=False)
        current.fold_metrics.to_csv(output_dir / f"selected_oof_fold_metrics_{target}.csv", index=False)
        final.to_csv(output_dir / f"report_{args.report_year}_predictions_{target}.csv", index=False)
        if ensemble is not None:
            ensemble.to_csv(output_dir / f"report_{args.report_year}_ensemble_{target}.csv", index=False)
        final_predictions[target] = ensemble if ensemble is not None else final
        target_summary[target] = {
            "selected_new_groups": selected_new,
            "selection_mode": "baseline_residual" if use_residual else "direct",
            "effective_selection_years": target_selection_years,
            "active_groups_after_backward_elimination": active_groups,
            "final_feature_count": len(features),
            "selected_features": features,
            "selection_oof": {
                "pooled_RMSE": current.pooled_rmse,
                "mean_fold_RMSE": current.mean_fold_rmse,
                "std_fold_RMSE": current.std_fold_rmse,
                "latest_fold_RMSE": current.latest_fold_rmse,
                "worst_fold_RMSE": current.worst_fold_rmse,
                "hyperparameter": current.hyperparameter,
            },
            f"report_{args.report_year}_engineered": final_metrics,
            f"report_{args.report_year}_ensemble": ensemble_metrics,
        }
        print(json.dumps(target_summary[target], ensure_ascii=False, indent=2), flush=True)

    pd.concat(all_forward, ignore_index=True).to_csv(output_dir / "forward_addition_trials.csv", index=False)
    pd.concat(all_group_pruning, ignore_index=True).to_csv(output_dir / "backward_group_pruning_trials.csv", index=False)
    pd.concat(all_feature_pruning, ignore_index=True).to_csv(output_dir / "correlation_pruning_trials.csv", index=False)
    pd.concat(all_clusters, ignore_index=True).to_csv(output_dir / "redundancy_clusters.csv", index=False)

    if {"TA", "HM"}.issubset(final_predictions):
        ta = final_predictions["TA"]
        hm = final_predictions["HM"]
        ta_col = "ensemble_TA" if "ensemble_TA" in ta else "engineered_TA"
        hm_col = "ensemble_HM" if "ensemble_HM" in hm else "engineered_HM"
        combined = ta[KEYS + ["TA", ta_col]].merge(
            hm[KEYS + ["HM", hm_col]], on=KEYS, how="inner", validate="1:1"
        )
        ta_rmse = literature.diagnostic_metrics(combined.TA.to_numpy(), combined[ta_col].to_numpy())["RMSE"]
        hm_rmse = literature.diagnostic_metrics(combined.HM.to_numpy(), combined[hm_col].to_numpy())["RMSE"]
        combined.to_csv(output_dir / f"report_{args.report_year}_combined_predictions.csv", index=False)
        target_summary["competition_proxy"] = {
            "TA_RMSE": ta_rmse, "HM_RMSE": hm_rmse,
            "score": ta_rmse + 0.1 * hm_rmse,
        }

    summary = {
        "design": {
            "selection_years": selection_years,
            "report_year": args.report_year,
            "report_year_used_for_selection": False,
            "estimator": args.estimator,
            "direct_year_terms": False,
            "correlation_threshold": args.correlation_threshold,
            "rules": "LE1B channels + official LAT/LON/ALT + date/time/coordinate calculations only; ASOS target-only",
        },
        "targets": target_summary,
    }
    (output_dir / "feature_engineering_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nOutputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
