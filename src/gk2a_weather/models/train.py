"""날짜 GroupKFold 기준 모델 학습과 모델 bundle 저장."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from gk2a_weather.models.bundle import ModelBundle
from gk2a_weather.models.evaluate import competition_score


PROHIBITED_ASOS_FEATURE_MARKERS = (
    "ta_lag",
    "hm_lag",
    "recent_ta",
    "recent_hm",
    "station_ta_mean",
    "station_hm_mean",
    "seasonal_ta",
    "seasonal_hm",
    "climatology",
    "climate_normal",
)


def _feature_columns(frame: pd.DataFrame, drop_columns: list[str]) -> list[str]:
    excluded = set(drop_columns) | {"timestamp_kst", "Date", "ID"}
    columns = [column for column in frame.columns if column not in excluded]
    if not columns:
        raise ValueError("학습에 사용할 특징 컬럼이 없습니다.")
    violations = [
        column
        for column in columns
        if any(marker in column.lower() for marker in PROHIBITED_ASOS_FEATURE_MARKERS)
    ]
    if violations:
        raise ValueError(
            "개정 규정상 ASOS 관측·평균·lag 파생값은 추론 특징으로 사용할 수 "
            f"없습니다: {violations}"
        )
    return columns


def _make_pipeline(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    model_config: dict[str, Any],
) -> Pipeline:
    configured = [column for column in categorical_columns if column in feature_columns]
    inferred = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = list(dict.fromkeys(configured + inferred))
    numeric = [column for column in feature_columns if column not in categorical]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("전처리할 특징 컬럼이 없습니다.")

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    estimator = HistGradientBoostingRegressor(
        learning_rate=float(model_config.get("learning_rate", 0.05)),
        max_iter=int(model_config.get("max_iter", 350)),
        max_leaf_nodes=int(model_config.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(model_config.get("min_samples_leaf", 20)),
        l2_regularization=float(model_config.get("l2_regularization", 1.0)),
        random_state=int(model_config.get("random_state", 42)),
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _fit_target(
    frame: pd.DataFrame,
    target: str,
    train_index: np.ndarray,
    feature_columns: list[str],
    categorical_columns: list[str],
    model_config: dict[str, Any],
) -> Pipeline:
    train_rows = frame.iloc[train_index]
    valid_target = pd.to_numeric(train_rows[target], errors="coerce").notna().to_numpy()
    if not valid_target.any():
        raise ValueError(f"{target} 학습용 유효 라벨이 없습니다.")
    selected = train_rows.loc[valid_target]
    model = _make_pipeline(frame, feature_columns, categorical_columns, model_config)
    model.fit(selected[feature_columns], selected[target].astype(float))
    return model


def train_with_group_cv(
    frame: pd.DataFrame,
    *,
    model_config: dict[str, Any],
    feature_config: dict[str, Any],
) -> ModelBundle:
    required = {"date", "STN_ID", "TA", "HM"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"학습 데이터 컬럼이 부족합니다: {sorted(missing)}")

    work = frame.copy().reset_index(drop=True)
    work["STN_ID"] = work["STN_ID"].astype(str)
    drop_columns = list(feature_config.get("drop_columns", ["date", "TA", "HM"]))
    feature_columns = _feature_columns(work, drop_columns)
    categorical_columns = list(feature_config.get("categorical_columns", ["STN_ID"]))
    unique_dates = pd.Series(work["date"].astype(str).unique())
    n_splits = min(int(model_config.get("n_splits", 5)), len(unique_dates))
    if n_splits < 2:
        raise ValueError("날짜 그룹 검증에는 서로 다른 날짜가 최소 2개 필요합니다.")

    oof_ta = np.full(len(work), np.nan)
    oof_hm = np.full(len(work), np.nan)
    fold_metrics: list[dict[str, float]] = []
    splitter = GroupKFold(n_splits=n_splits)

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(work, groups=work["date"].astype(str)), start=1
    ):
        ta_model = _fit_target(
            work,
            "TA",
            train_index,
            feature_columns,
            categorical_columns,
            model_config,
        )
        hm_model = _fit_target(
            work,
            "HM",
            train_index,
            feature_columns,
            categorical_columns,
            model_config,
        )
        valid_x = work.iloc[valid_index][feature_columns]
        oof_ta[valid_index] = ta_model.predict(valid_x)
        oof_hm[valid_index] = np.clip(hm_model.predict(valid_x), 0, 100)
        metrics = competition_score(
            work.iloc[valid_index]["TA"].to_numpy(),
            oof_ta[valid_index],
            work.iloc[valid_index]["HM"].to_numpy(),
            oof_hm[valid_index],
        )
        fold_metrics.append({"fold": fold, **metrics})

    overall = competition_score(
        work["TA"].to_numpy(), oof_ta, work["HM"].to_numpy(), oof_hm
    )
    all_indices = np.arange(len(work))
    final_ta = _fit_target(
        work,
        "TA",
        all_indices,
        feature_columns,
        categorical_columns,
        model_config,
    )
    final_hm = _fit_target(
        work,
        "HM",
        all_indices,
        feature_columns,
        categorical_columns,
        model_config,
    )
    satellite_columns = [
        column
        for column in feature_columns
        if column.startswith(("VI", "NR", "SW", "WV", "IR"))
        and not column.endswith("_missing")
        and not column.endswith("_valid_ratio")
    ]
    return ModelBundle(
        ta_model=final_ta,
        hm_model=final_hm,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        satellite_columns=satellite_columns,
        metrics={"overall": overall, "folds": fold_metrics},
    )


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def load_model_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"예상하지 못한 모델 파일 형식입니다: {type(bundle)}")
    return bundle
