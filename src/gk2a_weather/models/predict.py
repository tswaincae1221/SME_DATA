"""저장된 모델로 pred DataFrame과 submission.csv를 만든다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gk2a_weather.models.bundle import ModelBundle


def align_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in feature_columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[feature_columns]


def predict_frame(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "STN_ID"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"추론 데이터 컬럼이 부족합니다: {sorted(missing)}")
    work = frame.copy().reset_index(drop=True)
    work["STN_ID"] = work["STN_ID"].astype(str)
    aligned = align_features(work, bundle.feature_columns)
    pred_ta = bundle.ta_model.predict(aligned)
    pred_hm = bundle.hm_model.predict(aligned)
    if not np.isfinite(pred_ta).all() or not np.isfinite(pred_hm).all():
        raise ValueError(
            "모델 예측에 비유한값이 있습니다. ASOS 평균 fallback은 개정 규정상 "
            "사용하지 않으므로 학습 파이프라인의 결측 처리를 확인하세요."
        )
    dates = pd.to_datetime(frame["date"]).dt.strftime("%Y%m%d").astype(int)
    return pd.DataFrame(
        {
            "Date": dates,
            "STN_ID": pd.to_numeric(frame["STN_ID"], errors="raise").astype(int),
            "TA": np.clip(pred_ta, -50, 50),
            "HM": np.clip(pred_hm, 0, 100),
        }
    )


def validate_prediction_grid(
    pred: pd.DataFrame, expected_dates: list[int], expected_stations: list[int]
) -> None:
    required = {"Date", "STN_ID", "TA", "HM"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"pred 컬럼 부족: {sorted(missing)}")
    if pred.duplicated(["Date", "STN_ID"]).any():
        raise ValueError("pred에 중복된 Date×STN_ID가 있습니다.")
    expected = {(int(day), int(stn)) for day in expected_dates for stn in expected_stations}
    actual = set(zip(pred["Date"].astype(int), pred["STN_ID"].astype(int)))
    if expected != actual:
        absent = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(f"pred 행 구성 오류: 누락 예시={absent}, 불필요 예시={extra}")
    if pred[["TA", "HM"]].isna().any().any():
        raise ValueError("pred의 TA/HM에 NaN이 있습니다.")
    if not pred["HM"].between(0, 100).all():
        raise ValueError("pred의 HM이 0~100 범위를 벗어났습니다.")


def make_submission(pred: pd.DataFrame, sample_submission: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in sample_submission.columns:
        raise ValueError("sample_submission.csv에 ID 컬럼이 없습니다.")
    key = pred.copy()
    key["ID"] = key["Date"].astype(int).astype(str) + "_" + key["STN_ID"].astype(int).astype(str)
    if key["ID"].duplicated().any():
        raise ValueError("pred에서 중복 ID가 생성됐습니다.")
    submission = sample_submission[["ID"]].merge(
        key[["ID", "TA", "HM"]], on="ID", how="left", validate="one_to_one"
    )
    if submission[["TA", "HM"]].isna().any().any():
        missing_ids = submission.loc[submission["TA"].isna() | submission["HM"].isna(), "ID"]
        raise ValueError(f"제출 예측이 없는 ID가 있습니다: {missing_ids.head().tolist()}")
    submission["TA"] = submission["TA"].clip(-50, 50).round(2)
    submission["HM"] = submission["HM"].clip(0, 100).round(2)
    return submission
