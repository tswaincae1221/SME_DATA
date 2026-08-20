from __future__ import annotations

import numpy as np
import pandas as pd

import train_ta_hm_oof_calibrated_baseline as experiment


def _ta_frame(year: int, actual: list[float], prediction: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": [year * 10000 + 824 + index for index in range(len(actual))],
            "STN_ID": [90 + index for index in range(len(actual))],
            "year": year,
            "actual_TA": actual,
            experiment.TA_BASE_COLUMN: prediction,
        }
    )


def test_latest_oof_ta_calibration_uses_latest_fold_residual() -> None:
    oof = pd.concat(
        [
            _ta_frame(2023, [20.0, 21.0], [20.0, 21.0]),
            _ta_frame(2024, [22.0, 24.0], [20.0, 22.0]),
        ],
        ignore_index=True,
    )
    test = _ta_frame(2025, [25.0, 26.0], [23.0, 24.0])
    _, calibrated_test, recipe, _ = experiment.calibrate_ta(oof, test)

    assert recipe["latest_oof_year"] == 2024
    assert recipe["additive_correction"] == 2.0
    np.testing.assert_allclose(calibrated_test.TA_LatestOOFCalibrated, [25.0, 26.0])
    assert recipe["test_improved"] is True


def test_hm_calibrator_clips_relative_humidity() -> None:
    frame = pd.DataFrame(
        {
            "HM_CatResidualRaw": [-5.0, 55.0, 110.0],
            "TA_LatestOOFCalibrated": [20.0, 25.0, 30.0],
        }
    )
    prediction = experiment.apply_hm_calibrator("raw", None, frame)
    np.testing.assert_allclose(prediction, [0.0, 55.0, 100.0])


def test_competition_score_matches_official_formula() -> None:
    ta = pd.DataFrame(
        {
            "Date": [20250824, 20250825],
            "STN_ID": [90, 90],
            "actual_TA": [20.0, 22.0],
        }
    )
    hm = pd.DataFrame(
        {
            "Date": [20250824, 20250825],
            "STN_ID": [90, 90],
            "actual_HM": [50.0, 60.0],
            "TA_PooledOOF": [21.0, 21.0],
            "TA_LatestOOFCalibrated": [20.0, 22.0],
            "HM_CatResidualRaw": [55.0, 55.0],
            "HM_SelectedCalibrated": [50.0, 60.0],
        }
    )
    scores = experiment.competition_scores(ta, hm).set_index("system")
    assert scores.loc["final_selected", "competition_score"] == 0.0
    expected = 1.0 + 0.1 * 5.0
    assert scores.loc["before_TA_calibration_plus_HM_raw", "competition_score"] == expected
