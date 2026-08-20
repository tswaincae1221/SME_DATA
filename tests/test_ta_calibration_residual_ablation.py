from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiment_ta_calibration_residual_ablation.py"
SPEC = importlib.util.spec_from_file_location("ta_exp123", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_offset_removes_mean_bias() -> None:
    actual = np.array([10.0, 12.0, 14.0])
    prediction = np.array([8.0, 10.0, 12.0])
    correction = MODULE.offset(actual, prediction)
    assert correction == 2.0
    assert abs(np.mean(prediction + correction - actual)) < 1e-12


def test_recipe_outputs_expected_candidates() -> None:
    actual = np.array([20.0, 22.0, 24.0, 26.0])
    frame = pd.DataFrame(
        {
            "Date": [20220824, 20220825, 20230824, 20240824],
            "STN_ID": [90, 90, 90, 90],
            "year": [2022, 2022, 2023, 2024],
            "actual_TA": actual,
            "catboost_TA": actual - 2.0,
            "ridge_TA": actual - 3.0,
            "direct_lstm_TA": actual - 1.0,
            "residual_lstm_TA": np.full(4, 2.0),
            "sequence_missing_cells": [0, 0, 16, 16],
        }
    )
    args = Namespace(weight_step=0.02, residual_scale_max=1.5)
    recipe = MODULE.build_recipe(frame, args)
    scored = MODULE.apply_recipe(frame, recipe)
    assert abs(recipe["catboost_offset"] - 2.0) < 1e-12
    assert abs(recipe["direct_lstm_offset"] - 1.0) < 1e-12
    for candidate in ["Exp1_FullCalibrated", "Exp2_CatResidualLSTM", "Exp3_NoRidge"]:
        assert candidate in scored
        assert MODULE.base.rmse(actual, scored[candidate].to_numpy()) < 1e-10
