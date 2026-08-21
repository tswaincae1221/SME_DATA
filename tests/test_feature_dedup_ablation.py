from __future__ import annotations

import numpy as np
import pandas as pd

import experiment_feature_dedup_ablation as experiment


def test_cumulative_variant_feature_counts() -> None:
    variants = experiment.build_variants()
    assert [variant.step for variant in variants] == [0, 1, 2, 3, 4, 5]
    assert [len(variant.tabular_features) for variant in variants] == [40, 40, 34, 34, 34, 28]
    assert [len(variant.lstm_static_features) for variant in variants] == [8, 8, 8, 7, 4, 4]
    assert variants[1].effective is False


def test_step5_keeps_only_selected_pair_features() -> None:
    step5 = experiment.build_variants()[5]
    observed = [feature for feature in step5.tabular_features if feature in experiment.base.PAIR_FEATURES]
    assert observed == [feature for feature in experiment.base.PAIR_FEATURES if feature in experiment.PAIR_KEEP]
    assert set(observed) == set(experiment.PAIR_KEEP)


def test_hm_scoring_clips_to_physical_range() -> None:
    class Args:
        residual_scale_max = 0.0
        residual_scale_step = 0.02

    oof = pd.DataFrame(
        {
            "year": [2023, 2024],
            "actual_HM": [0.0, 100.0],
            "catboost_HM": [-5.0, 105.0],
            "residual_lstm_HM": [0.0, 0.0],
        }
    )
    test = oof.copy()
    _, scored, _, metrics = experiment.score_target(oof, test, "HM", Args())
    np.testing.assert_allclose(scored.prediction_HM, [0.0, 100.0])
    assert metrics["test_RMSE"] == 0.0
