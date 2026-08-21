from __future__ import annotations

import numpy as np
import pandas as pd

import experiment_feature_engineering_optimizer as optimizer
import extract_shortterm_spatial_features as spatial


def test_original_baseline_has_40_rules_safe_features() -> None:
    features = optimizer.flatten(optimizer.original_feature_groups())
    assert len(features) == 40
    assert len(features) == len(set(features))
    assert not {"year", "year_index", "year_index_sq", "TA", "HM"}.intersection(features)


def test_target_specific_candidate_groups_do_not_expose_labels() -> None:
    columns = [
        *optimizer.flatten(optimizer.original_feature_groups()),
        "SW038_mean_z", "WV063_mean_z", "IR112_mean_z",
        "profile_wv_mean_mean", "interaction_ir112_mean_x_alt",
        "quality_complete_timestep_count", "spatial_IR112_r5km_mean",
    ]
    frame = pd.DataFrame(columns=columns)
    for target in ["TA", "HM"]:
        groups = optimizer.candidate_groups(frame, target)
        features = optimizer.flatten(groups)
        assert "TA" not in features
        assert "HM" not in features
        assert "year" not in features


def test_correlation_components_find_transitive_cluster() -> None:
    frame = pd.DataFrame(
        {
            "a": np.arange(20, dtype=float),
            "b": np.arange(20, dtype=float) * 2.0,
            "c": np.arange(20, dtype=float) * -1.0,
            "independent": np.tile([0.0, 1.0], 10),
        }
    )
    components = optimizer.correlation_components(
        frame, ["a", "b", "c", "independent"], threshold=0.99
    )
    assert components == [["a", "b", "c"]]


def test_spatial_column_contract() -> None:
    columns = spatial.expected_columns("IR112", 5.0, 15.0)
    assert len(columns) == 8
    assert all(column.startswith("spatial_IR112_") for column in columns)
    assert "spatial_IR112_center_minus_r15km" in columns
