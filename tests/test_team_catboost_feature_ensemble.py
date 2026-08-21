from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import experiment_team_catboost_feature_ensemble as experiment
import confirm_team_catboost_feature_ensemble_multiseed as confirmation


def station_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "STN_ID": [1, 2, 3, 4],
        "LAT": [37.0, 37.1, 37.2, 37.3],
        "LON": [127.0, 127.1, 127.2, 127.3],
        "ALT": [10.0, 20.0, 30.0, 40.0],
    })


def test_nearest_station_ids_excludes_self():
    neighbours = experiment.nearest_station_ids(station_frame(), count=3)
    assert set(neighbours) == {1, 2, 3, 4}
    assert all(station not in values for station, values in neighbours.items())
    assert all(len(values) == 3 for values in neighbours.values())


def test_same_date_neighbour_features_do_not_cross_dates():
    frame = pd.DataFrame({
        "Date": [20240101] * 4 + [20240102] * 4,
        "STN_ID": [1, 2, 3, 4] * 2,
        "IR105": [1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0],
    })
    out = experiment.add_same_date_neighbour_features(
        frame, experiment.nearest_station_ids(station_frame()), ["IR105"]
    )
    first = out[(out.Date == 20240101) & (out.STN_ID == 1)].iloc[0]
    second = out[(out.Date == 20240102) & (out.STN_ID == 1)].iloc[0]
    assert first.team_nearest3_mean_ir105 == 3.0
    assert second.team_nearest3_mean_ir105 == 103.0


def test_feature_contract_rejects_year_and_lag():
    for feature in ["year", "Lag1_IR105", "actual_TA"]:
        try:
            experiment.validate_feature_contract([feature], "TA")
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected rejection: {feature}")


def test_hm_groups_use_predicted_hint_not_observed_ta():
    flattened = [feature for _, features, _ in experiment.feature_groups("HM") for feature in features]
    assert "TA" not in flattened
    assert experiment.TA_HINT_FEATURES == ["team_oof_ta_hint", "team_oof_ta_hint_missing"]


def test_blend_weight_is_target_independent_math():
    actual = np.array([0.0, 1.0, 2.0])
    current = np.array([0.0, 0.0, 0.0])
    candidate = actual.copy()
    weight, score = experiment.optimize_blend_weight(actual, current, candidate, 0.1)
    assert weight == 1.0
    assert score == 0.0


def test_audit_explicitly_excludes_future_fill_and_year():
    audit = experiment.audit_rows()
    excluded = audit[audit.decision.eq("excluded")]
    ideas = set(excluded.source_idea)
    assert "Year" in ideas
    assert "bfill validation features" in ideas


def test_validation_mask_limits_scoring_window_but_can_be_disabled():
    frame = pd.DataFrame({"year": [2024, 2024, 2024], "Date": [20240823, 20240824, 20240830]})
    args = Namespace(validation_start_mmdd=824, validation_end_mmdd=830)
    assert experiment.validation_mask(frame, 2024, args, True).tolist() == [False, True, True]
    assert experiment.validation_mask(frame, 2024, args, False).tolist() == [True, True, True]


def test_frozen_multiseed_features_exclude_unstable_groups():
    features = confirmation.FROZEN_HM_FEATURES
    assert "STN_ID_cat" not in features
    assert all(feature not in features for feature in experiment.TA_HINT_FEATURES)
    assert all(feature in features for feature in experiment.PHYSICAL_FEATURES)
    assert all(feature in features for feature in experiment.SPATIAL_FEATURES)
