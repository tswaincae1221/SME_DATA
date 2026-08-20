from __future__ import annotations

import numpy as np
import pandas as pd

import train_incremental_lstm_catboost_ensemble as ensemble


def shortterm_frame() -> pd.DataFrame:
    rows = []
    for time in ensemble.EXPECTED_TIMES:
        row = {
            "Date": 20240824,
            "TimeKST": int(time),
            "STN_ID": 90,
            "LAT": 38.25,
            "LON": 128.56,
            "ALT": 17.5,
            "TA": 28.0 if time == 1400 else np.nan,
            "HM": 70.0 if time == 1400 else np.nan,
        }
        row.update({channel: float(index + time) for index, channel in enumerate(ensemble.CHANNELS)})
        rows.append(row)
    rows[3]["VI004"] = np.nan
    return pd.DataFrame(rows)


def test_build_sequences_preserves_grid_and_missing_count() -> None:
    x, static, y, keys, missing = ensemble.build_sequences(shortterm_frame())

    assert x.shape == (1, 13, 16)
    assert static.shape == (1, 5)
    assert y.tolist() == [[28.0, 70.0]]
    assert keys.iloc[0].to_dict() == {"Date": 20240824, "STN_ID": 90, "year": 2024}
    assert missing.tolist() == [1]


def test_target_specific_weights_can_choose_different_models() -> None:
    actual = np.array([[0.0, 0.0], [1.0, 1.0]])
    lstm = np.array([[0.0, 2.0], [1.0, 3.0]])
    catboost = np.array([[2.0, 0.0], [3.0, 1.0]])

    weights = ensemble.optimize_ensemble_weights(actual, lstm, catboost, step=0.01)

    assert weights == {"TA": 1.0, "HM": 0.0}


def test_align_master_rows_follows_requested_key_order() -> None:
    base = {
        **{channel: 1.0 for channel in ensemble.CHANNELS},
        "LAT": 37.0,
        "LON": 127.0,
        "ALT": 10.0,
        "month": 8,
        "day": 24,
        "dayofyear": 237,
        "doy_sin": -0.8,
        "doy_cos": -0.6,
        "TA": 30.0,
        "HM": 60.0,
    }
    master = pd.DataFrame(
        [
            {"Date": 20240824, "STN_ID": 93, **base},
            {"Date": 20240824, "STN_ID": 90, **base},
        ]
    )
    keys = pd.DataFrame({"Date": [20240824, 20240824], "STN_ID": [90, 93]})

    aligned, indices = ensemble.align_master_rows(master, keys)

    assert aligned["STN_ID"].tolist() == [90, 93]
    assert indices.tolist() == [0, 1]

