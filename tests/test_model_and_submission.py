from __future__ import annotations

import numpy as np
import pandas as pd

from gk2a_weather.features.static import add_calendar_features
from gk2a_weather.models.predict import make_submission, predict_frame
from gk2a_weather.models.train import train_with_group_cv


def _synthetic_training_data() -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(42)
    for day_index, day in enumerate(pd.date_range("2025-07-01", periods=12)):
        for station in range(4):
            satellite = 285 + day_index * 0.3 + station * 0.2
            records.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "STN_ID": station + 100,
                    "latitude": 35 + station * 0.5,
                    "longitude": 127 + station * 0.2,
                    "altitude": station * 20,
                    "IR105_center_mean": satellite,
                    "TA": 25 + day_index * 0.2 - station * 0.1 + rng.normal(0, 0.05),
                    "HM": 70 - day_index * 0.4 + station + rng.normal(0, 0.1),
                }
            )
    return add_calendar_features(pd.DataFrame(records))


def test_train_predict_and_preserve_submission_order() -> None:
    train = _synthetic_training_data()
    bundle = train_with_group_cv(
        train,
        model_config={
            "n_splits": 3,
            "random_state": 42,
            "max_iter": 30,
            "min_samples_leaf": 2,
        },
        feature_config={
            "drop_columns": ["date", "TA", "HM"],
            "categorical_columns": ["STN_ID"],
        },
    )
    features = train.loc[train["date"] == "2025-07-12"].drop(columns=["TA", "HM"])
    pred = predict_frame(bundle, features)
    ids = (
        pred["Date"].astype(str) + "_" + pred["STN_ID"].astype(str)
    ).tolist()[::-1]
    sample = pd.DataFrame({"ID": ids, "TA": 0.0, "HM": 0.0})
    submission = make_submission(pred, sample)
    assert submission["ID"].tolist() == ids
    assert submission[["TA", "HM"]].notna().all().all()
    assert submission["HM"].between(0, 100).all()

