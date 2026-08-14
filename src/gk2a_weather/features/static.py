"""날짜와 관측소 정보에서 만드는 정적 특징."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result[date_column], errors="raise")
    day_of_year = dates.dt.dayofyear.astype(float)
    result["year"] = dates.dt.year.astype(int)
    result["month"] = dates.dt.month.astype(int)
    result["day"] = dates.dt.day.astype(int)
    result["day_of_year"] = day_of_year.astype(int)
    result["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return result
