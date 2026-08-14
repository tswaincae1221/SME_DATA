from __future__ import annotations

from datetime import datetime

import pytest

from gk2a_weather.data.gk2a import SatelliteTimeBasisError, api_timestamp


def test_api_timestamp_explicit_time_basis() -> None:
    timestamp = datetime(2026, 8, 24, 14, 0)
    assert api_timestamp(timestamp, "kst") == "202608241400"
    assert api_timestamp(timestamp, "utc") == "202608240500"


def test_api_timestamp_requires_review() -> None:
    with pytest.raises(SatelliteTimeBasisError):
        api_timestamp(datetime(2026, 8, 24, 14), "review_required")

