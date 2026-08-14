from __future__ import annotations

import numpy as np

from gk2a_weather.data.asos import parse_asos_response


def test_parse_asos_response_and_missing_values() -> None:
    response = """
# TM STN ...
202508241400 108 270 2.1 -9 -9 -9 1000.1 1010.2 0 0 31.2 22.0 58
202508241400 159 180 1.0 -9 -9 -9 1002.1 1011.3 0 0 -99 -99 -9
#7777END
"""
    frame = parse_asos_response(response)
    assert frame["STN_ID"].tolist() == [108, 159]
    assert frame.loc[0, "TA"] == 31.2
    assert frame.loc[0, "HM"] == 58.0
    assert np.isnan(frame.loc[1, "TA"])
    assert np.isnan(frame.loc[1, "HM"])
