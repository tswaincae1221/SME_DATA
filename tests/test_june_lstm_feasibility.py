from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kst_to_utc_and_core_contract() -> None:
    collector = load_script("collect_june_lstm_sequences.py")
    assert collector.hhmm("12:30") == 1230
    assert collector.api_timestamp(pd.Timestamp("2025-06-24"), 1400) == "202506240500"
    assert collector.DEFAULT_CHANNELS == [
        "IR087", "IR096", "IR105", "IR112", "IR123", "SW038", "WV069", "WV073",
    ]


def test_station_cache_validation(tmp_path: Path) -> None:
    collector = load_script("collect_june_lstm_sequences.py")
    stations = pd.DataFrame({
        "STN_ID": [90, 93], "LAT": [38.2, 37.9], "LON": [128.5, 127.7], "ALT": [10.0, 20.0],
    })
    path = tmp_path / "cache.csv"
    collector.write_station_cache(path, stations, "IR105", np.asarray([1.0, np.nan]))
    assert collector.valid_cache(path, stations, "IR105")
    assert pd.read_csv(path).columns.tolist() == ["STN_ID", "IR105"]


def test_notebook_cells_compile_and_rules_path_is_safe() -> None:
    path = ROOT / "notebooks" / "Colab_June_LSTM_Feasibility.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "ERA5" not in source
    collector_source = (ROOT / "scripts" / "collect_june_lstm_sequences.py").read_text(encoding="utf-8")
    predictor_source = (ROOT / "scripts" / "submission_predictor.py").read_text(encoding="utf-8")
    assert "submission_predictor" in collector_source
    assert "GK2A/LE1B/{channel}/KO/data" in predictor_source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{index}", "exec")


def test_experiment_has_no_direct_year_static_feature() -> None:
    experiment = load_script("experiment_june_lstm_feasibility.py")
    assert "year" not in experiment.STATIC_FEATURES
    assert experiment.TIMES.tolist() == [1200, 1230, 1300, 1330, 1400]
    assert len(experiment.CHANNELS) == 8
