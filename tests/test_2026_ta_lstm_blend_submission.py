from __future__ import annotations

import importlib.util
import json
import sys
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


def test_submission_notebook_cells_compile() -> None:
    path = ROOT / "notebooks" / "Colab_2026_TA_LSTM_Blend_Submission.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "evaluation-period ASOS" not in source
    assert "--download-1400" in source
    assert "TA80_LSTM20_HMbaseline" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{index}", "exec")


def test_submission_alignment_and_frozen_blend_without_torch(
    tmp_path: Path, monkeypatch,
) -> None:
    module = load_script("build_2026_ta_lstm_blend_submission.py")
    stations = pd.read_csv(ROOT / "data" / "metadata" / "station_list.csv")
    dates = [int(value.strftime("%Y%m%d")) for value in pd.date_range("2026-06-24", "2026-06-30")]
    keys = pd.DataFrame(
        [(date, int(station), 2026) for date in dates for station in stations.STN_ID],
        columns=["Date", "STN_ID", "year"],
    )
    baseline = keys.iloc[::-1].reset_index(drop=True).copy()
    baseline["ID"] = baseline.Date.astype(str) + "_" + baseline.STN_ID.astype(str)
    baseline["TA"] = np.linspace(20.0, 35.0, len(baseline))
    baseline["HM"] = np.linspace(40.0, 90.0, len(baseline))
    baseline_path = tmp_path / "baseline.csv"
    baseline[["ID", "TA", "HM"]].to_csv(baseline_path, index=False)

    historical_count = 1001
    historical = {
        "x": np.zeros((historical_count, 5, 8), dtype=np.float32),
        "static": np.zeros((historical_count, 7), dtype=np.float32),
        "keys": pd.DataFrame({"Date": np.arange(historical_count), "STN_ID": 90, "year": 2025}),
        "labels": {"TA": np.linspace(18.0, 32.0, historical_count), "HM": np.full(historical_count, np.nan)},
        "quality": pd.DataFrame({"usable": np.ones(historical_count, dtype=bool)}),
    }
    test = {
        "x": np.zeros((len(keys), 5, 8), dtype=np.float32),
        "static": np.zeros((len(keys), 7), dtype=np.float32),
        "keys": keys,
        "labels": {"TA": np.full(len(keys), np.nan), "HM": np.full(len(keys), np.nan)},
        "quality": keys.assign(missing_fraction=0.0, usable=True),
    }

    calls = iter([historical, test])
    monkeypatch.setattr(module.experiment, "load_sequences", lambda *args, **kwargs: next(calls))
    monkeypatch.setattr(
        module.experiment,
        "train_fixed_predict",
        lambda *args, **kwargs: np.linspace(19.0, 34.0, len(keys)),
    )
    output = tmp_path / "submission.csv"
    diagnostic = tmp_path / "diagnostic.csv"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "build_2026_ta_lstm_blend_submission.py",
        "--historical-shortterm-csv", str(tmp_path / "historical.csv"),
        "--test-shortterm-csv", str(tmp_path / "test.csv"),
        "--baseline-submission-csv", str(baseline_path),
        "--station-list", str(ROOT / "data" / "metadata" / "station_list.csv"),
        "--output-csv", str(output),
        "--diagnostic-csv", str(diagnostic),
        "--summary-json", str(summary),
        "--seeds", "42", "--epochs-by-seed", "1", "--device", "cpu",
    ])
    module.main()

    result = pd.read_csv(output)
    detail = pd.read_csv(diagnostic)
    assert result.columns.tolist() == ["ID", "TA", "HM"]
    assert len(result) == 672 and result.ID.tolist() == baseline.ID.tolist()
    assert np.allclose(result.HM, baseline.HM)
    assert np.allclose(result.TA, 0.8 * baseline.TA + 0.2 * detail.lstm_TA, atol=1e-6)
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["actual_leaderboard_score"] is None
    assert report["rules_contract"]["evaluation_ASOS_used"] is False
