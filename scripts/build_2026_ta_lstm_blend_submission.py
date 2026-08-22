#!/usr/bin/env python3
"""Create a rules-safe 2026 submission with a 20% direct TA LSTM blend.

This script never downloads or reads evaluation-period ASOS.  Historical
2020--2025 TA is used only as the training label.  The 2026 inputs are the five
GK-2A LE1B time slots collected by ``collect_june_lstm_sequences.py``.

The blend was frozen from the 2022--2024 rolling OOF experiment:

    TA = 0.80 * current_14h_TA + 0.20 * direct_LSTM_TA
    HM = current_14h_HM

The 2025 report is diagnostic only and is not used to alter the weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import experiment_june_lstm_feasibility as experiment  # noqa: E402
import experiment_team_catboost_feature_ensemble as team  # noqa: E402


KEYS = ["Date", "STN_ID"]
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_EPOCHS = [36, 36, 34]
FROZEN_TA_LSTM_WEIGHT = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-shortterm-csv", required=True)
    parser.add_argument("--test-shortterm-csv", required=True)
    parser.add_argument("--baseline-submission-csv", required=True)
    parser.add_argument("--station-list", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--diagnostic-csv")
    parser.add_argument("--summary-json")
    parser.add_argument("--pred-start", default="20260624")
    parser.add_argument("--pred-end", default="20260630")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs-by-seed", nargs="+", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--ta-lstm-weight", type=float, default=FROZEN_TA_LSTM_WEIGHT)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-sequence-missing-fraction", type=float, default=0.20)
    return parser.parse_args()


def parse_baseline(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["ID", "TA", "HM"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"baseline submission missing columns: {missing}")
    if frame.ID.duplicated().any():
        raise ValueError("baseline submission contains duplicate ID")
    key = frame.ID.astype(str).str.rsplit("_", n=1, expand=True)
    if key.shape[1] != 2:
        raise ValueError("ID must be {YYYYMMDD}_{STN_ID}")
    frame["Date"] = pd.to_numeric(key[0], errors="raise").astype("int64")
    frame["STN_ID"] = pd.to_numeric(key[1], errors="raise").astype("int64")
    frame["TA"] = pd.to_numeric(frame.TA, errors="raise")
    frame["HM"] = pd.to_numeric(frame.HM, errors="raise")
    if frame[required].isna().any().any():
        raise ValueError("baseline submission contains missing values")
    return frame


def expected_dates(start: str, end: str) -> list[int]:
    dates = pd.date_range(
        pd.to_datetime(start, format="%Y%m%d"),
        pd.to_datetime(end, format="%Y%m%d"),
        freq="D",
    )
    if len(dates) == 0:
        raise ValueError("prediction range is empty")
    return [int(value.strftime("%Y%m%d")) for value in dates]


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=args.hidden_size,
        lstm_layers=1,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        threads=args.threads,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    if len(args.seeds) != len(args.epochs_by_seed):
        raise ValueError("--seeds and --epochs-by-seed must have equal lengths")
    if not 0.0 <= args.ta_lstm_weight <= 1.0:
        raise ValueError("--ta-lstm-weight must be between 0 and 1")

    stations = team.clean_station_list(args.station_list)
    historical = experiment.load_sequences(
        args.historical_shortterm_csv, stations, args.max_sequence_missing_fraction
    )
    test = experiment.load_sequences(
        args.test_shortterm_csv, stations, args.max_sequence_missing_fraction
    )

    historical_quality = historical["quality"].usable.to_numpy(dtype=bool)
    historical_label = historical["labels"]["TA"].astype(np.float32)
    train_mask = historical_quality & np.isfinite(historical_label)
    if int(train_mask.sum()) < 1000:
        raise ValueError(f"too few usable historical TA sequences: {int(train_mask.sum())}")

    wanted_dates = expected_dates(args.pred_start, args.pred_end)
    test_keys = test["keys"].copy()
    test_quality = test["quality"].usable.to_numpy(dtype=bool)
    if not test_quality.all():
        bad = test["quality"].loc[~test_quality, KEYS + ["missing_fraction"]]
        raise ValueError(
            "2026 sequence missing fraction exceeds the allowed threshold; examples:\n"
            + bad.head(10).to_string(index=False)
        )
    if sorted(test_keys.Date.unique().tolist()) != wanted_dates:
        raise ValueError(
            f"2026 sequence dates mismatch: got={sorted(test_keys.Date.unique())}, wanted={wanted_dates}"
        )
    expected_rows = len(wanted_dates) * len(stations)
    if len(test_keys) != expected_rows or test_keys[KEYS].duplicated().any():
        raise ValueError(f"expected {expected_rows} unique 2026 sequences, found {len(test_keys)}")

    lstm_parts = []
    fit_args = model_args(args)
    for seed, epochs in zip(args.seeds, args.epochs_by_seed):
        # Keep the same random-seed convention used by the locked 2025 report.
        fit_seed = int(seed) + 2025
        print(
            f"[FINAL TA LSTM] seed={seed}, torch_seed={fit_seed}, epochs={epochs}, "
            f"train_n={int(train_mask.sum())}", flush=True,
        )
        prediction = experiment.train_fixed_predict(
            historical["x"][train_mask],
            historical["static"][train_mask],
            historical_label[train_mask],
            test["x"], test["static"],
            int(epochs), fit_args, fit_seed,
        )
        lstm_parts.append(np.asarray(prediction, dtype=float))
    lstm_ta = np.mean(np.stack(lstm_parts), axis=0)

    lstm_frame = test_keys[KEYS].copy()
    lstm_frame["lstm_TA"] = lstm_ta
    baseline = parse_baseline(args.baseline_submission_csv)
    baseline_dates = sorted(baseline.Date.unique().tolist())
    if baseline_dates != wanted_dates:
        raise ValueError(f"baseline dates mismatch: got={baseline_dates}, wanted={wanted_dates}")
    if len(baseline) != expected_rows:
        raise ValueError(f"baseline must contain {expected_rows} rows, found {len(baseline)}")

    diagnostic = baseline.rename(columns={"TA": "baseline_TA", "HM": "baseline_HM"}).merge(
        lstm_frame, on=KEYS, how="left", validate="1:1"
    )
    if diagnostic.lstm_TA.isna().any():
        raise ValueError("LSTM prediction alignment produced missing rows")
    weight = float(args.ta_lstm_weight)
    diagnostic["blend_TA"] = (
        (1.0 - weight) * diagnostic.baseline_TA + weight * diagnostic.lstm_TA
    )
    diagnostic["blend_HM"] = diagnostic.baseline_HM.clip(0.0, 100.0)
    diagnostic["TA_change"] = diagnostic.blend_TA - diagnostic.baseline_TA

    submission = diagnostic[["ID", "blend_TA", "blend_HM"]].rename(
        columns={"blend_TA": "TA", "blend_HM": "HM"}
    )
    if submission.ID.tolist() != baseline.ID.tolist():
        raise AssertionError("submission order changed relative to the accepted baseline")
    if submission[["TA", "HM"]].isna().any().any():
        raise RuntimeError("submission contains missing TA/HM")
    if not submission.HM.between(0.0, 100.0).all():
        raise RuntimeError("submission HM is outside 0..100")

    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, float_format="%.6f")
    diagnostic_path = Path(args.diagnostic_csv).expanduser().resolve() if args.diagnostic_csv else (
        output_path.with_name(output_path.stem + "_diagnostic.csv")
    )
    diagnostic.to_csv(diagnostic_path, index=False)

    summary = {
        "prediction_period": [args.pred_start, args.pred_end],
        "rows": len(submission),
        "historical_train_sequences": int(train_mask.sum()),
        "lstm_seeds": args.seeds,
        "epochs_by_seed": args.epochs_by_seed,
        "TA_blend": {"baseline_weight": 1.0 - weight, "direct_lstm_weight": weight},
        "HM_blend": {"baseline_weight": 1.0, "lstm_weight": 0.0},
        "prediction_ranges": {
            "baseline_TA": [float(diagnostic.baseline_TA.min()), float(diagnostic.baseline_TA.max())],
            "lstm_TA": [float(diagnostic.lstm_TA.min()), float(diagnostic.lstm_TA.max())],
            "blend_TA": [float(diagnostic.blend_TA.min()), float(diagnostic.blend_TA.max())],
            "HM": [float(diagnostic.blend_HM.min()), float(diagnostic.blend_HM.max())],
        },
        "TA_change": {
            "mean": float(diagnostic.TA_change.mean()),
            "mean_absolute": float(diagnostic.TA_change.abs().mean()),
            "min": float(diagnostic.TA_change.min()),
            "max": float(diagnostic.TA_change.max()),
        },
        "historical_reference": {
            "selection_2022_2024_baseline_score": 3.8741647087669264,
            "selection_2022_2024_TA20_HMbaseline_score": 3.8503460452442125,
            "locked_2025_baseline_score": 3.633517463420529,
            "locked_2025_TA20_HMbaseline_score": 3.5353536287618074,
            "note": "Historical reference is not the 2026 leaderboard score.",
        },
        "rules_contract": {
            "evaluation_ASOS_downloaded": False,
            "evaluation_ASOS_used": False,
            "historical_ASOS_used_as_label_only": True,
            "satellite_path": "/GK2A/LE1B/{channel}/KO/data",
            "official_station_coordinates": True,
        },
        "actual_leaderboard_score": None,
        "leaderboard_note": "Only Kaggle can reveal the hidden-label leaderboard score after submission.",
        "output_csv": str(output_path),
        "diagnostic_csv": str(diagnostic_path),
    }
    # Exact OOF hybrid score: direct-blend TA + baseline HM contribution.
    summary["historical_reference"]["selection_2022_2024_TA20_HMbaseline_score"] = (
        2.4066534837527405 + 0.1 * 14.436925614914719
    )
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else (
        output_path.with_name(output_path.stem + "_summary.json")
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[DONE] {output_path}", flush=True)


if __name__ == "__main__":
    main()
