from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from _common import ROOT

from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.models.train import save_model_bundle, train_with_group_cv
from gk2a_weather.utils.io import atomic_write_csv, atomic_write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="날짜 GroupKFold 기준 모델을 학습합니다.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--dataset", help="기본값은 data.yaml의 processed_dataset")
    parser.add_argument("--output", default="models/baseline.joblib")
    parser.add_argument("--experiment-id", default="B1")
    args = parser.parse_args()
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    dataset_path = resolve_project_path(
        args.dataset or data_config["project"]["processed_dataset"]
    )
    frame = pd.read_csv(dataset_path)
    bundle = train_with_group_cv(
        frame,
        model_config=model_config["model"],
        feature_config=model_config["features"],
    )
    output_path = save_model_bundle(bundle, resolve_project_path(args.output))
    metrics_path = output_path.with_suffix(".metrics.json")
    atomic_write_text(metrics_path, json.dumps(bundle.metrics, ensure_ascii=False, indent=2))

    log_path = resolve_project_path("experiments/experiment_log.csv")
    existing = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    overall = bundle.metrics["overall"]
    row = pd.DataFrame(
        [
            {
                "experiment_id": args.experiment_id,
                "date": datetime.now().isoformat(timespec="seconds"),
                "data_version": dataset_path.name,
                "features": f"{len(bundle.feature_columns)} columns",
                "model": "HistGradientBoostingRegressor",
                "validation": "GroupKFold(date)",
                "rmse_ta": overall["rmse_ta"],
                "rmse_hm": overall["rmse_hm"],
                "score": overall["score"],
                "seed": model_config["model"].get("random_state", 42),
                "notes": "",
            }
        ]
    )
    updated_log = row if existing.empty else pd.concat([existing, row], ignore_index=True)
    atomic_write_csv(updated_log, log_path)
    print(json.dumps(bundle.metrics, ensure_ascii=False, indent=2))
    print(f"모델 저장: {output_path}")


if __name__ == "__main__":
    main()
