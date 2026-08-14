from __future__ import annotations

import argparse

import pandas as pd

from _common import ROOT

from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.features.satellite import add_channel_differences
from gk2a_weather.features.static import add_calendar_features
from gk2a_weather.models.predict import make_submission, predict_frame
from gk2a_weather.models.train import load_model_bundle
from gk2a_weather.utils.io import atomic_write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="준비된 평가 특징으로 submission.csv를 만듭니다.")
    parser.add_argument("--features", required=True, help="date, STN_ID가 있는 평가 특징 CSV")
    parser.add_argument("--model", default="models/baseline.joblib")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--sample-submission", help="기본값은 data.yaml의 sample_submission")
    parser.add_argument("--output", default="outputs/submissions/submission.csv")
    args = parser.parse_args()
    config = load_yaml(args.data_config)
    features = pd.read_csv(resolve_project_path(args.features))
    features = add_calendar_features(features)
    features = add_channel_differences(features)
    bundle = load_model_bundle(resolve_project_path(args.model))
    pred = predict_frame(bundle, features)
    sample_path = args.sample_submission or config["project"]["sample_submission"]
    sample = pd.read_csv(resolve_project_path(sample_path))
    submission = make_submission(pred, sample)
    output_path = atomic_write_csv(submission, resolve_project_path(args.output))
    atomic_write_csv(pred, output_path.with_name("pred.csv"))
    print(f"{len(submission)}행 제출 파일 저장: {output_path}")


if __name__ == "__main__":
    main()
