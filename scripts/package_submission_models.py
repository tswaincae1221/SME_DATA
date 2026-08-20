#!/usr/bin/env python3
"""Create the minimal inference-only Kaggle Dataset folder and ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


MODEL_FILES = (
    "feature_spec.json",
    "ensemble_component_weights.csv",
    "lstm_model.pt",
    "lstm_normalization.npz",
    "catboost_TA.cbm",
    "catboost_HM.cbm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zip", dest="zip_path", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    predictor = Path(args.predictor).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    zip_path = Path(args.zip_path).expanduser().resolve()
    for path in [predictor, *(model_dir / name for name in MODEL_FILES)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir in {Path("/").resolve(), model_dir} or len(output_dir.parts) < 4:
        raise ValueError(f"unsafe output directory: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="submission_package_", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        for name in MODEL_FILES:
            shutil.copy2(model_dir / name, staging / name)
        shutil.copy2(predictor, staging / "submission_predictor.py")
        (staging / "README.txt").write_text(
            "Inference-only GK-2A LE1B/KO model package.\n"
            "Add this folder as a Kaggle Dataset input to "
            "submission_LSTM_CatBoost_Ensemble.ipynb.\n"
            "No API key, ASOS query code, evaluation labels, or external geography is included.\n",
            encoding="utf-8",
        )
        manifest = {
            path.name: sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.sha256.json"
        }
        (staging / "manifest.sha256.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(staging, output_dir)
    archive = Path(
        shutil.make_archive(
            str(zip_path.with_suffix("")), "zip", output_dir.parent, output_dir.name
        )
    )
    if archive != zip_path:
        if zip_path.exists():
            zip_path.unlink()
        archive.replace(zip_path)
    print(f"package={output_dir}\nzip={zip_path}")


if __name__ == "__main__":
    main()
