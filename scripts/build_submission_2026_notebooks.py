#!/usr/bin/env python3
"""Build the inference-only Kaggle notebook and the Colab bundle notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown", "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


INFERENCE_CELL = r'''
# 추론 환경과 모델 번들 로드 (학습/파인튜닝 없음)
import importlib.util
import subprocess

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "catboost==1.2.8", "xarray", "h5netcdf"],
    check=True,
)

_bundle_candidates = []
for _manifest in glob.glob("/kaggle/input/**/manifest.json", recursive=True):
    try:
        import json as _json
        with open(_manifest, encoding="utf-8") as _handle:
            _meta = _json.load(_handle)
        if str(_meta.get("bundle_version", "")).startswith("2026."):
            _bundle_candidates.append(os.path.dirname(_manifest))
    except Exception:
        continue

if len(_bundle_candidates) != 1:
    raise RuntimeError(
        f"2026 모델 번들은 정확히 하나여야 합니다: {_bundle_candidates}. "
        "Kaggle Dataset 연결을 확인하세요."
    )

BUNDLE_DIR = _bundle_candidates[0]
_module_path = os.path.join(BUNDLE_DIR, "submission_2026_inference.py")
if not os.path.isfile(_module_path):
    raise FileNotFoundError(_module_path)

_spec = importlib.util.spec_from_file_location("submission_2026_inference", _module_path)
_inference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inference)

_station_frame = pd.read_csv(_station_path)
_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# PRED_DATES의 매일 12:00~14:00 KST, 10분 간격 LE1B/KO 16채널만 사용합니다.
# 위성 API date는 설정 셀의 to_api_datetime()으로 UTC 변환됩니다.
pred = _inference.predict_from_api(
    api_key=API_KEY,
    pred_dates=PRED_DATES,
    stations=_station_frame,
    bundle_dir=BUNDLE_DIR,
    to_api_datetime=to_api_datetime,
    cache_dir="/kaggle/working/gk2a_cache",
    failure_csv="/kaggle/working/gk2a_failures.csv",
    device=_device,
    max_missing_fraction=0.05,
)

pred = pred[["Date", "STN_ID", "TA", "HM"]].copy()
print(pred.describe(include="all").to_string())
'''


def build_inference(guide: Path, output: Path) -> None:
    notebook = json.loads(guide.read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) < 7:
        raise ValueError("submission guide has an unexpected cell layout")
    # Only the organiser-designated free implementation cell is replaced.
    notebook["cells"][4] = code_cell(INFERENCE_CELL)
    notebook.setdefault("metadata", {}).setdefault("kernelspec", {
        "display_name": "Python 3", "language": "python", "name": "python3"
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def build_colab(output: Path) -> None:
    cells = [
        markdown_cell('''# 2026 제출 모델 번들 생성

이 노트북은 **제출 전에 한 번만** 실행하는 학습용 노트북입니다. 2019~2025 학습자료로 동결된 3-seed 모델을 재학습하고, Kaggle Dataset으로 업로드할 ZIP을 Google Drive에 만듭니다. 제출 노트북에는 이 학습 코드가 들어가지 않습니다.'''),
        code_cell('''from google.colab import drive
drive.mount("/content/drive")'''),
        code_cell('''import os, subprocess, sys

REPO_DIR = "/content/SME_DATA_submission_2026"
BRANCH = "agent/submission-2026-end-to-end"
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", BRANCH,
        "https://github.com/tswaincae1221/SME_DATA.git", REPO_DIR,
    ], check=True)
else:
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "checkout", BRANCH], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "pull", "--ff-only"], cwd=REPO_DIR, check=True)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "catboost==1.2.8", "torch", "scikit-learn", "pandas", "numpy",
], check=True)
print("repo:", REPO_DIR)'''),
        code_cell('''from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/SME_DATA")
MASTER_CSV = DRIVE_ROOT / "processed_station_features/final_train_dataset_19to25_master.csv"
SHORTTERM_CSV = DRIVE_ROOT / "processed_station_features/shortterm_12to14_data/incremental_12to14_tables/shortterm_long_2019to2025.csv"

station_candidates = sorted(DRIVE_ROOT.rglob("station_list.csv"))
if not station_candidates:
    raise FileNotFoundError("SME_DATA 아래 station_list.csv를 찾지 못했습니다")
STATION_CSV = station_candidates[0]
OUTPUT_DIR = DRIVE_ROOT / "submission_2026/model_bundle_v1"

for path in [MASTER_CSV, SHORTTERM_CSV, STATION_CSV]:
    if not path.is_file():
        raise FileNotFoundError(path)
print("master:", MASTER_CSV)
print("shortterm:", SHORTTERM_CSV)
print("station:", STATION_CSV)
print("output:", OUTPUT_DIR)'''),
        code_cell('''import subprocess, sys

oof_root = Path(REPO_DIR) / "assets/submission_2026"
cmd = [
    sys.executable, "-u", str(Path(REPO_DIR) / "scripts/build_submission_2026_bundle.py"),
    "--master-csv", str(MASTER_CSV),
    "--shortterm-long-csv", str(SHORTTERM_CSV),
    "--station-list", str(STATION_CSV),
    "--baseline-oof-dirs",
    str(oof_root / "seed_42"), str(oof_root / "seed_43"), str(oof_root / "seed_44"),
    "--output-dir", str(OUTPUT_DIR),
    "--threads", "4", "--device", "auto",
]
print(" ".join(cmd))
subprocess.run(cmd, cwd=REPO_DIR, check=True)'''),
        code_cell('''import json, shutil

manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
required = [OUTPUT_DIR / "manifest.json", OUTPUT_DIR / "submission_2026_inference.py"]
for seed in manifest["seeds"]:
    root = OUTPUT_DIR / f"seed_{seed}"
    required += [
        root / "base_catboost_TA.cbm", root / "base_catboost_HM.cbm",
        root / "residual_lstm_TA.pt", root / "residual_lstm_HM.pt",
        root / "ta_residual_ridge.joblib", root / "june_direct_ridge_TA.joblib",
        root / "team_spatial_catboost_HM.cbm",
    ]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError("bundle files missing:\n" + "\n".join(missing))

archive_base = str(OUTPUT_DIR.parent / "sme_submission_2026_model_bundle_v1")
archive = shutil.make_archive(archive_base, "zip", OUTPUT_DIR)
print("Kaggle Dataset upload ZIP:", archive)
print(json.dumps(manifest["historical_report"], ensure_ascii=False, indent=2))'''),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--inference-output", required=True)
    parser.add_argument("--colab-output", required=True)
    args = parser.parse_args()
    build_inference(Path(args.guide), Path(args.inference_output))
    build_colab(Path(args.colab_output))


if __name__ == "__main__":
    main()
