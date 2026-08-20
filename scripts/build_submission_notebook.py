#!/usr/bin/env python3
"""Fill the official submission notebook's free cell with ensemble inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FREE_CELL = r'''# ═══════════════════════════════════════════════════════════════════
#  ↓↓↓ 자유 구현 영역: 추론만 수행합니다 ↓↓↓
# ═══════════════════════════════════════════════════════════════════

import importlib.util
import subprocess

# Kaggle에는 torch가 기본 설치되어 있습니다. 나머지는 없을 때만 설치합니다.
for _module, _package in [
    ("catboost", "catboost==1.2.8"),
    ("xarray", "xarray"),
    ("h5netcdf", "h5netcdf"),
]:
    if importlib.util.find_spec(_module) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _package])

if importlib.util.find_spec("torch") is None:
    raise RuntimeError("PyTorch가 필요합니다. Kaggle 기본 GPU/CPU 이미지를 사용하세요.")

# 학습 산출물 Dataset 폴더를 엄격히 한 개만 선택합니다.
_predictor_hits = sorted(
    glob.glob("/kaggle/input/**/submission_predictor.py", recursive=True)
)
_artifact_candidates = []
_required_model_files = {
    "feature_spec.json",
    "ensemble_component_weights.csv",
    "lstm_model.pt",
    "lstm_normalization.npz",
    "catboost_TA.cbm",
    "catboost_HM.cbm",
}
for _predictor_path in _predictor_hits:
    _candidate = os.path.dirname(_predictor_path)
    if all(os.path.isfile(os.path.join(_candidate, _name)) for _name in _required_model_files):
        _artifact_candidates.append(_candidate)

_artifact_candidates = sorted(set(_artifact_candidates))
if len(_artifact_candidates) != 1:
    raise RuntimeError(
        "제출 모델 Dataset 폴더를 정확히 한 개 찾을 수 있어야 합니다. "
        f"현재 후보={_artifact_candidates}"
    )

MODEL_DIR = _artifact_candidates[0]
_module_path = os.path.join(MODEL_DIR, "submission_predictor.py")
_spec = importlib.util.spec_from_file_location("submission_predictor", _module_path)
_predictor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_predictor)

import torch

# 공식 station_list.csv의 위도·경도·고도만 사용합니다.
station_frame = pd.read_csv(_station_path)

# 2025 결측 실험(70/1,456 파일 = 4.81%)을 반영한 상한입니다.
# API가 전부 막힌 경우에는 평균값 제출로 숨기지 않고 즉시 실패합니다.
MAX_MISSING_FRACTION = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pred = _predictor.predict_from_api(
    api_key=API_KEY,
    pred_dates=PRED_DATES,
    stations=station_frame,
    artifact_dir=MODEL_DIR,
    to_api_datetime=to_api_datetime,
    cache_dir="/kaggle/working/gk2a_cache",
    failure_csv="/kaggle/working/satellite_download_failures.csv",
    device=DEVICE,
    max_missing_fraction=MAX_MISSING_FRACTION,
)

print(f"model_dir={MODEL_DIR}")
print(f"device={DEVICE}, pred_rows={len(pred)}")
print(pred.head().to_string(index=False))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    template = Path(args.template)
    output = Path(args.output)
    notebook = json.loads(template.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if len(cells) < 7 or cells[4].get("cell_type") != "code":
        raise ValueError("unexpected official notebook layout: free code cell 4 not found")
    cells[4]["source"] = FREE_CELL.splitlines(keepends=True)
    cells[4]["execution_count"] = None
    cells[4]["outputs"] = []
    notebook.setdefault("metadata", {}).setdefault("colab", {})["name"] = output.name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
