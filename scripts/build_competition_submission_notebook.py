#!/usr/bin/env python3
"""Build the Kaggle submission notebook from the official supplied template.

Only the template's free-implementation cell (cell 4) is replaced.  All
organizer-controlled setup, audit, validation, and submission cells remain
byte-for-byte equivalent at the JSON field level.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT.parent / "upload" / "submission_notebook_2(2).ipynb"
OUTPUT = ROOT / "notebooks" / "submission_2026_competition_ready.ipynb"


FREE_IMPLEMENTATION = r'''# ═══════════════════════════════════════════════════════════════════
#  ↓↓↓ 자유 구현 영역: 학습 없이 저장 모델로 추론만 수행 ↓↓↓
# ═══════════════════════════════════════════════════════════════════

import importlib.util
import json as _json
import os
import zipfile

# Kaggle 기본 이미지에 패키지가 있으면 그대로 사용하고, 없는 것만 설치합니다.
# 공식 템플릿 지침대로 Internet 옵션이 On이어야 위성 API와 설치가 동작합니다.
_missing_packages = []
for _package in ["catboost", "xarray", "h5netcdf", "torch", "joblib", "sklearn"]:
    try:
        __import__(_package)
    except ImportError:
        _missing_packages.append(_package)

if _missing_packages:
    import subprocess
    _pip_names = {
        "sklearn": "scikit-learn==1.8.0",
        "catboost": "catboost==1.2.8",
        "xarray": "xarray",
        "h5netcdf": "h5netcdf",
        "torch": "torch",
        "joblib": "joblib",
    }
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"]
        + [_pip_names[name] for name in _missing_packages],
        check=True,
    )

# Add Input으로 연결한 모델 Dataset에서 유효한 2026 번들을 정확히 하나 찾습니다.
_bundle_candidates = []
for _manifest_path in glob.glob("/kaggle/input/**/manifest.json", recursive=True):
    try:
        with open(_manifest_path, encoding="utf-8") as _handle:
            _manifest = _json.load(_handle)
        if str(_manifest.get("bundle_version", "")).startswith("2026."):
            _bundle_candidates.append(os.path.dirname(_manifest_path))
    except (OSError, ValueError, TypeError):
        continue

# Dataset에 압축된 번들만 들어 있는 경우 작업 폴더에 압축을 풉니다.
if not _bundle_candidates:
    _bundle_zips = sorted(glob.glob(
        "/kaggle/input/**/sme_submission_2026_model_bundle_v1.zip",
        recursive=True,
    ))
    if len(_bundle_zips) == 1:
        _extract_dir = "/kaggle/working/submission_2026_model_bundle"
        os.makedirs(_extract_dir, exist_ok=True)
        with zipfile.ZipFile(_bundle_zips[0]) as _archive:
            _archive.extractall(_extract_dir)
        if os.path.isfile(os.path.join(_extract_dir, "manifest.json")):
            _bundle_candidates = [_extract_dir]

_bundle_candidates = sorted(set(_bundle_candidates))
if len(_bundle_candidates) != 1:
    raise RuntimeError(
        "2026 모델 번들을 정확히 하나 찾지 못했습니다. "
        f"발견 경로={_bundle_candidates}. 우측 Add Input을 확인하세요."
    )

BUNDLE_DIR = _bundle_candidates[0]
_module_path = os.path.join(BUNDLE_DIR, "submission_2026_inference.py")
if not os.path.isfile(_module_path):
    raise FileNotFoundError(f"추론 모듈 없음: {_module_path}")

# 번들에 저장된 학습 완료 모델과 전처리 정의를 불러옵니다.
_spec = importlib.util.spec_from_file_location("submission_2026_inference", _module_path)
_inference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inference)

_station_frame = pd.read_csv(_station_path)
_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# PRED_DATES를 그대로 사용합니다. 날짜 하드코딩 및 평가기간 재학습은 없습니다.
# 요청 경로는 번들 내부에서 GK2A/LE1B/{channel}/KO/data로 제한됩니다.
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

if len(pred) != len(PRED_DATES) * len(STATIONS):
    raise RuntimeError(
        f"예측 행 수 오류: {len(pred)} != {len(PRED_DATES)} × {len(STATIONS)}"
    )
if pred[["TA", "HM"]].isna().any().any():
    raise RuntimeError("모델 예측에 TA/HM 결측값이 있습니다.")

print(f"모델 번들: {BUNDLE_DIR}")
print(f"추론 장치: {_device}")
print(pred[["TA", "HM"]].describe().to_string())
'''


def main() -> None:
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) != 10:
        raise ValueError("공식 템플릿 셀 수가 예상과 다릅니다.")
    if notebook["cells"][4].get("cell_type") != "code":
        raise ValueError("공식 템플릿의 자유 구현 셀 위치가 변경되었습니다.")

    notebook["cells"][4]["source"] = FREE_IMPLEMENTATION.splitlines(keepends=True)
    notebook["cells"][4]["execution_count"] = None
    notebook["cells"][4]["outputs"] = []

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
