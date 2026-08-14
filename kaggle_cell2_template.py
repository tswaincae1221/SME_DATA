"""Kaggle 제출 노트북에 셀 단위로 옮길 전체 추론 템플릿.

GitHub에는 아래 API_KEY 자리표시자만 둡니다. 최종 비공개 Kaggle 노트북에서
실제 키를 직접 입력하며, 해당 노트북이나 출력은 대회 종료 전에 공개하지 않습니다.
"""

# ╔═══════════════════════════════════════════════╗
# ║  ★ 운영진 수정 구역 ★                        ║
# ╚═══════════════════════════════════════════════╝
PRED_START = "20260624"
PRED_END = "20260630"

# Kaggle Secrets 사용 금지. 최종 비공개 제출본에만 실제 키를 직접 입력합니다.
API_KEY = "REPLACE_WITH_SUBMISSION_API_KEY"

import glob
import subprocess
import sys


# 캐글 기본 환경에 없는 패키지도 새 세션에서 설치되도록 합니다.
REQUIREMENTS_PATH = glob.glob(
    "/kaggle/input/**/requirements-inference.txt", recursive=True
)[0]
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-r", REQUIREMENTS_PATH]
)

import joblib
import pandas as pd
import yaml

CODE_ROOT = glob.glob("/kaggle/input/**/gk2a_weather", recursive=True)[0]
sys.path.insert(0, CODE_ROOT.rsplit("/gk2a_weather", 1)[0])

from gk2a_weather.kaggle import run_kaggle_inference


MODEL_PATH = glob.glob("/kaggle/input/**/baseline.joblib", recursive=True)[0]
STATION_PATH = glob.glob("/kaggle/input/**/station_list.csv", recursive=True)[0]
CONFIG_PATH = glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0]

bundle = joblib.load(MODEL_PATH)
stations = pd.read_csv(STATION_PATH)
with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
    satellite_config = yaml.safe_load(stream)["satellite"]

pred_dates = pd.date_range(PRED_START, PRED_END, freq="D")
if len(pred_dates) == 0:
    raise ValueError("PRED_START는 PRED_END보다 늦을 수 없습니다.")

# ★ 결과 변수명 pred는 대회 규칙상 고정 ★
pred = run_kaggle_inference(
    api_key=API_KEY,
    pred_dates=pred_dates.tolist(),
    stations=stations.copy(),
    bundle=bundle,
    satellite_config=satellite_config,
    observation_hour_kst=14,
)

expected_rows = len(pred_dates) * len(stations)
assert len(pred) == expected_rows
assert pred[["TA", "HM"]].notna().all().all()
assert pred["HM"].between(0, 100).all()
pred.head()
