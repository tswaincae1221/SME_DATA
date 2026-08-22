#!/usr/bin/env python3
"""Generate the Colab notebook for a rules-safe 2026 TA LSTM blend submission."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Colab_2026_TA_LSTM_Blend_Submission.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


cells = [
    markdown("""# 2026-06-24~30 TA LSTM 20% 앙상블 제출 생성

이 노트북은 **평가기간 ASOS TA/HM을 조회하지 않습니다.** 실제 정답은 대회에서 숨겨져 있으므로 로컬에서 리더보드 점수를 계산할 수 없습니다.

수행 내용:

1. 2026년 6월 24~30일의 GK-2A LE1B 위성 5시각·8채널만 수집
2. 2020~2025년 과거 TA 라벨로 direct LSTM 3개 seed 전체학습
3. `TA = 기존 14시 모델 80% + LSTM 20%`
4. `HM = 기존 14시 모델 100%`
5. 672행 제출 CSV 생성

20% 가중치는 2022~2024 rolling OOF에서 이미 고정했으며 2026 정보로 조정하지 않습니다.
"""),
    code("""# 1. Drive 마운트와 경로
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/SME_DATA')
HIST_ROOT = DRIVE_ROOT / 'processed_station_features' / 'june_lstm_feasibility'
HIST_SHORTTERM = HIST_ROOT / 'sequence_data' / 'june_shortterm_core8_2020to2025.csv'
TEST_ROOT = HIST_ROOT / 'sequence_data_2026'
OUTPUT_ROOT = DRIVE_ROOT / 'submission_2026' / 'lstm_ta20_experiment'
MASTER_CSV = DRIVE_ROOT / 'processed_station_features' / 'final_train_dataset_19to25_master.csv'
BASELINE_CSV = DRIVE_ROOT / 'submission_2026' / 'submission_2026_preview_0624_0630.csv'

for path in [TEST_ROOT, OUTPUT_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

print('HIST_SHORTTERM =', HIST_SHORTTERM)
print('BASELINE_CSV   =', BASELINE_CSV)
print('OUTPUT_ROOT    =', OUTPUT_ROOT)
"""),
    code("""# 2. GitHub 코드 설치
import subprocess, sys
from pathlib import Path

REPO_URL = 'https://github.com/tswaincae1221/SME_DATA.git'
BRANCH = 'agent/2026-ta-lstm-blend-submission'
REPO_DIR = Path('/content/SME_DATA_2026_lstm_blend')

if (REPO_DIR / '.git').is_dir():
    subprocess.run(['git', 'fetch', 'origin', BRANCH], cwd=REPO_DIR, check=True)
    subprocess.run(['git', 'checkout', '-B', BRANCH, f'origin/{BRANCH}'], cwd=REPO_DIR, check=True)
else:
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '-r',
    str(REPO_DIR / 'requirements-june-lstm-feasibility.txt')
], check=True)
print('REPO_DIR =', REPO_DIR)
"""),
    code("""# 3. 입력 파일 검증
import pandas as pd

if not MASTER_CSV.is_file():
    matches = list(DRIVE_ROOT.rglob('final_train_dataset_19to25_master.csv'))
    if len(matches) != 1:
        raise FileNotFoundError('final_train_dataset_19to25_master.csv를 한 개로 특정하지 못했습니다.')
    MASTER_CSV = matches[0]
if not HIST_SHORTTERM.is_file():
    raise FileNotFoundError('과거 2020~2025 5시각 시퀀스 CSV가 없습니다.')
if not BASELINE_CSV.is_file():
    raise FileNotFoundError('기존 2026 baseline 제출 CSV가 없습니다.')

STATION_LIST = REPO_DIR / 'data' / 'metadata' / 'station_list.csv'
baseline = pd.read_csv(BASELINE_CSV)
assert baseline.columns.tolist() == ['ID', 'TA', 'HM']
assert len(baseline) == 672 and baseline.ID.nunique() == 672
assert baseline[['TA','HM']].notna().all().all()
print('입력 검증 완료:', len(baseline), 'rows')
"""),
    markdown("""## 4. 2026 위성 시퀀스 수집

호출 대상은 GK-2A `/LE1B/` KO 영역뿐입니다. 평가기간 ASOS API는 호출하지 않습니다.

예상 위성 파일은 `7일 × 5시각 × 8채널 = 280개`입니다. 중단되면 같은 셀을 다시 실행하면 캐시된 파일을 건너뛰고 이어받습니다.
"""),
    code("""# 4. 2026 위성만 수집 — ASOS 조회 없음
import getpass, os, subprocess, sys

API_KEY = getpass.getpass('기상청 API Hub authKey: ').strip()
if not API_KEY:
    raise ValueError('API 키가 비어 있습니다.')
env = os.environ.copy()
env['KMA_API_KEY'] = API_KEY

cmd = [
    sys.executable, '-u', str(REPO_DIR / 'scripts' / 'collect_june_lstm_sequences.py'),
    '--master-csv', str(MASTER_CSV),
    '--station-list', str(STATION_LIST),
    '--output-root', str(TEST_ROOT),
    '--years', '2026', '--start-mmdd', '06-24', '--end-mmdd', '06-30',
    '--times', '12:00', '12:30', '13:00', '13:30', '14:00',
    '--channels', 'IR087', 'IR096', 'IR105', 'IR112', 'IR123', 'SW038', 'WV069', 'WV073',
    '--download-1400', '--request-interval-seconds', '0.25',
]
subprocess.run(cmd, cwd=REPO_DIR, env=env, check=True)
del API_KEY
"""),
    code("""# 5. 수집 완전성 검사
import json, pandas as pd
from IPython.display import display

collection = json.loads((TEST_ROOT / 'collection_summary.json').read_text(encoding='utf-8'))
display(pd.DataFrame([collection]).T.rename(columns={0:'value'}))
missing = pd.read_csv(TEST_ROOT / 'missing_inventory.csv')
if collection['complete_channel_files'] != 280 or len(missing) != 0:
    display(missing.head(30))
    raise RuntimeError('2026 위성 파일이 완전하지 않습니다. 제한 해제 후 수집 셀을 다시 실행하세요.')
assert collection['rules_contract']['asos_used_as_input'] is False
print('2026 위성 수집 완료: 280/280, ASOS 입력 없음')
"""),
    code("""# 6. 과거 전체학습 + 2026 TA LSTM 20% 앙상블
import subprocess, sys

TEST_SHORTTERM = TEST_ROOT / 'june_shortterm_core8_2026to2026.csv'
SUBMISSION_CSV = OUTPUT_ROOT / 'submission_TA80_LSTM20_HMbaseline_20260624_0630.csv'
DIAGNOSTIC_CSV = OUTPUT_ROOT / 'submission_TA80_LSTM20_diagnostic.csv'
SUMMARY_JSON = OUTPUT_ROOT / 'submission_TA80_LSTM20_summary.json'

cmd = [
    sys.executable, '-u', str(REPO_DIR / 'scripts' / 'build_2026_ta_lstm_blend_submission.py'),
    '--historical-shortterm-csv', str(HIST_SHORTTERM),
    '--test-shortterm-csv', str(TEST_SHORTTERM),
    '--baseline-submission-csv', str(BASELINE_CSV),
    '--station-list', str(STATION_LIST),
    '--output-csv', str(SUBMISSION_CSV),
    '--diagnostic-csv', str(DIAGNOSTIC_CSV),
    '--summary-json', str(SUMMARY_JSON),
    '--pred-start', '20260624', '--pred-end', '20260630',
    '--seeds', '42', '43', '44', '--epochs-by-seed', '36', '36', '34',
    '--ta-lstm-weight', '0.20', '--device', 'auto', '--threads', '4',
]
subprocess.run(cmd, cwd=REPO_DIR, check=True)
"""),
    code("""# 7. 제출 파일 최종 감사
import json, pandas as pd
from IPython.display import display

submission = pd.read_csv(SUBMISSION_CSV)
diagnostic = pd.read_csv(DIAGNOSTIC_CSV)
summary = json.loads(SUMMARY_JSON.read_text(encoding='utf-8'))

assert submission.columns.tolist() == ['ID','TA','HM']
assert len(submission) == 672 and submission.ID.nunique() == 672
assert submission[['TA','HM']].notna().all().all()
assert submission.HM.between(0,100).all()
assert submission.ID.tolist() == pd.read_csv(BASELINE_CSV).ID.tolist()

display(submission.head())
display(diagnostic[['baseline_TA','lstm_TA','blend_TA','TA_change']].describe())
print(json.dumps(summary, ensure_ascii=False, indent=2))
print()
print('제출 파일:', SUBMISSION_CSV)
"""),
    markdown("""## 8. 실제 리더보드 점수 확인

생성된 `submission_TA80_LSTM20_HMbaseline_20260624_0630.csv`를 기존 baseline 제출과 동일한 방식으로 Kaggle에 제출하세요.

- 이 노트북은 실제 TA/HM 정답을 조회하지 않습니다.
- 로컬에서 출력되는 과거 점수는 2026 리더보드 점수가 아닙니다.
- Kaggle에 제출한 뒤 표시되는 점수가 유일한 실제 리더보드 점수입니다.
- baseline 점수와 LSTM 20% 점수를 비교할 때는 동일한 제출 기간·제출 형식인지 확인하세요.

LSTM 제출이 개선되면 다음 단계에서 이 TA LSTM을 최종 Kaggle 추론 번들에 포함하고, 악화되면 기존 baseline으로 되돌립니다.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
