#!/usr/bin/env python3
"""Generate the restart-safe Colab notebook for the June LSTM experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Colab_June_LSTM_Feasibility.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


cells = [
    markdown("""# 6월 LSTM 유효성 검증 — Colab 실행본

현재 14시 CatBoost+Ridge 모델에 12:00~14:00 위성 시계열 LSTM을 추가할 가치가 있는지 검증합니다.

- 기간: 2020~2025년 6월 24~30일
- 시각: 12:00, 12:30, 13:00, 13:30, 14:00 KST
- 채널: IR087, IR096, IR105, IR112, IR123, SW038, WV069, WV073
- 선택: 2022~2024 rolling OOF
- 최종 확인: 2025 잠금 검증(가중치 선택에 사용하지 않음)

수집은 중단되어도 캐시부터 이어집니다. 원본 NC는 96지점 추출 성공 후 삭제됩니다.
"""),
    code("""# 1. Drive 마운트와 실험 설정
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/SME_DATA')
WORK_ROOT = DRIVE_ROOT / 'processed_station_features' / 'june_lstm_feasibility'
COLLECTION_ROOT = WORK_ROOT / 'sequence_data'
RESULT_ROOT = WORK_ROOT / 'experiment_results'

# 처음 코드 경로만 확인할 때 True. 실제 판정은 반드시 False로 다시 실행하세요.
QUICK_SMOKE = False
RUN_COLLECTION = True

for path in [WORK_ROOT, COLLECTION_ROOT, RESULT_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

print('WORK_ROOT =', WORK_ROOT)
"""),
    code("""# 2. GitHub 코드 설치 — 이 셀은 Drive 파일을 삭제하지 않습니다
import os, subprocess
from pathlib import Path

REPO_URL = 'https://github.com/tswaincae1221/SME_DATA.git'
BRANCH = 'agent/june-lstm-feasibility'
REPO_DIR = Path('/content/SME_DATA_june_lstm')

if (REPO_DIR / '.git').is_dir():
    subprocess.run(['git', 'fetch', 'origin', BRANCH], cwd=REPO_DIR, check=True)
    subprocess.run(['git', 'checkout', '-B', BRANCH, f'origin/{BRANCH}'], cwd=REPO_DIR, check=True)
else:
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)

subprocess.run([
    'python', '-m', 'pip', 'install', '-q', '-r',
    str(REPO_DIR / 'requirements-june-lstm-feasibility.txt')
], check=True)

print('REPO_DIR =', REPO_DIR)
"""),
    code("""# 3. 입력 파일 자동 탐색 및 검증
import pandas as pd

MASTER_CANDIDATES = [
    DRIVE_ROOT / 'processed_station_features' / 'final_train_dataset_19to25_master.csv',
    DRIVE_ROOT / 'final_train_dataset_19to25_master.csv',
]
MASTER_CSV = next((p for p in MASTER_CANDIDATES if p.is_file()), None)
if MASTER_CSV is None:
    matches = list(DRIVE_ROOT.rglob('final_train_dataset_19to25_master.csv'))
    MASTER_CSV = matches[0] if len(matches) == 1 else None
if MASTER_CSV is None:
    raise FileNotFoundError('Drive의 SME_DATA 아래에서 final_train_dataset_19to25_master.csv를 찾지 못했습니다.')

# 저장소에 포함된 대회 공식 96지점 좌표를 사용합니다.
STATION_LIST = REPO_DIR / 'data' / 'metadata' / 'station_list.csv'
stations = pd.read_csv(STATION_LIST)
master_head = pd.read_csv(MASTER_CSV, nrows=5)
assert len(stations) == 96 and stations.STN_ID.nunique() == 96
assert {'Date','STN_ID','TA','HM'}.issubset(master_head.columns)

SHORTTERM_CSV = COLLECTION_ROOT / 'june_shortterm_core8_2020to2025.csv'
print('MASTER_CSV   =', MASTER_CSV)
print('STATION_LIST =', STATION_LIST)
print('SHORTTERM    =', SHORTTERM_CSV)
"""),
    markdown("""## 4. 위성 수집

API 키는 출력에 남지 않도록 `getpass`로 입력합니다. 기본 설정에서 예상 파일은 1,680개이고, 14시 336개는 master CSV를 재사용하므로 신규 요청은 최대 1,344개입니다.

HTTP 429가 나오면 완료된 캐시는 그대로 보존됩니다. 제한이 풀린 뒤 같은 셀을 다시 실행하면 누락분만 받습니다.
"""),
    code("""# 4. 6월 시퀀스 수집/재개
import getpass, os, subprocess, sys

if RUN_COLLECTION:
    API_KEY = getpass.getpass('기상청 API Hub authKey: ').strip()
    if not API_KEY:
        raise ValueError('API 키가 비어 있습니다.')
    env = os.environ.copy()
    env['KMA_API_KEY'] = API_KEY
    cmd = [
        sys.executable, '-u', str(REPO_DIR / 'scripts' / 'collect_june_lstm_sequences.py'),
        '--master-csv', str(MASTER_CSV),
        '--station-list', str(STATION_LIST),
        '--output-root', str(COLLECTION_ROOT),
        '--years', '2020', '2021', '2022', '2023', '2024', '2025',
        '--start-mmdd', '06-24', '--end-mmdd', '06-30',
        '--times', '12:00', '12:30', '13:00', '13:30', '14:00',
        '--channels', 'IR087', 'IR096', 'IR105', 'IR112', 'IR123', 'SW038', 'WV069', 'WV073',
        '--request-interval-seconds', '0.25',
    ]
    # 키는 환경변수로만 전달하므로 셀 출력과 프로세스 목록에 노출되지 않습니다.
    subprocess.run(cmd, cwd=REPO_DIR, env=env, check=True)
    del API_KEY
else:
    print('RUN_COLLECTION=False: 기존 테이블을 사용합니다.')
"""),
    code("""# 5. 수집 완성도 감사
import json, pandas as pd
from IPython.display import display

summary_path = COLLECTION_ROOT / 'collection_summary.json'
missing_path = COLLECTION_ROOT / 'missing_inventory.csv'
if not summary_path.is_file():
    raise FileNotFoundError('collection_summary.json이 없습니다. 수집 셀을 먼저 실행하세요.')

collection_summary = json.loads(summary_path.read_text(encoding='utf-8'))
display(pd.DataFrame([collection_summary]).T.rename(columns={0:'value'}))
missing = pd.read_csv(missing_path)
print(f"누락 채널-시각 파일: {len(missing):,} / {collection_summary['expected_channel_files']:,}")
if len(missing):
    display(missing.head(30))
if collection_summary['completion_fraction'] < 0.90:
    raise RuntimeError(
        '완성도가 90% 미만입니다. API 제한 해제 후 수집 셀을 재실행하세요. '
        '테이블은 이미 만들어졌지만 이 상태의 LSTM 평가는 신뢰하기 어렵습니다.'
    )
"""),
    markdown("""## 6. 모델 비교

`QUICK_SMOKE=True`는 코드 경로 확인용일 뿐 결과 판정에 쓰면 안 됩니다. 실제 실험은 `False`로 두고 실행합니다. Colab T4 GPU 기준 LSTM은 GPU를 사용하고 CatBoost는 CPU를 사용합니다.
"""),
    code("""# 6. rolling OOF + 2025 잠금 검증
import subprocess, sys

cmd = [
    sys.executable, '-u', str(REPO_DIR / 'scripts' / 'experiment_june_lstm_feasibility.py'),
    '--master-csv', str(MASTER_CSV),
    '--shortterm-csv', str(SHORTTERM_CSV),
    '--station-list', str(STATION_LIST),
    '--output-dir', str(RESULT_ROOT),
    '--selection-years', '2022', '2023', '2024',
    '--report-year', '2025',
    '--sequence-years', '2020', '2021', '2022', '2023', '2024', '2025',
    '--seeds', '42', '43', '44',
    '--device', 'auto', '--threads', '4',
]
if QUICK_SMOKE:
    cmd.append('--quick')
subprocess.run(cmd, cwd=REPO_DIR, check=True)
"""),
    code("""# 7. 결과표와 자동 판정 확인
import json, pandas as pd
from IPython.display import display

comparison = pd.read_csv(RESULT_ROOT / 'model_comparison.csv')
summary = json.loads((RESULT_ROOT / 'experiment_summary.json').read_text(encoding='utf-8'))

display(comparison.pivot_table(index=['period','candidate'], columns='target', values='RMSE'))
print(json.dumps(summary['OOF_decision'], ensure_ascii=False, indent=2))
print('\\n2022~2024 OOF:', json.dumps(summary['selection_competition_score'], ensure_ascii=False, indent=2))
print('\\n2025 잠금 검증:', json.dumps(summary['locked_report_competition_score'], ensure_ascii=False, indent=2))
print('\\nTA recipe:', json.dumps(summary['target_recipes']['TA'], ensure_ascii=False, indent=2))
print('\\nHM recipe:', json.dumps(summary['target_recipes']['HM'], ensure_ascii=False, indent=2))
"""),
    markdown("""## 판정 기준

- TA는 OOF RMSE가 최소 **0.03** 개선되어야 합니다.
- HM은 OOF RMSE가 최소 **0.10** 개선되어야 합니다.
- 2022~2024 중 최소 2개 연도에서 악화되지 않아야 합니다.
- 가중치와 residual scale은 2022~2024에서만 고정합니다.
- 2025는 잠금 검증이며 재조정에 사용하지 않습니다.
- `add_LSTM_to_TA/HM`가 `false`이면 현재 14시 모델을 유지합니다.

주요 결과는 모두 Drive의 `SME_DATA/processed_station_features/june_lstm_feasibility/experiment_results`에 저장됩니다.
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


if __name__ == "__main__":
    pass
