# 2026년 6월 24~30일 TA LSTM 앙상블 제출 가이드

## 결론

평가기간의 실제 ASOS 기온·습도는 대회 규정상 조회하거나 입력으로 사용할 수 없다. 따라서 이 실험은 2026년 GK-2A LE1B 위성 시퀀스만 새로 수집하고, 과거 2020~2025년 ASOS 기온을 학습 라벨로만 사용한다.

최종 실험 제출식은 다음과 같이 고정한다.

- `TA = 0.80 × 기존 14시 모델 + 0.20 × direct LSTM`
- `HM = 기존 14시 모델`

LSTM 가중치 0.20은 2022~2024 rolling OOF에서 선택했으며, 2025년 또는 2026년 결과를 보고 다시 조정하지 않는다.

## 과거 검증 근거

| 검증 구간 | 기존 모델 점수 | TA LSTM 20% + 기존 HM | 변화 |
|---|---:|---:|---:|
| 2022~2024 rolling OOF | 3.874165 | 3.850346 | -0.023819 |
| 2025 locked report | 3.633517 | 3.535354 | -0.098164 |

점수는 낮을수록 좋다. direct LSTM 단독은 기존 모델보다 나빴지만 오차가 완전히 같지 않아, TA에 20%만 섞었을 때 두 검증 구간에서 모두 개선됐다. HM LSTM은 개선 근거가 약해 제출 실험에서 제외한다.

## Colab 실행

1. Google Drive에 다음 세 파일이 존재하는지 확인한다.
   - `SME_DATA/processed_station_features/final_train_dataset_19to25_master.csv`
   - `SME_DATA/processed_station_features/june_lstm_feasibility/sequence_data/june_shortterm_core8_2020to2025.csv`
   - `SME_DATA/submission_2026/submission_2026_preview_0624_0630.csv`
2. `notebooks/Colab_2026_TA_LSTM_Blend_Submission.ipynb`를 Colab에서 연다.
3. GPU 런타임을 선택하고 위에서 아래로 실행한다.
4. 기상청 API Hub 키는 프롬프트에 입력한다. 노트북이나 출력 파일에는 키를 저장하지 않는다.
5. API 제한으로 중단되면 동일한 수집 셀을 다시 실행한다. 완료 캐시는 건너뛰며 누락 파일만 다시 받는다.
6. 최종 제출 파일은 다음 경로에 생성된다.
   - `SME_DATA/submission_2026/lstm_ta20_experiment/submission_TA80_LSTM20_HMbaseline_20260624_0630.csv`

## 수집 및 제출 감사 조건

- 위성 호출 경로: `/GK2A/LE1B/{channel}/KO/data`
- 영역: `KO`
- 기간: 2026-06-24~2026-06-30
- 시각: 12:00, 12:30, 13:00, 13:30, 14:00 KST
- 채널: IR087, IR096, IR105, IR112, IR123, SW038, WV069, WV073
- 예상 파일: 7일 × 5시각 × 8채널 = 280개
- 예상 제출 행: 7일 × 96지점 = 672행
- 제출 열: `ID,TA,HM`
- ID 순서: 기존 정상 제출 파일과 동일
- TA/HM 결측 없음, HM은 0~100 범위

## 실제 리더보드 점수

2026년 정답은 숨겨져 있으므로 로컬에서 실제 리더보드 점수를 계산할 수 없다. Kaggle에 생성 CSV를 제출한 뒤 표시되는 점수를 기존 baseline 제출 점수와 비교해야 한다. 과거 검증 점수는 2026 리더보드 점수의 보장이 아니다.
