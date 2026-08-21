# 6월 LSTM 유효성 검증 가이드

## 목적

리더보드 3점대가 나온 현재 14시 모델에 LSTM을 추가했을 때 실제로 일반화 성능이 개선되는지 확인한다. LSTM을 무조건 추가하지 않고, 시간순 OOF에서 정해 둔 최소 개선폭과 연도 안정성을 통과한 경우에만 채택한다.

## 1차 실험 범위

| 항목 | 설정 |
|---|---|
| 날짜 | 2020~2025년, 매년 6월 24~30일 |
| 시각(KST) | 12:00, 12:30, 13:00, 13:30, 14:00 |
| 채널 | IR087, IR096, IR105, IR112, IR123, SW038, WV069, WV073 |
| 지점 | 대회 공식 station_list.csv의 96개 지점 |
| 라벨 | 각 날짜·지점의 14:00 ASOS TA/HM |
| 선택 구간 | 2022, 2023, 2024 rolling OOF |
| 잠금 검증 | 2025 |

2019년은 현재 master CSV에 6월 라벨이 없어 이번 실험에서 제외한다. 2020~2021은 최초 LSTM 학습·내부 검증에 사용하고, 2022년부터 시간순 예측을 만든다.

## 데이터 수집량

- 전체 논리 파일 수: `6년 × 7일 × 5시각 × 8채널 = 1,680`
- 기존 master의 14시 채널 재사용: `6 × 7 × 1 × 8 = 336`
- 최대 신규 API 요청: `1,344`

각 NC는 다운로드 직후 공식 96지점 값만 작은 CSV 캐시에 저장한다. 추출에 성공한 NC는 기본적으로 삭제해 Drive 용량을 절약한다. 중단·429·런타임 종료 후 같은 수집 셀을 다시 실행하면 유효한 캐시는 건너뛰고 누락된 파일만 요청한다.

## Colab 실행 순서

1. `notebooks/Colab_June_LSTM_Feasibility.ipynb`를 Colab에서 연다.
2. 런타임 유형을 GPU로 설정한다. T4면 충분하다.
3. 1~3번 셀을 실행해 Drive, GitHub 코드, master 경로를 확인한다.
4. 4번 셀에서 기상청 API 키를 비공개 입력한다.
5. 5번 셀의 수집 완성도를 확인한다. 90% 미만이면 API 제한 해제 후 4번 셀을 재실행한다.
6. `QUICK_SMOKE=False`로 6번 셀을 실행한다.
7. 7번 셀에서 OOF 결정과 2025 잠금 검증을 확인한다.

`QUICK_SMOKE=True`는 설치·경로·GPU 동작 확인용이다. seed 1개, 축소 epoch/iteration을 사용하므로 모델 채택 판단에는 사용할 수 없다.

## 비교 모델

| 이름 | 구조 | 역할 |
|---|---|---|
| baseline | 14시 TA CatBoost+Ridge, HM 기본+공간 CatBoost | 현재 제출 모델의 14시 구조 |
| direct | 5시각 위성 시퀀스로 TA/HM 직접 예측 | LSTM 단독 정보량 확인 |
| direct_blend | baseline과 direct LSTM의 볼록 결합 | 서로 다른 오차 상쇄 확인 |
| residual_blend | baseline + scale × LSTM 예측 잔차 | 14시 모델이 놓친 시간 변화만 교정 |

LSTM 입력은 `5시각 × (8채널 값 + 8결측 마스크)`와 공식 좌표·고도·계절/태양 위치·결측률이다. 직접적인 `year` 변수와 ASOS lag는 사용하지 않는다.

## 누수 방지

예측 연도 `Y`에 대해 다음 순서를 사용한다.

1. 14시 baseline은 `Y-1`년까지만 학습해 `Y`년을 예측한다.
2. LSTM epoch는 `Y-2`년까지 학습하고 `Y-1`년에서 선택한다.
3. 선택된 epoch 수만 사용해 `Y-1`년까지 재학습하고 `Y`년을 예측한다.
4. blend weight와 residual scale은 2022~2024 OOF에서만 한 번 선택한다.
5. 2025 결과를 보고 가중치·feature·epoch를 다시 조정하지 않는다.

잔차 LSTM의 학습 라벨도 각 연도의 시간순 baseline OOF 오차를 사용한다. 동일 행을 학습한 baseline의 in-sample 오차를 사용하지 않는다.

## 자동 채택 기준

- TA OOF RMSE 개선이 `0.03` 이상
- HM OOF RMSE 개선이 `0.10` 이상
- 2022~2024 중 최소 2개 연도에서 baseline보다 악화되지 않음
- 위 조건을 통과하지 못하면 해당 target은 LSTM 가중치를 사실상 0으로 두고 baseline을 유지

`experiment_summary.json`의 `OOF_decision.add_LSTM_to_TA`와 `add_LSTM_to_HM`가 최종 자동 판정이다. 2025 잠금 결과는 이 결정이 새로운 연도에서도 유지되는지 확인하는 진단값이다.

## 결과 파일

Drive 경로:

`SME_DATA/processed_station_features/june_lstm_feasibility/`

주요 파일:

- `sequence_data/june_shortterm_core8_2020to2025.csv`: 완전한 날짜·시각·지점 격자
- `sequence_data/collection_summary.json`: 수집 완성도
- `sequence_data/missing_inventory.csv`: 재수집 대상 목록
- `experiment_results/model_comparison.csv`: 후보별 OOF/2025 성능
- `experiment_results/experiment_summary.json`: 가중치, 개선폭, 자동 판정
- `experiment_results/all_candidate_predictions.csv`: 행 단위 실제값·예측값
- `experiment_results/baseline_feature_importance.csv`: 14시 CatBoost 중요도
- `experiment_results/lstm_epoch_selection.csv`: seed/target/연도별 epoch

## 오류별 처리

### HTTP 429 또는 API 제한

완료 캐시는 보존된다. 제한이 풀린 뒤 수집 셀을 그대로 다시 실행한다. 폴더를 삭제하거나 처음부터 받을 필요가 없다.

### Colab 런타임 종료

Drive 캐시가 남아 있으므로 1~4번 셀을 다시 실행한다. `collection_summary.json`과 `missing_inventory.csv`가 자동 갱신된다.

### 수집 완성도 90% 미만

모델 셀을 억지로 실행하지 않는다. 일부 결측은 마스크로 처리하지만, 시퀀스 대부분이 비면 LSTM 유효성 자체를 판정할 수 없다.

### baseline 재학습을 다시 하고 싶은 경우

`experiment_results/baseline_chronological_predictions.csv`를 보존하면 다음 실행에서 재사용한다. master나 feature 코드를 바꿨다면 실험 명령에 `--force-baseline`을 추가한다.

## 대회 규정 준수

- 위성 API 경로는 `/GK2A/LE1B/{channel}/KO/data`만 사용한다.
- ASOS TA/HM은 과거 학습 라벨로만 사용한다.
- ASOS lag, 최근 관측값, 평가기간 ASOS 조회를 사용하지 않는다.
- 위도·경도·고도는 저장소의 대회 공식 station_list.csv만 사용한다.
- 외부 DEM, 토지피복, 재분석, 수치예보, 기후 평년값을 사용하지 않는다.

