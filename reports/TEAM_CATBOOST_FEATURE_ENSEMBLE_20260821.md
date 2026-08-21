# Team CatBoost 피처·앙상블 개선 실험

## 결론

팀원 파일에서 CatBoost 하이퍼파라미터는 가져오지 않고, 규정에 맞는 피처 아이디어와
앙상블 구조만 재검증했다. 최종 권장안은 TA를 변경하지 않고, HM에만 기존 40개 피처와
14개 물리·공간 피처를 사용하는 CatBoost 보조 분기를 12~18% 결합하는 구조다.

시드 42~44의 고정 피처 확인 결과, 2025년 8월 24~30일에서 다중 시드 평균 성능은 다음과
같았다.

| 구성 | TA RMSE | HM RMSE | `TA + 0.1 × HM` |
|---|---:|---:|---:|
| 기존 다중 시드 평균 | 1.668450 | 7.208087 | 2.389258 |
| 고정 피처 HM CatBoost 결합 | 1.668450 | **7.147951** | **2.383245** |
| 변화 | 0.000000 | **-0.060137** | **-0.006014** |

이 값은 2025년 과거 라벨을 이용한 로컬 확인 점수이며 대회의 공식 점수가 아니다. 피처와
가중치는 2022~2024 rolling OOF에서 정하고, 2025 라벨은 마지막 보고에만 사용했다.

## 팀원 코드 감사

### 채택·수정한 아이디어

| 아이디어 | 반영 방식 |
|---|---|
| 채널 물리차 | 14시 LE1B 채널만으로 6개 raw difference 생성 |
| 전 지점 anomaly | 같은 날짜·시각의 관측소 평균에서 각 지점 값을 뺀 동시간 공간 편차 |
| 최근접 3지점 | 공식 `station_list.csv` 위경도로 haversine 최근접 지점을 계산하고 IR105/WV069 평균·편차 생성 |
| NDVI proxy | 분모 폭주를 막는 bounded normalized difference로 수정 후 후보 실험 |
| TA→HM 체인 | 실제 TA 대신 rolling OOF/frozen TA 예측만 쓰도록 수정 후 후보 실험 |
| 모델 혼합 | TA/HM을 따로, 2022~2024 rolling OOF로 비음수 가중치 선택 |

### 제외한 부분

| 원 코드 | 제외 이유 |
|---|---|
| `Year` | 앞선 연도의존 실험에서 분포 이동에 불안정했고 직접 연도항은 최종 피처 정책에서 제외 |
| `Hour` | 마스터의 `Time=500 UTC`, 즉 14 KST로 상수 |
| `Lag1`, `Roll3`, `Delta1` | 과거 스냅샷 의존성을 다시 만들며 첫 행을 미래 행으로 `bfill`하는 누수 위험 |
| 검증 구간 `bfill` | 미래 날짜의 값이 앞 날짜 입력에 들어감 |
| 검증 구간 전체 최솟값 로그변환 | 검증 전체 분포를 변환에 사용하고 train/test 변환이 달라짐 |
| 실제/인샘플 TA를 HM 힌트로 사용 | 학습·추론 분포가 다르고 ASOS 입력 금지 취지에 맞지 않음 |
| 한 검증 구간에서 3모델 가중치 탐색 | 같은 구간으로 모델·TA/HM 가중치를 함께 고르면 과적합 위험 |

첨부 노트북의 `Ratio_SW_VI`는 HM에서 스파이크를 이유로 직접 제거되어 있었다. 실험에서도
bounded visible 계열은 고정 최종안에 포함되지 않았다. 예측 TA 힌트와 지점 범주형도 시드
간 채택이 불안정해 최종 고정 피처에서 제외했다.

## 최종 고정 HM 피처

기존 40개 피처에 아래 14개만 더했다.

### 물리 채널차 6개

- `team_diff_ir105_ir087`
- `team_diff_ir105_ir123`
- `team_diff_wv063_wv073`
- `team_diff_ir105_wv069`
- `team_diff_ir096_wv063`
- `team_diff_sw038_ir096`

### 동시간 공간 피처 8개

- `team_daily_anomaly_ir105`
- `team_daily_anomaly_wv069`
- `team_daily_anomaly_team_diff_ir096_wv063`
- `team_daily_anomaly_team_ndvi_safe`
- `team_nearest3_mean_ir105`
- `team_nearest3_delta_ir105`
- `team_nearest3_mean_wv069`
- `team_nearest3_delta_wv069`

`team_diff_ir105_ir123`과 기존 `window_105_123`은 관련되지만 완전히 같은 값은 아니다. 전자는
raw DN 차이이고 후자는 학습기간 채널별 평균·표준편차를 각각 적용한 뒤 뺀 값이다. 고정
다중 시드 OOF에서 물리차 묶음이 공간 묶음과 함께 일관된 보조 신호를 보여 유지했다.

## 고정 피처 다중 시드 결과

| seed | 새 HM 분기 비중 | 현재 HM OOF | 결합 HM OOF | 현재 2025 HM | 결합 2025 HM |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.12 | 8.958380 | 8.930809 | 7.441432 | **7.363599** |
| 43 | 0.14 | 8.992773 | 8.956601 | 7.262872 | **7.170049** |
| 44 | 0.18 | 9.141926 | 9.091459 | 7.208921 | **7.120372** |

모든 시드가 pooled OOF에서 0.02 이상 개선됐고, 최신 2024 OOF도 각각
`7.7101→7.5471`, `7.6436→7.4435`, `7.7487→7.5450`으로 좋아졌다. 최종 예측은 각
시드의 OOF 선택 가중치를 적용한 뒤 세 시드 예측을 평균한다.

## 영향도가 높은 피처

고정 HM CatBoost 3개 시드의 `PredictionValuesChange` 평균 상위 피처는 다음과 같다. 이
중 새 피처는 굵게 표시했다.

| 순위 | 피처 | 평균 중요도 |
|---:|---|---:|
| 1 | **`team_diff_ir105_ir123`** | 15.0067 |
| 2 | `LAT` | 9.7742 |
| 3 | `LON` | 8.9110 |
| 4 | `doy_sin` | 6.3638 |
| 5 | `NR016` | 5.9323 |
| 6 | `co2_window_133_112` | 5.6469 |
| 7 | **`team_nearest3_mean_ir105`** | 5.4980 |
| 8 | `dayofyear` | 5.0663 |
| 9 | `solar_declination` | 3.6148 |
| 10 | `equation_of_time` | 2.5709 |
| 15 | **`team_nearest3_delta_ir105`** | 1.7223 |
| 20 | **`team_daily_anomaly_team_diff_ir096_wv063`** | 1.2440 |

중요도는 인과효과가 아니며 상관 피처 사이에서 분산될 수 있다. 다만 raw IR105–IR123 관계와
IR105의 주변 지점 수준·편차가 세 시드 모두에서 상위권인 점은 공간적인 구름·수증기 상태가
HM 잔차 보정에 유효하다는 해석과 일치한다.

## 실행 파일

- `scripts/experiment_team_catboost_feature_ensemble.py`: 팀원 아이디어를 그룹별로 추가하고 TA/HM 독립 rolling OOF 선택
- `scripts/confirm_team_catboost_feature_ensemble_multiseed.py`: 최종 54개 HM 피처를 고정한 시드 42~44 확인과 평균
- `tests/test_team_catboost_feature_ensemble.py`: 공식 좌표, 동시간 공간 범위, 금지 피처, 고정 피처 계약 테스트
- `notebooks/Colab_Team_CatBoost_Feature_Ensemble.ipynb`: Google Drive/Colab 실행 노트북

CatBoost의 iterations, depth, learning rate, L2 등은 기존 저장소 설정을 그대로 사용했고 이번
실험에서는 탐색하지 않았다.
