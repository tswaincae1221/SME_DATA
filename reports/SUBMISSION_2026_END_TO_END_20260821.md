# 2026 제출 파이프라인 검증 보고서

## 결론

- 첨부 테스트셋은 2026-06-01~2026-07-31, 96지점, 총 5,856행의 14:00 KST GK-2A 16채널 입력이다.
- 실제 평가 구간 2026-06-24~2026-06-30은 7일 × 96지점 = 672행이다.
- TA/HM 정답은 포함되지 않으므로 실제 리더보드 점수는 제출 전 계산할 수 없다.
- 동일 기간을 잠근 2025 검증 점수는 3.71850, 2022~2025 rolling OOF 대용점수는 3.83594였다. 현재 근거에 따른 2026 기대 범위는 약 3.6~4.0이며 보장값이 아니다.

## 왜 6월에는 LSTM을 끄는가

12:00~14:00 10분 간격 LSTM의 과거 학습·검증 시퀀스는 8월 24~30일에만 존재한다. 이 모델과 8월 OOF 온도 보정값을 6월에 그대로 적용하면 2025-06-24~30 대용점수가 3.57124에서 4.57695로 악화됐다. 따라서 날짜를 코드에 하드코딩하는 대신 모델 번들에 `sequence_supported_mmdd=[824,830]`을 기록했다.

- 지원 구간: 13시점 LSTM + 14시 CatBoost + 잔차 Ridge/공간 CatBoost
- 그 외 구간(현재 6월 평가): 14시 CatBoost + 직접 Ridge/공간 CatBoost

이 분기는 평가 날짜를 미리 박아 넣는 것이 아니라 모델이 실제 검증된 입력 지원 범위를 적용하는 모델 계약이다.

## 6월 제출 모델

각 분기는 seed 42, 43, 44로 학습하고 마지막에 산술평균한다.

### TA

`0.78 × CatBoost_TA + 0.22 × Ridge_TA + 0.542146`

- CatBoost와 Ridge는 2019~2025 전체 6~8월 14시 라벨로 최종 재학습한다.
- 비중과 오프셋은 2022~2025년 6월 24~30일 rolling OOF에서 동결했다.
- 직접 연도 변수, ASOS lag, 기후평년값은 사용하지 않는다.

### HM

`0.36 × base_CatBoost_HM + 0.64 × spatial_CatBoost_HM`

- spatial CatBoost는 동일 날짜·14시의 96지점 LE1B 값과 공식 위경도·고도로 계산한 최근접 3지점 평균/편차를 쓴다.
- 관측 TA는 HM 입력으로 사용하지 않는다.
- 두 CatBoost의 비중은 2022~2025년 6월 rolling OOF에서 동결했다.

## 규정 준수

- 위성 API 경로: `GK2A/LE1B/{channel}/KO/data`만 호출
- 14:00 KST는 `to_api_datetime()`으로 05:00 UTC로 변환
- 6월 제출 시 날짜당 16개 파일, 7일 총 112개 LE1B 파일만 요청
- ASOS TA/HM은 학습 라벨로만 사용하고 제출 추론에서는 읽지 않음
- 공식 `station_list.csv`의 96개 STN_ID/LAT/LON/ALT 사용
- 제출 노트북에 `.fit()` 호출이나 파인튜닝 없음
- 운영진 설정의 `PRED_DATES`만 사용하며 자유 구현 셀에 평가 날짜 하드코딩 없음

## 2026 테스트셋 예측 리허설

| 항목 | 결과 |
|---|---:|
| 행 수 | 672 |
| 고유 ID | 672 |
| TA/HM 결측 | 0 |
| TA 최소 / 평균 / 최대 | 20.71 / 27.94 / 32.22 °C |
| HM 최소 / 평균 / 최대 | 35.12 / 62.88 / 98.43 % |

생성 파일은 `ID,TA,HM` 순서이고, ID는 `YYYYMMDD_STN_ID`, TA/HM은 소수 둘째 자리로 반올림된다.

## 제출 순서

1. `Colab_Build_Submission_2026_Bundle.ipynb`를 실행해 Drive에 모델 번들과 ZIP을 생성한다.
2. ZIP 내용을 Kaggle Dataset으로 업로드하고 제출 노트북의 Add Input에 연결한다.
3. 공식 `station_list.csv` 데이터셋도 연결하고 Internet을 On으로 설정한다.
4. `submission_2026_end_to_end.ipynb`의 설정 셀에 API_KEY를 직접 입력한다.
5. Save Version → Save & Run All로 실행한다.
6. `/kaggle/working/submission.csv`, `api_audit_log.csv`, `gk2a_failures.csv`를 확인한다.

## 주의

`submission_2026_preview_0624_0630.csv`는 첨부된 2026 테스트셋으로 만든 형식·분포 확인용 파일이다. 실제 Kaggle 제출 노트북은 운영진의 `PRED_DATES`와 KMA API 입력으로 같은 추론 코드를 다시 실행한다.
