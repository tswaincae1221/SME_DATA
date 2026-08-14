# 대회 제출·실격 방지 가이드

이 문서는 최종 제출 단계에서 세 명이 같은 기준으로 검수하기 위한 운영 문서입니다.  
마감은 **2026년 8월 23일(일) 23:59 KST**이며, 아래 두 제출물을 모두 완료해야 합니다.

## 1. 반드시 제출할 두 가지

| 제출물 | 제출 방식 | 완료 기준 |
|---|---|---|
| 추론 코드 + 모델 | Kaggle 노트북 동결 + 연결 Dataset | 운영진이 날짜만 바꾸고 `Run All` 했을 때 `pred` 생성 |
| 학습 코드 | 실행 출력이 남아 있는 `.ipynb`를 구글폼 업로드 | 데이터 준비·학습·검증 결과가 셀 출력에 남아 있음 |

학습 코드 자체의 성능을 다시 채점하는 것은 아니지만, 실제 제출 모델이 어떤 과정으로 만들어졌는지 검증할 수 있어야 합니다.

## 2. GitHub와 최종 Kaggle 제출본을 분리하는 이유

공식 규정상 Kaggle 제출 노트북에는 API 키를 직접 입력해야 하지만, GitHub에 키를 올리면 안 됩니다. 따라서 다음처럼 관리합니다.

| 위치 | API 키 | 공개 상태 |
|---|---|---|
| GitHub `kaggle_cell2_template.py` | 자리표시자만 유지 | 팀 저장소만 사용, 대회 종료 전 외부 공개 금지 |
| 최종 Kaggle 노트북 | 실제 키를 문자열로 직접 입력 | 비공개 상태로 동결 |
| 모델 Dataset | API 키를 절대 포함하지 않음 | 운영진 계정에 공유하는 비공개 Dataset 우선 |

Kaggle Secrets는 사용하지 않습니다. 최종 노트북을 공개 저장소로 다시 내려받거나 복사하지 않도록 주의하고, 대회 종료 후 제출용 API 키는 폐기·재발급합니다.

## 3. 추론 노트북의 고정 구조

첫 코드 셀 최상단은 다음 형식을 유지합니다.

```python
# ╔═══════════════════════════════════════════════╗
# ║  ★ 운영진 수정 구역 ★                        ║
# ╚═══════════════════════════════════════════════╝
PRED_START = "20260624"
PRED_END = "20260630"

API_KEY = "실제_제출용_키"
```

그 아래에서만 날짜 배열을 만듭니다.

```python
pred_dates = pd.date_range(PRED_START, PRED_END, freq="D")
```

코드 다른 곳에 특정 평가 날짜, 7일, 672행을 하드코딩하지 않습니다. 행 수 검사는 다음처럼 동적으로 계산합니다.

```python
expected_rows = len(pred_dates) * len(stations)
assert len(pred) == expected_rows
```

최종 출력 변수명은 반드시 `pred`입니다.

## 4. 평가 기간 ASOS 차단 원칙

평가 기간에는 다음 행위를 모두 금지합니다.

- ASOS 또는 AWS 관측값 API 호출
- 평가 기간의 TA/HM을 읽거나 캐시에 접근
- 평가 기간 자료를 학습, 검증, 하이퍼파라미터 선택에 사용
- 평가 정답을 유추할 수 있는 다른 시간가변 외부 기상자료 사용

과거 ASOS는 학습 라벨로만 사용합니다. ASOS lag·최근값·지점별 과거 평균·평년값을 특징이나 대체값으로 모델에 저장하지 않습니다. 위성 누락에 대비한 보조 모델은 대회 공식 좌표·고도와 날짜 특징만을 입력으로 사용합니다.

보수적으로 해석해 최종 추론 Dataset에서는 ASOS 수집 코드 자체도 제외합니다. `scripts/build_inference_dataset.py`는 추론에 필요한 파일만 허용 목록으로 복사합니다.

## 5. 모델 Dataset에 들어갈 것과 빠질 것

최종 Dataset 생성 명령:

```bash
python scripts/build_inference_dataset.py \
  --model models/baseline.joblib \
  --station-list data/metadata/station_list.csv \
  --config configs/data.yaml
```

생성물:

```text
outputs/kaggle_dataset/
outputs/kaggle_dataset.zip
```

포함되는 항목:

- `baseline.joblib`
- `station_list.csv` (`STN_ID`, `LAT`, `LON`, `ALT`)
- 위성 설정만 남긴 `data.yaml`
- 학습 환경 버전을 고정한 `requirements-inference.txt`
- 최소 추론용 `gk2a_weather/` 패키지
- 파일 무결성 확인용 `manifest.sha256.json`

제외되는 항목:

- ASOS API 주소와 라벨 수집 코드
- 학습 스크립트와 교차검증 코드
- 학습 CSV, ASOS 원본, 위성 원본
- `.env`, API 키, 로그
- 테스트와 실험 파일

Dataset은 가능하면 **비공개로 유지한 뒤 운영진 계정에 직접 공유**합니다. 업로드 후 운영진이 `Copy & Edit`한 환경에서도 접근 가능한지 반드시 확인합니다. 대회 종료 전 Public Dataset에 팀 코드가 포함되지 않게 주의합니다.

## 6. 자동 점검

GitHub 템플릿은 API 키 자리표시자를 허용해 검사합니다.

```bash
python scripts/audit_submission.py \
  --notebook kaggle_cell2_template.py \
  --dataset-dir outputs/kaggle_dataset \
  --allow-placeholder
```

최종 Kaggle 노트북을 내려받은 뒤에는 자리표시자를 허용하지 않고 검사합니다.

```bash
python scripts/audit_submission.py \
  --notebook final_inference.ipynb \
  --dataset-dir outputs/kaggle_dataset
```

이 검사는 다음을 확인합니다.

- `PRED_START`, `PRED_END`, `API_KEY`가 최상위 문자열로 존재
- 날짜 배열이 설정 변수로부터 동적 생성
- 결과 변수 `pred` 생성
- Kaggle Secrets 미사용
- ASOS API·라벨 수집 코드 미포함
- 추론 중 `.fit()` 또는 학습 함수 미사용
- 모델, 좌표, 설정, 의존성, 최소 추론 코드 존재

자동 검사는 보조 수단입니다. 최종 통과 조건은 새 비공개 Kaggle 세션의 `Copy & Edit → 날짜만 수정 → Run All` 성공입니다.

## 7. 학습 `.ipynb` 제출 체크

학습 노트북에는 다음 흐름이 순서대로 보여야 합니다.

1. 사용 라이브러리와 랜덤 시드
2. 학습 데이터 경로와 기간
3. 평가 기간 자료를 제외했다는 확인
4. 특징 생성
5. 날짜 기준 `GroupKFold`
6. fold별 `RMSE_TA`, `RMSE_HM`, 종합 점수
7. 전체 학습 데이터로 최종 모델 학습
8. 모델 저장 경로와 파일 크기

제출 직전:

- `Restart & Run All` 또는 새 세션 전체 실행
- 오류 셀이 없는지 확인
- 표·로그·점수 출력이 남아 있는지 확인
- `.ipynb` 형식으로 다운로드
- 다운로드한 파일을 다시 열어 출력이 실제 저장됐는지 확인

## 8. 세 명 최종 검수 역할

| 담당 | 검수 항목 | 서명 기준 |
|---|---|---|
| 데이터 담당 | 평가 기간 ASOS 미사용, Dataset 내용, 96개 관측소 | ASOS 코드·원자료·키가 Dataset에 없음 |
| 모델 담당 | 모델/특징 버전, 학습 노트북 출력, 재현성 | 제출 모델 해시와 학습 출력 확인 |
| 팀장 | 날짜 구역, API 키, 공유 권한, Run All, 두 제출물 | 비공개 복제본에서 날짜만 바꿔 완주 |

한 사람이 최종 업로드하고, 다른 한 사람이 Kaggle 버전 시각과 구글폼 업로드 파일명을 화면 공유로 확인합니다.

## 9. 최종 체크리스트

### 추론 노트북

- [ ] `PRED_START`, `PRED_END`가 첫 코드 셀 최상단에 있음
- [ ] `API_KEY`가 실제 키 문자열로 직접 입력됨
- [ ] `kaggle_secrets`를 사용하지 않음
- [ ] 날짜·행 수가 동적으로 계산됨
- [ ] 평가 기간 ASOS/AWS를 조회하지 않음
- [ ] 추론 중 모델을 학습하거나 튜닝하지 않음
- [ ] 필요한 패키지를 코드에서 설치함
- [ ] Dataset 연결 권한을 운영진 기준으로 확인함
- [ ] 출력 변수 `pred`가 생성됨
- [ ] TA/HM에 NaN이 없고 HM이 0~100임
- [ ] 비공개 Copy & Edit 후 날짜만 바꿔 `Run All` 성공
- [ ] 마감 전 Save Version 완료

### 학습 코드

- [ ] 실제 제출 모델을 만드는 코드와 일치함
- [ ] 평가 기간 데이터가 학습·튜닝에서 제외됨
- [ ] 날짜 기준 검증 결과가 출력에 남음
- [ ] 전체 실행 출력이 저장된 `.ipynb`임
- [ ] 구글폼 업로드 완료를 다른 팀원이 확인함

### 공개·보안

- [ ] 대회 종료 전 저장소·노트북·Dataset을 외부 공개하지 않음
- [ ] API 키가 GitHub, Dataset, 출력, 로그에 없음
- [ ] 최종 Kaggle 노트북만 API 키를 포함하고 비공개임
- [ ] 복수 계정을 사용하지 않음
