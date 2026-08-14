# 처음 실행하는 팀원을 위한 사용 설명서

이 문서는 세 명 모두 공모전·GitHub 협업이 처음이라는 전제로 작성했습니다. 첫 목표는 최고 점수가 아니라 **세 사람의 컴퓨터에서 같은 명령이 실행되는 것**입니다.

## 0. 지금 코드에서 되는 것과 아직 확인할 것

현재 구현된 기능:

- ASOS 14시 TA/HM 수집, 파싱, 결측 처리
- API 실패 재시도와 날짜별 캐시
- GK-2A 16채널 다운로드와 오류 파일 검사
- 검증된 픽셀 좌표가 있을 때 중심·주변 패치 특징 추출
- ASOS·관측소·위성 특징 결합
- 날짜 기준 GroupKFold 학습과 대회 점수 계산
- 위성 전체 결측 시 관측소×월 평균 fallback
- `pred` 및 `submission.csv` 생성·검증
- 공식 날짜 설정 구역을 포함한 Kaggle 추론 노트북 템플릿
- ASOS·학습 코드를 제외한 최소 추론 Dataset 생성기
- 제출 규정 자동 감사 스크립트
- API 없이 연습할 수 있는 합성 데이터

대회 제공 파일을 받아야 확정할 부분:

1. GK-2A KO 파일의 정확한 변수명·채널별 보정 방식
2. 위경도→채널별 픽셀 좌표 변환
3. 공식 Kaggle 환경에서 최소 추론 Dataset과 노트북의 최종 결합

GK-2A 파일명과 API 요청 시각은 UTC 기준으로 확인했습니다. 따라서 매일
14:00 KST 자료는 API에 05:00 UTC로 요청하며, `configs/data.yaml`의
`api_time_basis`는 `utc`로 고정되어 있습니다.

## 1. 저장소를 받은 직후

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

### macOS·Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

`.env` 안의 `KMA_API_KEY`에는 노출된 기존 키가 아니라 재발급한 키를 넣습니다. `.env`는 `.gitignore`에 포함되어 있으므로 커밋하지 않습니다.

## 2. API 없이 10분 연습

세 명 모두 아래 명령을 먼저 실행합니다.

```bash
python scripts/create_demo_data.py
python scripts/train_baseline.py \
  --dataset data/processed/demo_train.csv \
  --output models/demo.joblib \
  --experiment-id DEMO
python scripts/make_submission.py \
  --features data/processed/demo_inference_features.csv \
  --model models/demo.joblib \
  --sample-submission data/metadata/demo_sample_submission.csv \
  --output outputs/submissions/demo_submission.csv
```

Windows PowerShell에서는 줄 끝의 `\` 대신 한 줄로 입력하거나 백틱을 사용합니다.

성공 기준:

- `models/demo.joblib` 생성
- 콘솔에 fold별 점수 출력
- `outputs/submissions/demo_submission.csv` 생성
- TA/HM에 빈값이 없음

이 과정이 한 명에게만 성공하면 실제 데이터를 받기 전에 환경 차이부터 해결합니다.

## 3. 대회 제공 파일 넣기

아래 두 파일을 GitHub에 포함해도 되는지는 대회 규칙을 확인하고, 우선 로컬의 다음 위치에 둡니다.

```text
data/metadata/station_list.csv
data/metadata/sample_submission.csv
```

그다음 검사합니다.

```bash
python scripts/check_setup.py
```

관측소가 96개인지, 제출 양식이 존재하는지 확인합니다.

## 4. ASOS 한 날짜 시험

먼저 대량 수집을 하지 말고 한 날짜만 확인합니다.

```bash
python scripts/collect_labels.py --start 2025-08-24 --end 2025-08-24
```

확인할 파일:

```text
data/raw/asos/asos_202508241400.txt
data/interim/asos/asos_202508241400.csv
```

정제 CSV에서 확인할 것:

- `date`가 요청한 날짜인가
- `timestamp_kst`가 14:00인가
- `STN_ID`가 대회 관측소만 남았는가
- TA/HM가 현실적인 범위인가
- `-99`, `-9`가 빈값으로 바뀌었는가

## 5. 위성 한 채널 시험

`configs/data.yaml`의 시간 기준이 다음과 같은지 확인합니다.

```yaml
satellite:
  api_time_basis: utc  # 14:00 KST = 05:00 UTC
```

한 날짜·한 채널만 받습니다.

```bash
python scripts/collect_satellite.py \
  --start 2025-08-24 \
  --end 2025-08-24 \
  --channels IR105
```

저장된 파일명에 `0500`이 들어가면 14:00 KST 자료를 올바르게 요청한 것입니다.

## 6. 공식 좌표를 위성 픽셀로 변환하기

`data/metadata/station_list.csv`에는 운영진이 제공한 96개 지점의
`STN_ID,LAT,LON,ALT`만 두어야 합니다. 외부 관측소 목록이나 지형·토지피복·
해안선 거리를 결합하지 마세요.

공식 위경도를 채널별 LE1B KO 격자로 투영해 96개 지점이 모두
범위 안인지 확인합니다.

```bash
python scripts/build_official_station_pixels.py
```

검증용 `station_pixels_official_16ch.csv`는 공식 좌표에서 재생성할 수 있는
캐시입니다. 실제 특징 추출기는 공식 `station_list.csv`를 직접 읽어
원본 영상 shape에 맞는 row/col을 계산합니다. 서울·부산·제주 등
표본 지점의 위치도 영상 위에서 눈으로 교차 확인하세요.

```bash
python scripts/extract_satellite_features.py \
  --start 2025-08-24 \
  --end 2025-08-24
```

## 7. 실제 학습

```bash
python scripts/build_dataset.py
python scripts/train_baseline.py \
  --output models/baseline.joblib \
  --experiment-id B1
```

위성 특징이 아직 없으면 `build_dataset.py`가 자동으로 B1 정적 기준 데이터만 만듭니다. 이후 위성 특징을 만든 뒤 다시 실행하면 B2/B3 실험으로 확장할 수 있습니다.

점수는 다음 두 곳에 남습니다.

```text
models/baseline.metrics.json
experiments/experiment_log.csv
```

## 8. 실제 수집 범위 확대

한 날짜의 시간·좌표·값 범위를 세 명 중 두 명이 확인한 뒤에만 범위를 늘립니다.

```bash
python scripts/collect_labels.py --start 2019-07-01 --end 2025-09-30
python scripts/collect_satellite.py --start 2019-07-01 --end 2025-09-30
```

날짜별 캐시가 있으므로 중간에 중단돼도 같은 명령을 다시 실행하면 완료된 파일은 재사용합니다. 오류 목록은 `outputs/asos_failures.csv`, `outputs/gk2a_failures.csv`에 저장됩니다.

## 9. Kaggle 최종 추론 제출

자세한 최종 검수는 [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)를 따릅니다. 기본 템플릿은 `notebooks/kaggle_inference_template.ipynb`이며, 코드만 확인하려면 `kaggle_cell2_template.py`를 봅니다.

첫 코드 셀 최상단은 아래 구조를 유지합니다.

```python
PRED_START = "20260624"
PRED_END = "20260630"
API_KEY = "실제_제출용_키"
```

- 운영진은 `PRED_START`, `PRED_END`만 변경합니다.
- Kaggle Secrets는 사용하지 않습니다.
- 실제 키는 최종 비공개 Kaggle 노트북에만 직접 입력합니다.
- 결과 변수명은 반드시 `pred`입니다.
- 평가 기간 ASOS/AWS를 조회하거나 모델을 다시 학습하지 않습니다.
- 행 수는 `날짜 수 × 관측소 수`로 동적으로 계산합니다.

최종 모델 학습 후 최소 추론 Dataset을 만듭니다.

```bash
python scripts/build_inference_dataset.py \
  --model models/baseline.joblib \
  --station-list data/metadata/station_list.csv \
  --config configs/data.yaml
```

Kaggle Dataset에는 다음만 포함됩니다.

```text
gk2a_weather/                 # src 아래 패키지
baseline.joblib
station_list.csv
data.yaml
requirements-inference.txt
manifest.sha256.json
```

ASOS 수집기, 학습 코드, 원자료, API 키는 자동으로 제외됩니다. Dataset은 운영진 계정에 공유하는 비공개 상태를 우선 사용하고, `Copy & Edit`된 운영진 환경에서도 접근 가능한지 확인합니다.

GitHub 템플릿 점검:

```bash
python scripts/audit_submission.py \
  --notebook kaggle_cell2_template.py \
  --dataset-dir outputs/kaggle_dataset \
  --allow-placeholder
```

최종 Kaggle `.ipynb`를 다운로드한 뒤에는 `--allow-placeholder` 없이 다시 검사합니다. 그다음 비공개 `Copy & Edit`에서 날짜만 바꾸고 `Run All` 해야 합니다.

학습 코드는 별도 제출물입니다. 실제 실행 출력이 남아 있는 `.ipynb`를 다운로드해 구글폼에 업로드해야 최종 제출이 완료됩니다.

## 10. 세 명의 첫 역할

| 담당 | 오늘 완료할 일 | 다른 사람이 확인할 것 |
|---|---|---|
| A 데이터 | ASOS 한 날짜 수집과 값 범위 확인 | B가 같은 명령 재실행 |
| B 위성 | IR105 한 장과 픽셀 좌표 3곳 검증 | C가 위치 그림 확인 |
| C 모델 | 데모 학습·제출과 실험표 확인 | A가 새 가상환경에서 재현 |

세 명 모두 데모 파이프라인은 직접 한 번씩 실행합니다. 이후에만 담당을 나눕니다.

## 11. 자주 생기는 오류

| 오류 | 의미 | 해결 |
|---|---|---|
| `KMA_API_KEY가 없습니다` | `.env`가 없거나 키가 비어 있음 | `.env.example`을 `.env`로 복사 후 새 키 입력 |
| 위성 요청 시각이 `1400`으로 저장됨 | KST를 UTC로 변환하지 않음 | `api_time_basis: utc`로 유지하고 `0500` 요청 확인 |
| `station_list.csv가 없습니다` | 캐글 제공 파일 미배치 | `data/metadata/`에 넣기 |
| `공식 station_list.csv 좌표가 필요` | `LAT/LON/ALT` 누락 | 운영진 제공 4열 파일을 `data/metadata/` 아래에 배치 |
| 위성 파일이 너무 작음 | 오류 페이지·없는 시각일 가능성 | API 승인과 요청 시각 확인 |
| CV 점수가 비정상적으로 좋음 | 행 랜덤 분할 누수 가능성 | 현재 `GroupKFold(date)` 유지 |
| 제출 ID 누락 | 날짜×관측소 조합 불완전 | `pred` 행 구성과 sample ID 비교 |
