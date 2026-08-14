# Colab에서 Google Drive NC 전처리

[`notebooks/Colab_Drive_NC_Preprocessing.ipynb`](notebooks/Colab_Drive_NC_Preprocessing.ipynb)을 Colab에서 열고 위에서부터 실행합니다.

## 처음에 바꿀 값

노트북의 **3. 경로와 기간 설정** 셀에서 아래 두 경로만 본인의 Drive 구조에 맞게 바꿉니다.

```python
NC_INPUT_DIR = "/content/drive/MyDrive/SME_DATA/nc"
RESULT_DIR = "/content/drive/MyDrive/SME_DATA/processed_station_features"
```

- `NC_INPUT_DIR`: 업로드한 `.nc` 파일이 들어 있는 최상위 폴더
- `RESULT_DIR`: 전처리 결과를 저장할 폴더. 없어도 자동 생성
- 입력 폴더 아래의 연도/월/일 하위 폴더는 재귀적으로 자동 검색
- KMA 파일명 시각은 기본적으로 UTC로 해석. `0500 UTC`가 `1400 KST`로 변환됨
- 기본 기간은 2019-06-01~2025-08-31, 대상 월은 6·7·8월

## 실행 순서

1. Drive 마운트
2. `SME_DATA` 최신 코드 설치
3. 경로와 기간 설정
4. 스캔만 실행
5. 실제 전처리 실행
6. 결과 검수

스캔 셀은 NC 내용을 열지 않고 파일명만 확인합니다. `16채널 완성 시각` 개수를 확인한 후 실제 실행 셀을 실행합니다.

## 저장되는 결과

```text
RESULT_DIR/
├── daily/
│   ├── 2019/features_20190601_1400KST.parquet
│   └── ...
├── gk2a_station_features_20190601_20250831_1400KST.parquet
├── preprocessing_manifest.csv
├── preprocessing_failures.csv
└── preprocessing_summary.json
```

- 날짜별 Parquet: 하루 96개 관측소의 체크포인트
- 전체 Parquet: 날짜별 결과를 합친 모델 입력용 테이블
- manifest: 날짜별 처리 상태와 채널 수
- failures: 16채널 미완성 또는 처리 오류 날짜
- summary: 전체 행·열·날짜·관측소 개수

Drive 용량을 아끼기 위해 기본 결과는 Parquet입니다. `WRITE_CSV = True`로 바꾸면 전체 결과의 압축 CSV(`.csv.gz`)도 생성합니다.

## 중단 후 다시 실행

날짜별 결과를 먼저 저장하므로 Colab 연결이 끊겨도 같은 셀을 다시 실행하면 완료된 날짜는 건너뜁니다. 특정 날짜를 다시 만들려면 명령에 `--overwrite`를 추가하거나 해당 날짜별 Parquet만 직접 확인한 후 제거합니다.

## 규정 반영

- 위성 입력: GK-2A LE1B/KO 16채널만 가정
- 관측소 정보: 저장소의 대회 공식 `station_list.csv` 네 열만 사용
- 날짜 파생변수: 연·월·일, 연중일, sin/cos만 생성
- 외부 DEM·토지피복·해안선 거리·평년값·ASOS lag 사용 안 함
- API 키 사용 안 함: 이미 Drive에 업로드한 NC만 읽음

파일이 FD이거나 KO 격자 크기와 다르면 좌표 검증 단계에서 실패로 기록됩니다. 기본 설정은 16채널이 모두 업로드된 날짜만 처리하며, 미완성 날짜는 다음 실행에서 다시 확인합니다.

