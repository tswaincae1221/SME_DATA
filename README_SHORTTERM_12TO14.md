# GK-2A 12:00~14:00 short-term pipeline

## 목적
- GK-2A LE1B/KO 16채널을 12:00~14:00 KST, 10분 간격(13시각)으로 수집
- ASOS TA/HM은 같은 날짜의 14:00 정답 라벨만 수집
- 운영진 제공 station_list.csv 좌표로 각 지점의 위성 중심 픽셀값 추출
- LSTM/GRU용 long 데이터와 tabular 모델용 wide 데이터를 동시에 생성

## 파일
- `scripts/collect_shortterm_12to14.py`: 위성 + 14시 ASOS 수집, 캐시/재시작/실패기록
- `scripts/build_shortterm_12to14_dataset.py`: 지점 픽셀 추출 및 TA/HM 병합

## 결과
### long
`Date, TimeKST, STN_ID, LAT, LON, ALT, 16 channels, TA, HM`

13개 시간행이 한 날짜-지점 sequence를 이룹니다. TA/HM은 14:00 행에만 존재합니다.

### wide
한 날짜-지점이 한 행이며 `IR087_1200 ... WV073_1400`처럼 16채널×13시각을 펼친 뒤 `LAT/LON/ALT/TA/HM`을 붙입니다.

## 규정 안전장치
ASOS 12:00~13:50 온습도는 모델 입력으로 만들지 않습니다. 14:00 TA/HM만 학습 라벨로 사용합니다.

## 권장 테스트
처음에는 2024-08-24~2024-08-30처럼 짧은 범위로 수집 및 모델 효과를 확인한 뒤 전체 2019~2025로 확장하는 것을 권장합니다.

## 실행 예시
```bash
python scripts/collect_shortterm_12to14.py \
  --start 2024-08-24 --end 2024-08-30 \
  --output-root /content/drive/MyDrive/SME_DATA/processed_station_features/shortterm_12to14_data \
  --station-list data/metadata/station_list.csv

python scripts/build_shortterm_12to14_dataset.py \
  --start 2024-08-24 --end 2024-08-30 \
  --input-root /content/drive/MyDrive/SME_DATA/processed_station_features/shortterm_12to14_data \
  --station-list data/metadata/station_list.csv
```
