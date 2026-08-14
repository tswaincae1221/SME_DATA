"""2019~2025년 6~8월 매일 14:00 KST ASOS TA/HM 라벨 수집.

운영진 제공 station_list.csv의 STN_ID만 ASOS 라벨 필터에 사용한다.
공식 LAT/LON/ALT는 별도 정적 입력으로 사용할 수 있지만 ASOS 라벨 CSV에는
중복 저장하지 않는다. 인증키는 환경변수 또는 별도 파일에서 읽고
로그·CSV·패키지에 기록하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
MISSING_TA = -99.0
MISSING_HM = -9.0


@dataclass
class DayResult:
    date: str
    timestamp_kst: str
    status: str
    source: str
    api_rows: int
    target_rows: int
    ta_valid: int
    hm_valid: int
    missing_station_count: int
    missing_station_ids: str
    message: str = ""


class ApiResponseError(RuntimeError):
    """기상청 API가 데이터 대신 오류 문서를 반환했을 때 발생한다."""


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_api_key(api_key_file: Path | None) -> str:
    api_key = os.getenv("KMA_API_KEY", "").strip()
    if not api_key and api_key_file is not None and api_key_file.exists():
        api_key = api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(
            "KMA_API_KEY 환경변수 또는 --api-key-file로 인증키를 제공해야 합니다."
        )
    return api_key


def load_station_ids(path: Path) -> list[int]:
    frame = pd.read_csv(path)
    if "STN_ID" not in frame.columns:
        raise ValueError(f"{path}에 STN_ID 열이 없습니다.")
    station_ids = pd.to_numeric(frame["STN_ID"], errors="raise").astype(int)
    if station_ids.duplicated().any():
        raise ValueError("station_list.csv에 중복 STN_ID가 있습니다.")
    if len(station_ids) != 96:
        raise ValueError(f"대회 관측소는 96개여야 합니다. 현재 {len(station_ids)}개")
    return station_ids.tolist()


def target_dates(start_year: int, end_year: int, months: list[int]) -> pd.DatetimeIndex:
    dates: list[pd.Timestamp] = []
    for year in range(start_year, end_year + 1):
        for month in months:
            start = pd.Timestamp(year=year, month=month, day=1)
            end = start + pd.offsets.MonthEnd(0)
            dates.extend(pd.date_range(start, end, freq="D"))
    return pd.DatetimeIndex(dates)


def parse_response(text: str) -> pd.DataFrame:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(text)
            result = payload.get("result", {})
            status = result.get("status", "unknown")
            message = result.get("message", "알 수 없는 API 오류")
        except json.JSONDecodeError:
            status = "unknown"
            message = "JSON 형식의 오류 응답"
        raise ApiResponseError(f"API status={status}: {message}")

    if not text.rstrip().endswith("#7777END"):
        raise ApiResponseError(
            "응답 종료표시 #7777END가 없어 전송 중 잘린 원문으로 판단했습니다."
        )

    records: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 14 or not fields[0].isdigit():
            continue
        try:
            timestamp = pd.to_datetime(fields[0], format="%Y%m%d%H%M")
            station_id = int(fields[1])
            ta = float(fields[11])
            hm = float(fields[13])
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "date": timestamp.strftime("%Y-%m-%d"),
                "timestamp_kst": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "STN_ID": station_id,
                "TA": pd.NA if ta == MISSING_TA else ta,
                "HM": pd.NA if hm == MISSING_HM else hm,
            }
        )

    if not records:
        raise ApiResponseError("응답에서 ASOS 데이터 행을 찾지 못했습니다.")

    frame = pd.DataFrame.from_records(records)
    frame["TA"] = pd.to_numeric(frame["TA"], errors="coerce")
    frame["HM"] = pd.to_numeric(frame["HM"], errors="coerce")
    if frame.duplicated(["timestamp_kst", "STN_ID"]).any():
        raise ApiResponseError("응답에 timestamp×STN_ID 중복이 있습니다.")
    return frame.sort_values(["timestamp_kst", "STN_ID"]).reset_index(drop=True)


def fetch_text(
    timestamp: pd.Timestamp,
    api_key: str,
    timeout: float,
    max_retries: int,
) -> str:
    params = urllib.parse.urlencode({
        "tm": timestamp.strftime("%Y%m%d%H%M"),
        "stn": 0,
        "help": 0,
        "authKey": api_key,
    })
    url = f"{API_URL}?{params}"
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "asos-label-collector/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
            text = content.decode(charset, errors="replace")
            parse_response(text)
            return text
        except (urllib.error.URLError, TimeoutError, ApiResponseError) as exc:
            last_error = exc
            message = str(exc)
            if "활용신청" in message or "인증" in message or "제한" in message:
                raise ApiResponseError(message) from exc
            if attempt < max_retries:
                time.sleep(1.5 * (2 ** (attempt - 1)))
    raise ApiResponseError(f"{max_retries}회 재시도 후 실패: {last_error}")


def parallel_download_raw(
    dates: pd.DatetimeIndex,
    *,
    hour_kst: int,
    raw_dir: Path,
    api_key: str,
    parallel: int,
    batch_size: int,
    timeout: float,
    max_retries: int,
) -> None:
    """curl 병렬 모드로 원문을 먼저 캐시한다.

    인증키가 포함된 curl 설정은 표준입력으로만 전달하고 파일·로그로 남기지 않는다.
    각 응답은 파싱 검증을 통과한 뒤에만 최종 ``.txt`` 캐시로 승격한다.
    """
    if parallel <= 0 or shutil.which("curl") is None:
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[pd.Timestamp, Path]] = []
    for date_value in dates:
        day = date_value.replace(hour=hour_kst)
        raw_path = raw_dir / f"asos_{day.strftime('%Y%m%d%H%M')}.txt"
        if raw_path.exists():
            try:
                parse_response(raw_path.read_text(encoding="utf-8", errors="replace"))
                continue
            except ApiResponseError:
                pass
        pending.append((day, raw_path))

    if not pending:
        return

    print(
        f"병렬 원문 수집 시작: 미수집 {len(pending)}일, 동시 요청 {parallel}개",
        flush=True,
    )
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        config_lines: list[str] = []
        temporary_paths: list[tuple[pd.Timestamp, Path, Path]] = []
        for day, raw_path in batch:
            temporary = raw_path.with_suffix(raw_path.suffix + ".download")
            temporary.unlink(missing_ok=True)
            params = urllib.parse.urlencode(
                {
                    "tm": day.strftime("%Y%m%d%H%M"),
                    "stn": 0,
                    "help": 0,
                    "authKey": api_key,
                }
            )
            url = f"{API_URL}?{params}"
            config_lines.extend(
                [
                    f'url = "{url}"',
                    f'output = "{temporary.resolve()}"',
                ]
            )
            temporary_paths.append((day, temporary, raw_path))

        process = subprocess.run(
            [
                "curl",
                "--parallel",
                "--parallel-immediate",
                "--parallel-max",
                str(parallel),
                "--retry",
                str(max(0, max_retries - 1)),
                "--retry-delay",
                "1",
                "--max-time",
                str(int(timeout)),
                "--fail-with-body",
                "--remove-on-error",
                "--silent",
                "--show-error",
                "--config",
                "-",
            ],
            input="\n".join(config_lines) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )

        valid_count = 0
        blocking_error: str | None = None
        for day, temporary, raw_path in temporary_paths:
            if not temporary.exists():
                continue
            try:
                text = temporary.read_text(encoding="utf-8", errors="replace")
                parse_response(text)
                temporary.replace(raw_path)
                valid_count += 1
            except ApiResponseError as exc:
                temporary.unlink(missing_ok=True)
                message = str(exc)
                if "활용신청" in message or "인증" in message or "제한" in message:
                    blocking_error = message

        completed = min(batch_start + len(batch), len(pending))
        print(
            f"원문 캐시 {completed:03d}/{len(pending)}: 이번 배치 정상 {valid_count}일",
            flush=True,
        )
        if process.returncode != 0:
            safe_error = process.stderr.replace(api_key, "{REDACTED}").strip()
            if safe_error:
                print(f"curl 경고: {safe_error[:500]}", file=sys.stderr)
        if blocking_error:
            print(f"API가 수집 중단을 요구했습니다: {blocking_error}", file=sys.stderr)
            break


def make_day_result(
    day: pd.Timestamp,
    target: pd.DataFrame,
    api_rows: int,
    source: str,
    station_ids: list[int],
) -> DayResult:
    present = set(target["STN_ID"].astype(int))
    missing = [value for value in station_ids if value not in present]
    return DayResult(
        date=day.strftime("%Y-%m-%d"),
        timestamp_kst=day.strftime("%Y-%m-%d %H:%M:%S"),
        status="ok",
        source=source,
        api_rows=api_rows,
        target_rows=len(target),
        ta_valid=int(target["TA"].notna().sum()),
        hm_valid=int(target["HM"].notna().sum()),
        missing_station_count=len(missing),
        missing_station_ids=":".join(map(str, missing)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--station-list", type=Path, default=Path("data/metadata/station_list.csv")
    )
    parser.add_argument("--api-key-file", type=Path, default=Path("upload/kma_key.txt"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/asos_2019_2025_jja_1400kst"),
    )
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--hour-kst", type=int, default=14)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--request-interval", type=float, default=0.25)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()

    station_ids = load_station_ids(args.station_list)
    station_set = set(station_ids)
    api_key = read_api_key(args.api_key_file)
    dates = target_dates(args.start_year, args.end_year, args.months)
    raw_dir = args.output_dir / "raw"
    parsed_dir = args.output_dir / "parsed"
    manifest_path = args.output_dir / "collection_manifest.csv"

    results: list[DayResult] = []
    frames: list[pd.DataFrame] = []
    failed = False

    parallel_download_raw(
        dates,
        hour_kst=args.hour_kst,
        raw_dir=raw_dir,
        api_key=api_key,
        parallel=args.parallel,
        batch_size=args.batch_size,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    for index, date_value in enumerate(dates, start=1):
        day = date_value.replace(hour=args.hour_kst)
        key = day.strftime("%Y%m%d%H%M")
        raw_path = raw_dir / f"asos_{key}.txt"
        parsed_path = parsed_dir / f"asos_{key}.csv"
        source = "raw_cache"
        try:
            if raw_path.exists():
                text = raw_path.read_text(encoding="utf-8", errors="replace")
            else:
                text = fetch_text(
                    day,
                    api_key,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
                atomic_write_text(raw_path, text)
                source = "api"
            full_frame = parse_response(text)
            api_rows = len(full_frame)
            target = full_frame[full_frame["STN_ID"].isin(station_set)].copy()
            atomic_write_csv(parsed_path, target)

            target["STN_ID"] = pd.to_numeric(target["STN_ID"], errors="raise").astype(int)
            target["TA"] = pd.to_numeric(target["TA"], errors="coerce")
            target["HM"] = pd.to_numeric(target["HM"], errors="coerce")
            expected_timestamp = day.strftime("%Y-%m-%d %H:%M:%S")
            if not target["timestamp_kst"].eq(expected_timestamp).all():
                raise ValueError("요청 시각과 응답 시각이 일치하지 않습니다.")
            if target.duplicated(["timestamp_kst", "STN_ID"]).any():
                raise ValueError("필터 결과에 중복 관측소가 있습니다.")
            if not set(target["STN_ID"]).issubset(station_set):
                raise ValueError("대회 대상이 아닌 관측소가 포함됐습니다.")

            results.append(make_day_result(day, target, api_rows, source, station_ids))
            frames.append(target)
            if index == 1 or index % 10 == 0 or index == len(dates):
                print(
                    f"[{index:03d}/{len(dates)}] {key} 완료: "
                    f"대상 {len(target)}개, TA {target['TA'].notna().sum()}개, "
                    f"HM {target['HM'].notna().sum()}개",
                    flush=True,
                )
        except Exception as exc:
            failed = True
            results.append(
                DayResult(
                    date=day.strftime("%Y-%m-%d"),
                    timestamp_kst=day.strftime("%Y-%m-%d %H:%M:%S"),
                    status="failed",
                    source=source,
                    api_rows=0,
                    target_rows=0,
                    ta_valid=0,
                    hm_valid=0,
                    missing_station_count=len(station_ids),
                    missing_station_ids=":".join(map(str, station_ids)),
                    message=str(exc),
                )
            )
            print(f"[{index:03d}/{len(dates)}] {key} 실패: {exc}", file=sys.stderr)
            if "활용신청" in str(exc) or "인증" in str(exc) or "제한" in str(exc):
                print("권한·인증·호출 제한 오류이므로 안전하게 중단합니다.", file=sys.stderr)
                break

        atomic_write_csv(manifest_path, pd.DataFrame(asdict(item) for item in results))
        if source == "api" and args.request_interval > 0:
            time.sleep(args.request_interval)

    manifest = pd.DataFrame(asdict(item) for item in results)
    atomic_write_csv(manifest_path, manifest)

    if frames:
        observed = pd.concat(frames, ignore_index=True)
        observed = observed.drop_duplicates(["date", "STN_ID"], keep="last")
    else:
        observed = pd.DataFrame(columns=["date", "timestamp_kst", "STN_ID", "TA", "HM"])

    date_frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    station_frame = pd.DataFrame({"STN_ID": station_ids})
    grid = date_frame.merge(station_frame, how="cross")
    combined = grid.merge(
        observed[["date", "STN_ID", "TA", "HM"]],
        on=["date", "STN_ID"],
        how="left",
        validate="one_to_one",
    )
    combined.insert(
        1,
        "timestamp_kst",
        pd.to_datetime(combined["date"]).dt.strftime(f"%Y-%m-%d {args.hour_kst:02d}:00:00"),
    )
    combined = combined.sort_values(["date", "STN_ID"]).reset_index(drop=True)
    combined_path = args.output_dir / "asos_2019_2025_jja_1400kst.csv"
    atomic_write_csv(combined_path, combined)

    success_dates = int((manifest["status"] == "ok").sum()) if len(manifest) else 0
    report = {
        "period": f"{args.start_year}-06-01~{args.end_year}-08-31 (June/July/August only)",
        "observation_time": f"{args.hour_kst:02d}:00 KST",
        "requested_dates": len(dates),
        "successful_dates": success_dates,
        "failed_or_unattempted_dates": len(dates) - success_dates,
        "competition_station_count": len(station_ids),
        "expected_rows": len(dates) * len(station_ids),
        "combined_rows": len(combined),
        "ta_valid_rows": int(combined["TA"].notna().sum()),
        "hm_valid_rows": int(combined["HM"].notna().sum()),
        "duplicate_date_station_rows": int(combined.duplicated(["date", "STN_ID"]).sum()),
        "ta_out_of_physical_check_range": int(
            ((combined["TA"] < -90) | (combined["TA"] > 60)).fillna(False).sum()
        ),
        "hm_out_of_range": int(
            ((combined["HM"] < 0) | (combined["HM"] > 100)).fillna(False).sum()
        ),
        "external_station_metadata_columns": [],
        "allowed_columns": ["date", "timestamp_kst", "STN_ID", "TA", "HM"],
        "api_key_in_output": False,
    }
    report_path = args.output_dir / "validation_report.json"
    atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 1 if failed or success_dates != len(dates) else 0


if __name__ == "__main__":
    raise SystemExit(main())
