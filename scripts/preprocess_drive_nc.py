#!/usr/bin/env python3
"""Colab에 마운트한 Drive의 GK-2A NC를 관측소 피처로 변환한다."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd

from _common import ROOT

from gk2a_weather.constants import GK2A_CHANNELS
from gk2a_weather.data.stations import load_station_list
from gk2a_weather.drive_preprocess import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    combine_daily_parquets,
    discover_nc_files,
    group_nc_records,
    process_timestamp,
    record_is_in_scope,
    select_group_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Drive의 NC 최상위 폴더")
    parser.add_argument("--output-dir", required=True, help="Drive 결과 저장 폴더")
    parser.add_argument(
        "--station-list",
        default=str(ROOT / "data/metadata/station_list.csv"),
        help="운영진 제공 공식 station_list.csv",
    )
    parser.add_argument("--start-date", default="2019-06-01")
    parser.add_argument("--end-date", default="2025-08-31")
    parser.add_argument("--months", nargs="+", type=int, default=[6, 7, 8])
    parser.add_argument("--target-hour-kst", type=int, default=14)
    parser.add_argument(
        "--filename-timezone",
        choices=("utc", "kst"),
        default="utc",
        help="NC 파일명의 12자리 시각 기준. KMA 파일은 보통 UTC",
    )
    parser.add_argument(
        "--radii-km",
        nargs="+",
        type=float,
        default=[0.0, 5.0, 15.0],
        help="중심과 주변 패치 반경(km)",
    )
    parser.add_argument(
        "--stage-dir",
        help="Drive 파일을 하나씩 복사해 읽을 Colab 로컬 임시 폴더",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="16채널이 덜 모인 시각도 처리. 기본값은 완성된 시각만 처리",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 생성된 날짜별 체크포인트를 다시 처리",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="전체 Parquet와 함께 압축 CSV(.csv.gz)도 저장",
    )
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="날짜별 체크포인트만 만들고 전체 병합은 생략",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="앞에서부터 지정한 개수의 시각만 처리(시험 실행용)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일을 읽지 않고 발견 개수와 누락 채널만 확인",
    )
    return parser.parse_args()


def _scope_groups(args: argparse.Namespace):
    records, unparsed = discover_nc_files(
        args.input_dir,
        source_timezone=args.filename_timezone,
    )
    scoped = [
        record
        for record in records
        if record_is_in_scope(
            record,
            start_date=args.start_date,
            end_date=args.end_date,
            months=args.months,
            target_hour_kst=args.target_hour_kst,
        )
    ]
    groups = group_nc_records(scoped)
    ordered = sorted(groups.items())
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit은 1 이상이어야 합니다.")
        ordered = ordered[: args.limit]
    return records, unparsed, ordered


def _preview(args: argparse.Namespace, records, unparsed, groups) -> None:
    expected = set(GK2A_CHANNELS)
    complete = 0
    incomplete_rows = []
    for timestamp, group in groups:
        missing = sorted(expected - set(group))
        complete += int(not missing)
        if missing:
            incomplete_rows.append(
                {
                    "timestamp_kst": timestamp.isoformat(),
                    "channels_found": len(group),
                    "missing_channels": ",".join(missing),
                }
            )

    print("\n[Drive NC 스캔 결과]")
    print(f"입력 폴더: {Path(args.input_dir)}")
    print(f"전체 NC: {len(records):,}개")
    print(f"파일명을 해석하지 못한 NC: {len(unparsed):,}개")
    print(f"조건에 맞는 14시 KST 시각: {len(groups):,}개")
    print(f"16채널 완성 시각: {complete:,}개")
    print(f"미완성 시각: {len(groups) - complete:,}개")
    if incomplete_rows:
        print("\n미완성 예시(최대 10개):")
        print(pd.DataFrame(incomplete_rows).head(10).to_string(index=False))
    if unparsed:
        print("\n해석하지 못한 파일 예시(최대 10개):")
        for path in unparsed[:10]:
            print(f"- {path}")


def main() -> None:
    args = parse_args()
    if sorted(set(args.months)) != sorted(args.months):
        args.months = sorted(set(args.months))
    if any(month < 1 or month > 12 for month in args.months):
        raise ValueError("--months는 1~12만 사용할 수 있습니다.")

    records, unparsed, groups = _scope_groups(args)
    _preview(args, records, unparsed, groups)
    if args.dry_run:
        print("\n--dry-run이므로 NC를 열거나 결과를 쓰지 않았습니다.")
        return
    if not groups:
        raise ValueError(
            "조건에 맞는 NC 묶음이 없습니다. 입력 경로, 파일명 시각대(UTC/KST), "
            "날짜 범위를 확인하세요."
        )

    stations = load_station_list(args.station_list)
    if len(stations) != 96:
        raise ValueError(f"공식 station_list는 96개 지점이어야 합니다. 현재 {len(stations)}개")

    output_root = Path(args.output_dir).expanduser()
    daily_root = output_root / "daily"
    manifest_rows: list[dict] = []
    daily_paths: list[Path] = []
    expected = set(GK2A_CHANNELS)

    for index, (timestamp, raw_group) in enumerate(groups, start=1):
        selected, duplicates = select_group_files(raw_group)
        missing = sorted(expected - set(selected))
        output_path = (
            daily_root
            / timestamp.strftime("%Y")
            / f"features_{timestamp.strftime('%Y%m%d_%H%M')}KST.parquet"
        )
        row = {
            "timestamp_kst": timestamp.isoformat(),
            "date": timestamp.strftime("%Y-%m-%d"),
            "channels_found": len(selected),
            "missing_channels": ",".join(missing),
            "duplicate_channels": ",".join(sorted(duplicates)),
            "output_file": str(output_path),
            "status": "pending",
            "error": "",
        }

        if missing and not args.allow_partial:
            row["status"] = "incomplete"
            manifest_rows.append(row)
            print(
                f"[{index}/{len(groups)}] {timestamp.date()} 건너뜀: "
                f"{len(missing)}개 채널 미업로드"
            )
            continue
        if output_path.exists() and not args.overwrite:
            row["status"] = "already_done"
            manifest_rows.append(row)
            daily_paths.append(output_path)
            print(f"[{index}/{len(groups)}] {timestamp.date()} 기존 결과 사용")
            continue

        try:
            frame = process_timestamp(
                timestamp_kst=timestamp,
                selected_files=selected,
                stations=stations,
                radii_km=tuple(args.radii_km),
                stage_dir=args.stage_dir,
            )
            if len(frame) != 96:
                raise ValueError(f"출력 행이 96개가 아닙니다: {len(frame)}")
            atomic_write_parquet(frame, output_path)
            row["status"] = "processed"
            manifest_rows.append(row)
            daily_paths.append(output_path)
            print(
                f"[{index}/{len(groups)}] {timestamp.date()} 완료: "
                f"96행 × {len(frame.columns)}열"
            )
        except Exception as exc:  # 날짜 하나의 실패가 전체 작업을 중단하지 않게 한다.
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            manifest_rows.append(row)
            print(f"[{index}/{len(groups)}] {timestamp.date()} 실패: {row['error']}")
            traceback.print_exc(limit=2)

    manifest = pd.DataFrame(manifest_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(manifest, output_root / "preprocessing_manifest.csv")
    failures = manifest[manifest["status"].isin(["failed", "incomplete"])]
    atomic_write_csv(failures, output_root / "preprocessing_failures.csv")

    status_counts = Counter(manifest["status"])
    summary = {
        "input_dir": str(Path(args.input_dir)),
        "output_dir": str(output_root),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "months": args.months,
        "target_hour_kst": args.target_hour_kst,
        "filename_timezone": args.filename_timezone,
        "nc_files_found": len(records),
        "nc_files_unparsed": len(unparsed),
        "timestamps_in_scope": len(groups),
        "status_counts": dict(status_counts),
        "daily_outputs": len(daily_paths),
        "complete_channels_required": not args.allow_partial,
        "radii_km": args.radii_km,
    }

    if daily_paths and not args.skip_finalize:
        final_name = (
            f"gk2a_station_features_{args.start_date.replace('-', '')}_"
            f"{args.end_date.replace('-', '')}_{args.target_hour_kst:02d}00KST.parquet"
        )
        final_path = output_root / final_name
        combined = combine_daily_parquets(
            daily_paths,
            output_path=final_path,
            write_csv_gzip=args.write_csv,
        )
        summary.update(
            {
                "final_parquet": str(final_path),
                "final_csv_gzip": (
                    str(final_path.with_suffix(".csv.gz")) if args.write_csv else None
                ),
                "final_rows": len(combined),
                "final_columns": len(combined.columns),
                "final_unique_dates": int(combined["date"].nunique()),
                "final_unique_stations": int(combined["STN_ID"].nunique()),
            }
        )
        print(
            f"\n전체 병합 완료: {final_path}\n"
            f"{len(combined):,}행 × {len(combined.columns):,}열, "
            f"{combined['date'].nunique():,}일"
        )

    atomic_write_json(summary, output_root / "preprocessing_summary.json")
    print("\n검수 파일:")
    print(f"- {output_root / 'preprocessing_summary.json'}")
    print(f"- {output_root / 'preprocessing_manifest.csv'}")
    print(f"- {output_root / 'preprocessing_failures.csv'}")

    if status_counts.get("failed", 0):
        print("\n실패한 날짜가 있습니다. failures.csv를 확인하고 같은 명령을 다시 실행하세요.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다. 이미 저장된 날짜별 결과는 다음 실행에서 재사용됩니다.")
        sys.exit(130)
