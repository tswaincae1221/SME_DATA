from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_YEARS = list(range(2019, 2026))


def parse_mmdd(value: str) -> tuple[int, int]:
    try:
        month_text, day_text = value.split("-")
        month, day = int(month_text), int(day_text)
        datetime(2000, month, day)
    except Exception as exc:
        raise argparse.ArgumentTypeError("MM-DD 형식으로 입력하세요. 예: 08-24") from exc
    return month, day


def date_text(year: int, mmdd: tuple[int, int]) -> str:
    month, day = mmdd
    return f"{year:04d}-{month:02d}-{day:02d}"


def run(cmd: list[str], *, cwd: Path) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def safe_read_csv(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "2019~2025 등 여러 연도의 8/24~8/30 short-term 위성 자료를 수집하고, "
            "지점별 tabular 데이터 및 14:00 TA/HM 라벨까지 자동 병합합니다."
        )
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--start-mmdd", type=parse_mmdd, default=parse_mmdd("08-24"))
    parser.add_argument("--end-mmdd", type=parse_mmdd, default=parse_mmdd("08-30"))
    parser.add_argument("--start-time", default="12:00")
    parser.add_argument("--end-time", default="14:00")
    parser.add_argument("--step-minutes", type=int, default=10)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--station-list", default="")
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="이미 원본 수집이 끝났다면 다운로드를 생략하고 전처리/병합만 수행",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="수집만 하고 tabular 변환/라벨 병합은 생략",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    years = sorted(dict.fromkeys(args.years))
    if not years:
        raise ValueError("최소 한 개 연도를 지정해야 합니다.")

    by_year_root = output_root / "datasets" / "by_year"
    by_year_root.mkdir(parents=True, exist_ok=True)
    output_logs = output_root / "outputs"
    output_logs.mkdir(parents=True, exist_ok=True)

    all_long: list[pd.DataFrame] = []
    all_wide: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    all_build_missing: list[pd.DataFrame] = []
    all_collection_failures: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for year in years:
        start_date = date_text(year, args.start_mmdd)
        end_date = date_text(year, args.end_mmdd)
        print("\n" + "=" * 72)
        print(f"{year}: {start_date} ~ {end_date} / {args.start_time}~{args.end_time}")
        print("=" * 72)

        if not args.skip_collection:
            collect_cmd = [
                sys.executable,
                "scripts/collect_shortterm_12to14.py",
                "--start", start_date,
                "--end", end_date,
                "--start-time", args.start_time,
                "--end-time", args.end_time,
                "--step-minutes", str(args.step_minutes),
                "--output-root", str(output_root),
                "--config", args.config,
            ]
            if args.station_list:
                collect_cmd += ["--station-list", args.station_list]
            if args.force:
                collect_cmd.append("--force")
            run(collect_cmd, cwd=repo_dir)

            latest_failure = output_logs / "shortterm_collection_failures.csv"
            yearly_failure = output_logs / f"collection_failures_{year}.csv"
            if latest_failure.exists():
                shutil.copy2(latest_failure, yearly_failure)
            failure_frame = safe_read_csv(
                yearly_failure,
                columns=["kind", "date_kst", "time_kst", "channel", "error"],
            )
            if len(failure_frame):
                failure_frame.insert(0, "year", year)
                all_collection_failures.append(failure_frame)
        else:
            yearly_failure = output_logs / f"collection_failures_{year}.csv"
            failure_frame = safe_read_csv(
                yearly_failure,
                columns=["kind", "date_kst", "time_kst", "channel", "error"],
            )
            if len(failure_frame):
                failure_frame.insert(0, "year", year)
                all_collection_failures.append(failure_frame)

        year_dir = by_year_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_build:
            build_cmd = [
                sys.executable,
                "scripts/build_shortterm_12to14_dataset.py",
                "--start", start_date,
                "--end", end_date,
                "--start-time", args.start_time,
                "--end-time", args.end_time,
                "--step-minutes", str(args.step_minutes),
                "--input-root", str(output_root),
                "--output-dir", str(year_dir),
                "--config", args.config,
            ]
            if args.station_list:
                build_cmd += ["--station-list", args.station_list]
            run(build_cmd, cwd=repo_dir)

        long_path = year_dir / "shortterm_long.csv"
        wide_path = year_dir / "shortterm_wide.csv"
        labels_path = year_dir / "shortterm_labels_1400.csv"
        missing_path = year_dir / "shortterm_build_missing.csv"

        long_df = safe_read_csv(long_path)
        wide_df = safe_read_csv(wide_path)
        labels_df = safe_read_csv(labels_path)
        missing_df = safe_read_csv(missing_path)

        if len(long_df):
            long_df.insert(0, "Year", year)
            all_long.append(long_df)
        if len(wide_df):
            wide_df.insert(0, "Year", year)
            all_wide.append(wide_df)
        if len(labels_df):
            labels_df.insert(0, "Year", year)
            all_labels.append(labels_df)
        if len(missing_df):
            missing_df.insert(0, "Year", year)
            all_build_missing.append(missing_df)

        label_missing_ta = int(labels_df["TA"].isna().sum()) if "TA" in labels_df else 0
        label_missing_hm = int(labels_df["HM"].isna().sum()) if "HM" in labels_df else 0
        summary_rows.append(
            {
                "year": year,
                "start_date": start_date,
                "end_date": end_date,
                "long_rows": len(long_df),
                "wide_rows": len(wide_df),
                "label_rows": len(labels_df),
                "TA_missing": label_missing_ta,
                "HM_missing": label_missing_hm,
                "build_issues": len(missing_df),
                "collection_failures": len(failure_frame),
            }
        )

    combined_dir = output_root / "datasets" / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    if all_long:
        combined_long = pd.concat(all_long, ignore_index=True)
        combined_long = combined_long.sort_values(["Date", "STN_ID", "TimeKST"]).reset_index(drop=True)
        combined_long.to_csv(combined_dir / "shortterm_long_2019to2025.csv", index=False)
    else:
        combined_long = pd.DataFrame()

    if all_wide:
        combined_wide = pd.concat(all_wide, ignore_index=True)
        combined_wide = combined_wide.sort_values(["Date", "STN_ID"]).reset_index(drop=True)
        combined_wide.to_csv(combined_dir / "shortterm_wide_2019to2025.csv", index=False)
    else:
        combined_wide = pd.DataFrame()

    if all_labels:
        combined_labels = pd.concat(all_labels, ignore_index=True)
        combined_labels = combined_labels.sort_values(["Date", "STN_ID"]).reset_index(drop=True)
        combined_labels.to_csv(combined_dir / "shortterm_labels_1400_2019to2025.csv", index=False)
    else:
        combined_labels = pd.DataFrame()

    combined_missing = (
        pd.concat(all_build_missing, ignore_index=True)
        if all_build_missing
        else pd.DataFrame(columns=["Year", "Date", "TimeKST", "channel", "reason"])
    )
    combined_missing.to_csv(combined_dir / "shortterm_build_missing_2019to2025.csv", index=False)

    combined_failures = (
        pd.concat(all_collection_failures, ignore_index=True)
        if all_collection_failures
        else pd.DataFrame(columns=["year", "kind", "date_kst", "time_kst", "channel", "error"])
    )
    combined_failures.to_csv(combined_dir / "shortterm_collection_failures_2019to2025.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(combined_dir / "shortterm_multiyear_summary.csv", index=False)

    print("\n" + "=" * 72)
    print("MULTI-YEAR PIPELINE COMPLETE")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\ncombined long : {combined_long.shape}")
    print(f"combined wide : {combined_wide.shape}")
    print(f"combined labels: {combined_labels.shape}")
    print(f"collection failures: {len(combined_failures)}")
    print(f"build issues: {len(combined_missing)}")
    print(f"output: {combined_dir}")


if __name__ == "__main__":
    main()
