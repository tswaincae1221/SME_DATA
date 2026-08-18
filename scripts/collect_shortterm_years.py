from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_YEARS = list(range(2019, 2026))
FAILURE_COLUMNS = ["kind", "date_kst", "time_kst", "channel", "error"]


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


def year_paths(by_year_root: Path, year: int) -> dict[str, Path]:
    year_dir = by_year_root / str(year)
    return {
        "dir": year_dir,
        "long": year_dir / "shortterm_long.csv",
        "wide": year_dir / "shortterm_wide.csv",
        "labels": year_dir / "shortterm_labels_1400.csv",
        "missing": year_dir / "shortterm_build_missing.csv",
    }


def built_output_status(by_year_root: Path, year: int, expected_days: int, expected_steps: int) -> tuple[bool, str]:
    paths = year_paths(by_year_root, year)
    required = [paths["long"], paths["wide"], paths["labels"]]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False, "required CSV missing"

    try:
        long_df = safe_read_csv(paths["long"])
        wide_df = safe_read_csv(paths["wide"])
        labels_df = safe_read_csv(paths["labels"])
        if long_df.empty or wide_df.empty or labels_df.empty:
            return False, "one or more CSVs empty"
        if not {"Date", "STN_ID", "TimeKST"}.issubset(long_df.columns):
            return False, "long CSV columns invalid"
        if not {"Date", "STN_ID", "TA", "HM"}.issubset(wide_df.columns):
            return False, "wide CSV columns invalid"
        if long_df["Date"].nunique() != expected_days or wide_df["Date"].nunique() != expected_days:
            return False, "date count incomplete"
        counts = long_df.groupby(["Date", "STN_ID"])["TimeKST"].nunique()
        if counts.empty or int(counts.min()) != expected_steps or int(counts.max()) != expected_steps:
            return False, "timestep count incomplete"
        if len(long_df) != len(wide_df) * expected_steps:
            return False, "long/wide row count mismatch"
        if len(labels_df) != len(wide_df):
            return False, "label/wide row count mismatch"
    except Exception as exc:
        return False, f"validation error: {exc}"

    issues = len(safe_read_csv(paths["missing"]))
    if issues:
        return True, f"built outputs valid; WARNING build_issues={issues}"
    return True, "built outputs valid"


def add_year_column(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    out = frame.copy()
    if "Year" in out.columns:
        out["Year"] = year
    else:
        out.insert(0, "Year", year)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "여러 연도의 short-term 위성 자료를 수집하고 tabular/long 데이터와 "
            "14:00 TA/HM 라벨을 자동 병합합니다. 재실행 시 이미 build가 끝난 연도는 건너뜁니다."
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
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--rebuild-existing-years",
        action="store_true",
        help="이미 long/wide/labels가 완성된 연도도 다시 수집/전처리",
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

    start_month, start_day = args.start_mmdd
    end_month, end_day = args.end_mmdd
    example_year = years[0]
    expected_days = len(pd.date_range(
        f"{example_year:04d}-{start_month:02d}-{start_day:02d}",
        f"{example_year:04d}-{end_month:02d}-{end_day:02d}",
        freq="D",
    ))
    start_minutes = int(args.start_time[:2]) * 60 + int(args.start_time[3:])
    end_minutes = int(args.end_time[:2]) * 60 + int(args.end_time[3:])
    expected_steps = (end_minutes - start_minutes) // args.step_minutes + 1

    print(
        f"[MULTIYEAR] years={years} expected_days/year={expected_days} "
        f"expected_steps/day={expected_steps}",
        flush=True,
    )

    # 1) 필요한 연도만 수집/build. 이미 완성된 연도는 건너뛴다.
    for year in years:
        start_date = date_text(year, args.start_mmdd)
        end_date = date_text(year, args.end_mmdd)
        paths = year_paths(by_year_root, year)
        paths["dir"].mkdir(parents=True, exist_ok=True)

        complete, status = built_output_status(by_year_root, year, expected_days, expected_steps)
        print("\n" + "=" * 72, flush=True)
        print(f"{year}: {start_date} ~ {end_date} / {args.start_time}~{args.end_time}", flush=True)
        print(f"[STATUS] {status}", flush=True)
        print("=" * 72, flush=True)

        if complete and not args.rebuild_existing_years and not args.force:
            print(f"[SKIP YEAR] {year}: 기존 long/wide/labels 재사용", flush=True)
            continue

        if not args.skip_collection:
            collect_cmd = [
                sys.executable, "-u", "scripts/collect_shortterm_12to14.py",
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
            failure_frame = safe_read_csv(yearly_failure, columns=FAILURE_COLUMNS)
            print(f"[COLLECTION RESULT] year={year} failures={len(failure_frame)}", flush=True)
        else:
            yearly_failure = output_logs / f"collection_failures_{year}.csv"
            failure_frame = safe_read_csv(yearly_failure, columns=FAILURE_COLUMNS)

        if not args.skip_build:
            build_cmd = [
                sys.executable, "-u", "scripts/build_shortterm_12to14_dataset.py",
                "--start", start_date,
                "--end", end_date,
                "--start-time", args.start_time,
                "--end-time", args.end_time,
                "--step-minutes", str(args.step_minutes),
                "--input-root", str(output_root),
                "--output-dir", str(paths["dir"]),
                "--config", args.config,
            ]
            if args.station_list:
                build_cmd += ["--station-list", args.station_list]
            print(f"[BUILD START] {year}", flush=True)
            run(build_cmd, cwd=repo_dir)
            build_missing = safe_read_csv(paths["missing"])
            print(f"[BUILD DONE] {year} issues={len(build_missing)}", flush=True)

    # 2) 기존 결과까지 포함해 요청된 모든 연도를 다시 합친다.
    all_long: list[pd.DataFrame] = []
    all_wide: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    all_build_missing: list[pd.DataFrame] = []
    all_collection_failures: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for year in years:
        paths = year_paths(by_year_root, year)
        long_df = safe_read_csv(paths["long"])
        wide_df = safe_read_csv(paths["wide"])
        labels_df = safe_read_csv(paths["labels"])
        missing_df = safe_read_csv(paths["missing"])
        yearly_failure = output_logs / f"collection_failures_{year}.csv"
        failure_df = safe_read_csv(yearly_failure, columns=FAILURE_COLUMNS)

        if len(long_df):
            all_long.append(add_year_column(long_df, year))
        if len(wide_df):
            all_wide.append(add_year_column(wide_df, year))
        if len(labels_df):
            all_labels.append(add_year_column(labels_df, year))
        if len(missing_df):
            all_build_missing.append(add_year_column(missing_df, year))
        if len(failure_df):
            all_collection_failures.append(add_year_column(failure_df, year))

        summary_rows.append({
            "year": year,
            "long_rows": len(long_df),
            "wide_rows": len(wide_df),
            "label_rows": len(labels_df),
            "TA_missing": int(labels_df["TA"].isna().sum()) if "TA" in labels_df else 0,
            "HM_missing": int(labels_df["HM"].isna().sum()) if "HM" in labels_df else 0,
            "build_issues": len(missing_df),
            "collection_failures": len(failure_df),
        })

    combined_dir = output_root / "datasets" / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_long = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    combined_wide = pd.concat(all_wide, ignore_index=True) if all_wide else pd.DataFrame()
    combined_labels = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()

    if len(combined_long):
        combined_long = combined_long.sort_values(["Date", "STN_ID", "TimeKST"]).reset_index(drop=True)
        combined_long.to_csv(combined_dir / "shortterm_long_2019to2025.csv", index=False)
    if len(combined_wide):
        combined_wide = combined_wide.sort_values(["Date", "STN_ID"]).reset_index(drop=True)
        combined_wide.to_csv(combined_dir / "shortterm_wide_2019to2025.csv", index=False)
    if len(combined_labels):
        combined_labels = combined_labels.sort_values(["Date", "STN_ID"]).reset_index(drop=True)
        combined_labels.to_csv(combined_dir / "shortterm_labels_1400_2019to2025.csv", index=False)

    combined_missing = pd.concat(all_build_missing, ignore_index=True) if all_build_missing else pd.DataFrame(columns=["Year", "Date", "TimeKST", "channel", "reason"])
    combined_failures = pd.concat(all_collection_failures, ignore_index=True) if all_collection_failures else pd.DataFrame(columns=["Year", *FAILURE_COLUMNS])
    combined_missing.to_csv(combined_dir / "shortterm_build_missing_2019to2025.csv", index=False)
    combined_failures.to_csv(combined_dir / "shortterm_collection_failures_2019to2025.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(combined_dir / "shortterm_multiyear_summary.csv", index=False)

    print("\n" + "=" * 72, flush=True)
    print("MULTI-YEAR PIPELINE COMPLETE", flush=True)
    print("=" * 72, flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"combined long={combined_long.shape} wide={combined_wide.shape} labels={combined_labels.shape}", flush=True)
    print(f"collection failures={len(combined_failures)} build issues={len(combined_missing)}", flush=True)
    print(f"output={combined_dir}", flush=True)


if __name__ == "__main__":
    main()
