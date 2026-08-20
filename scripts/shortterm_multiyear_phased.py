from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _common import inclusive_dates
from gk2a_weather.config import load_yaml
from gk2a_weather.data.gk2a import api_timestamp, canonical_gk2a_path


def time_grid(day: pd.Timestamp, start_hhmm: str, end_hhmm: str, step_minutes: int):
    sh, sm = map(int, start_hhmm.split(":"))
    eh, em = map(int, end_hhmm.split(":"))
    current = day.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = day.replace(hour=eh, minute=em, second=0, microsecond=0)
    while current <= end:
        yield current.to_pydatetime()
        current += pd.Timedelta(minutes=step_minutes)


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"하위 명령이 exit code {exc.returncode}로 실패했습니다: {' '.join(cmd)}"
        ) from exc


def expected_year_status(year: int, args, config: dict) -> dict:
    output_root = Path(args.output_root).expanduser().resolve()
    raw_root = output_root / "raw_gk2a"
    asos_root = output_root / "asos" / "parsed"
    satellite = config["satellite"]
    channels = [str(c).upper() for c in satellite["channels"]]
    min_bytes = int(satellite["minimum_file_bytes"])

    start = f"{year}-{args.start_mmdd}"
    end = f"{year}-{args.end_mmdd}"
    expected_nc = 0
    valid_nc = 0
    missing_examples: list[str] = []

    for day in inclusive_dates(start, end):
        for ts in time_grid(day, args.start_time, args.end_time, args.step_minutes):
            requested = api_timestamp(ts, str(satellite["api_time_basis"]))
            for channel in channels:
                expected_nc += 1
                path = canonical_gk2a_path(raw_root, day, channel, requested)
                if path.exists() and path.stat().st_size >= min_bytes:
                    valid_nc += 1
                elif len(missing_examples) < 8:
                    missing_examples.append(str(path))

    expected_asos = len(list(inclusive_dates(start, end)))
    valid_asos = 0
    for day in inclusive_dates(start, end):
        key = day.replace(hour=14, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
        path = asos_root / f"asos_{key}.csv"
        if path.exists() and path.stat().st_size > 0:
            valid_asos += 1

    return {
        "year": year,
        "valid_nc": valid_nc,
        "expected_nc": expected_nc,
        "valid_asos": valid_asos,
        "expected_asos": expected_asos,
        "complete": valid_nc == expected_nc and valid_asos == expected_asos,
        "missing_examples": missing_examples,
    }


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def yearly_build_is_reusable(year_dir: Path, source_status: dict) -> bool:
    """Return True only for a finished per-year build that is safe to reuse."""
    long_df = safe_read_csv(year_dir / "shortterm_long.csv")
    wide_df = safe_read_csv(year_dir / "shortterm_wide.csv")
    labels_df = safe_read_csv(year_dir / "shortterm_labels_1400.csv")
    if not len(long_df) or not len(wide_df) or not len(labels_df):
        return False

    # A previous build may have been made before later downloads filled the raw gaps.
    # Rebuild that year so stale NaNs do not remain in otherwise complete source data.
    missing_df = safe_read_csv(year_dir / "shortterm_build_missing.csv")
    if bool(source_status["complete"]) and len(missing_df) > 0:
        return False

    # Never reuse an output created from an empty/wrong Drive root.
    maximum_reasonable_issues = max(1, int(source_status["expected_nc"]) // 2)
    return len(missing_df) < maximum_reasonable_issues


def validate_build_source_root(statuses: list[dict], output_root: Path) -> None:
    expected_nc = sum(int(status["expected_nc"]) for status in statuses)
    valid_nc = sum(int(status["valid_nc"]) for status in statuses)
    valid_asos = sum(int(status["valid_asos"]) for status in statuses)
    minimum_nc = max(1, expected_nc // 2)
    if valid_nc >= minimum_nc and valid_asos > 0:
        return

    raise RuntimeError(
        "Phase 2 입력 폴더에 실제 원본이 거의 보이지 않습니다. "
        f"GK2A={valid_nc}/{expected_nc}, ASOS={valid_asos}, root={output_root}\n"
        "이 상태에서 계속하면 모든 위성값과 라벨이 비어 있는 CSV가 생성됩니다. "
        "Colab에서 데이터를 보유한 Google 계정을 마운트했는지, 또는 "
        "OUTPUT_ROOT가 실제 shortterm_12to14_data 폴더를 가리키는지 확인하세요."
    )


def collect_phase(args, config: dict, repo_dir: Path) -> None:
    root = Path(args.output_root).expanduser().resolve()
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    status_rows = []

    for year in args.years:
        before = expected_year_status(year, args, config)
        print("\n" + "=" * 80, flush=True)
        print(
            f"[CHECK] {year}: GK2A {before['valid_nc']}/{before['expected_nc']} | "
            f"ASOS {before['valid_asos']}/{before['expected_asos']}",
            flush=True,
        )

        if before["complete"]:
            print(f"[SKIP DOWNLOAD] {year}: 원본 데이터가 이미 모두 존재합니다.", flush=True)
            after = before
        else:
            print(f"[RESUME DOWNLOAD] {year}: 없는/불완전한 원본만 이어받습니다.", flush=True)
            cmd = [
                sys.executable, "-u", "scripts/collect_shortterm_12to14.py",
                "--start", f"{year}-{args.start_mmdd}",
                "--end", f"{year}-{args.end_mmdd}",
                "--start-time", args.start_time,
                "--end-time", args.end_time,
                "--step-minutes", str(args.step_minutes),
                "--output-root", str(root),
                "--config", args.config,
            ]
            if args.station_list:
                cmd += ["--station-list", args.station_list]
            run(cmd, repo_dir)

            latest = outputs / "shortterm_collection_failures.csv"
            yearly = outputs / f"collection_failures_{year}.csv"
            if latest.exists():
                shutil.copy2(latest, yearly)
            after = expected_year_status(year, args, config)

        print(
            f"[AFTER] {year}: GK2A {after['valid_nc']}/{after['expected_nc']} | "
            f"ASOS {after['valid_asos']}/{after['expected_asos']} | complete={after['complete']}",
            flush=True,
        )
        status_rows.append({k: v for k, v in after.items() if k != "missing_examples"})

        # If this year still failed after retries, stop here rather than hammering later years.
        if not after["complete"]:
            print("[STOP] 이 연도의 원본이 아직 불완전합니다. 같은 collect 명령을 다시 실행하면 이어받습니다.", flush=True)
            if after["missing_examples"]:
                print("missing examples:", *after["missing_examples"], sep="\n  - ", flush=True)
            break

    status = pd.DataFrame(status_rows)
    status_path = outputs / "multiyear_collection_status.csv"
    status.to_csv(status_path, index=False)
    print(f"\ncollection status -> {status_path}", flush=True)

    full_status = pd.DataFrame([expected_year_status(y, args, config) for y in args.years])
    printable = full_status.drop(columns=["missing_examples"])
    print("\n=== CURRENT COLLECTION STATUS ===", flush=True)
    print(printable.to_string(index=False), flush=True)
    all_complete = bool(full_status["complete"].all())
    print(f"\nALL_COLLECTION_COMPLETE={all_complete}", flush=True)


def combine_years(args) -> None:
    root = Path(args.output_root).expanduser().resolve()
    by_year = root / "datasets" / "by_year"
    combined = root / "datasets" / "combined"
    combined.mkdir(parents=True, exist_ok=True)

    longs, wides, labels, missings = [], [], [], []
    summary = []
    missing_dataset_parts: list[str] = []
    for year in args.years:
        d = by_year / str(year)
        long_df = safe_read_csv(d / "shortterm_long.csv")
        wide_df = safe_read_csv(d / "shortterm_wide.csv")
        labels_df = safe_read_csv(d / "shortterm_labels_1400.csv")
        missing_df = safe_read_csv(d / "shortterm_build_missing.csv")
        for df in (long_df, wide_df, labels_df, missing_df):
            if len(df) and "Year" not in df.columns:
                df.insert(0, "Year", year)
        if len(long_df): longs.append(long_df)
        if len(wide_df): wides.append(wide_df)
        if len(labels_df): labels.append(labels_df)
        if len(missing_df): missings.append(missing_df)
        empty_parts = [
            name
            for name, frame in (
                ("long", long_df),
                ("wide", wide_df),
                ("labels", labels_df),
            )
            if not len(frame)
        ]
        if empty_parts:
            missing_dataset_parts.append(f"{year}({','.join(empty_parts)})")
        summary.append({
            "year": year,
            "long_rows": len(long_df),
            "wide_rows": len(wide_df),
            "label_rows": len(labels_df),
            "build_issues": len(missing_df),
            "TA_missing": int(labels_df["TA"].isna().sum()) if "TA" in labels_df else 0,
            "HM_missing": int(labels_df["HM"].isna().sum()) if "HM" in labels_df else 0,
        })

    if missing_dataset_parts:
        raise RuntimeError(
            "연도별 build 결과 일부가 비어 있어 combined CSV를 만들 수 없습니다: "
            + ", ".join(missing_dataset_parts)
        )

    year_tag = f"{min(args.years)}to{max(args.years)}"
    if longs:
        pd.concat(longs, ignore_index=True).sort_values(["Date", "STN_ID", "TimeKST"]).to_csv(
            combined / f"shortterm_long_{year_tag}.csv", index=False
        )
    if wides:
        pd.concat(wides, ignore_index=True).sort_values(["Date", "STN_ID"]).to_csv(
            combined / f"shortterm_wide_{year_tag}.csv", index=False
        )
    if labels:
        pd.concat(labels, ignore_index=True).sort_values(["Date", "STN_ID"]).to_csv(
            combined / f"shortterm_labels_1400_{year_tag}.csv", index=False
        )
    (pd.concat(missings, ignore_index=True) if missings else pd.DataFrame()).to_csv(
        combined / f"shortterm_build_missing_{year_tag}.csv", index=False
    )
    pd.DataFrame(summary).to_csv(combined / "shortterm_multiyear_summary.csv", index=False)
    print(f"[COMBINE DONE] -> {combined}", flush=True)


def build_phase(args, config: dict, repo_dir: Path) -> None:
    statuses = [expected_year_status(y, args, config) for y in args.years]
    root = Path(args.output_root).expanduser().resolve()
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{k: v for k, v in status.items() if k != "missing_examples"} for status in statuses]
    ).to_csv(outputs / "build_preflight_status.csv", index=False)
    validate_build_source_root(statuses, root)

    incomplete = [s for s in statuses if not s["complete"]]
    if incomplete:
        msg = ", ".join(
            f"{s['year']} GK2A={s['valid_nc']}/{s['expected_nc']} ASOS={s['valid_asos']}/{s['expected_asos']}"
            for s in incomplete
        )
        if not args.allow_incomplete:
            raise RuntimeError(
                "원본 수집이 아직 완료되지 않아 build를 중단합니다: "
                + msg
                + "\nPhase 1을 다시 실행하거나, 누락 채널을 NaN으로 기록해 진행하려면 "
                "--allow-incomplete를 지정하세요."
            )
        print(
            "[BUILD PRECHECK WARNING] 불완전한 원본을 허용합니다. "
            "누락 채널은 NaN으로 저장되고 shortterm_build_missing CSV에 기록됩니다.",
            flush=True,
        )
        print("[INCOMPLETE] " + msg, flush=True)
        for status in incomplete:
            if status["missing_examples"]:
                print(
                    f"[{status['year']} missing examples]",
                    *status["missing_examples"],
                    sep="\n  - ",
                    flush=True,
                )
    else:
        print("[BUILD PRECHECK] 모든 원본 수집 완료.", flush=True)

    by_year = root / "datasets" / "by_year"
    by_year.mkdir(parents=True, exist_ok=True)

    status_by_year = {status["year"]: status for status in statuses}
    print("[BUILD PHASE] tabular/long 생성을 시작합니다.", flush=True)
    for year in args.years:
        year_dir = by_year / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80, flush=True)
        if args.resume_build and yearly_build_is_reusable(
            year_dir, status_by_year[year]
        ):
            print(f"[SKIP BUILD] {year}: 기존 연도별 결과를 재사용합니다.", flush=True)
            continue
        print(f"[BUILD] {year}", flush=True)
        cmd = [
            sys.executable, "-u", "scripts/build_shortterm_12to14_dataset.py",
            "--start", f"{year}-{args.start_mmdd}",
            "--end", f"{year}-{args.end_mmdd}",
            "--start-time", args.start_time,
            "--end-time", args.end_time,
            "--step-minutes", str(args.step_minutes),
            "--input-root", str(root),
            "--output-dir", str(year_dir),
            "--config", args.config,
        ]
        if args.station_list:
            cmd += ["--station-list", args.station_list]
        run(cmd, repo_dir)

    combine_years(args)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect short-term raw data and build per-year/combined tabular datasets"
    )
    p.add_argument("--phase", required=True, choices=["collect", "status", "build"])
    p.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2026)))
    p.add_argument("--start-mmdd", default="08-24")
    p.add_argument("--end-mmdd", default="08-30")
    p.add_argument("--start-time", default="12:00")
    p.add_argument("--end-time", default="14:00")
    p.add_argument("--step-minutes", type=int, default=10)
    p.add_argument("--output-root", required=True)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--station-list", default="")
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build with available raw files; missing channels are saved as NaN and logged",
    )
    p.add_argument(
        "--resume-build",
        action="store_true",
        help="Reuse finished per-year outputs, but rebuild stale outputs that still log gaps after raw completion",
    )
    args = p.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    config = load_yaml(args.config)

    if args.phase == "collect":
        collect_phase(args, config, repo_dir)
    elif args.phase == "status":
        rows = []
        for year in args.years:
            s = expected_year_status(year, args, config)
            rows.append({k: v for k, v in s.items() if k != "missing_examples"})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print("ALL_COLLECTION_COMPLETE=", bool(df["complete"].all()))
    else:
        build_phase(args, config, repo_dir)


if __name__ == "__main__":
    main()
