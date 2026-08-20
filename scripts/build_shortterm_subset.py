from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a completed subset of short-term years from existing Drive raw files only."
    )
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--start-mmdd", default="08-24")
    p.add_argument("--end-mmdd", default="08-30")
    p.add_argument("--start-time", default="12:00")
    p.add_argument("--end-time", default="14:00")
    p.add_argument("--step-minutes", type=int, default=10)
    p.add_argument("--output-root", required=True)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--station-list", default="")
    args = p.parse_args()

    years = sorted(dict.fromkeys(args.years))
    if not years:
        raise ValueError("at least one year is required")

    repo_dir = Path(__file__).resolve().parents[1]
    root = Path(args.output_root).expanduser().resolve()
    by_year_root = root / "datasets" / "by_year"
    checkpoint_root = root / "datasets" / "checkpoints"
    by_year_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Validate raw completeness using the same checker as Phase 1.
    status_cmd = [
        sys.executable,
        "-u",
        "scripts/shortterm_multiyear_phased.py",
        "--phase",
        "status",
        "--years",
        *[str(y) for y in years],
        "--start-mmdd",
        args.start_mmdd,
        "--end-mmdd",
        args.end_mmdd,
        "--start-time",
        args.start_time,
        "--end-time",
        args.end_time,
        "--step-minutes",
        str(args.step_minutes),
        "--output-root",
        str(root),
        "--config",
        args.config,
    ]
    if args.station_list:
        status_cmd += ["--station-list", args.station_list]

    print("=== RAW COMPLETENESS CHECK ===", flush=True)
    status = subprocess.run(status_cmd, cwd=repo_dir, text=True, capture_output=True)
    print(status.stdout, flush=True)
    if status.stderr:
        print(status.stderr, file=sys.stderr, flush=True)
    if status.returncode != 0:
        raise RuntimeError("raw status check failed")
    if "ALL_COLLECTION_COMPLETE= True" not in status.stdout and "ALL_COLLECTION_COMPLETE=True" not in status.stdout:
        raise RuntimeError("requested years are not fully collected; refusing to build checkpoint")

    # Build each requested year from raw files already present in Drive.
    for year in years:
        year_dir = by_year_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80, flush=True)
        print(f"[BUILD CHECKPOINT] {year}", flush=True)
        cmd = [
            sys.executable,
            "-u",
            "scripts/build_shortterm_12to14_dataset.py",
            "--start",
            f"{year}-{args.start_mmdd}",
            "--end",
            f"{year}-{args.end_mmdd}",
            "--start-time",
            args.start_time,
            "--end-time",
            args.end_time,
            "--step-minutes",
            str(args.step_minutes),
            "--input-root",
            str(root),
            "--output-dir",
            str(year_dir),
            "--config",
            args.config,
        ]
        if args.station_list:
            cmd += ["--station-list", args.station_list]
        run(cmd, repo_dir)

    tag = f"{years[0]}to{years[-1]}"
    longs: list[pd.DataFrame] = []
    wides: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    missings: list[pd.DataFrame] = []
    summary_rows: list[dict[str, int]] = []

    for year in years:
        d = by_year_root / str(year)
        long_df = safe_read_csv(d / "shortterm_long.csv")
        wide_df = safe_read_csv(d / "shortterm_wide.csv")
        labels_df = safe_read_csv(d / "shortterm_labels_1400.csv")
        missing_df = safe_read_csv(d / "shortterm_build_missing.csv")

        for df in (long_df, wide_df, labels_df, missing_df):
            if len(df):
                if "Year" in df.columns:
                    df["Year"] = year
                else:
                    df.insert(0, "Year", year)

        if len(long_df):
            longs.append(long_df)
        if len(wide_df):
            wides.append(wide_df)
        if len(labels_df):
            labels.append(labels_df)
        if len(missing_df):
            missings.append(missing_df)

        summary_rows.append(
            {
                "year": year,
                "long_rows": len(long_df),
                "wide_rows": len(wide_df),
                "label_rows": len(labels_df),
                "build_issues": len(missing_df),
                "TA_missing": int(labels_df["TA"].isna().sum()) if "TA" in labels_df else 0,
                "HM_missing": int(labels_df["HM"].isna().sum()) if "HM" in labels_df else 0,
            }
        )

    if not longs or not wides or not labels:
        raise RuntimeError("checkpoint build produced no combined data")

    long_all = pd.concat(longs, ignore_index=True).sort_values(["Date", "STN_ID", "TimeKST"]).reset_index(drop=True)
    wide_all = pd.concat(wides, ignore_index=True).sort_values(["Date", "STN_ID"]).reset_index(drop=True)
    labels_all = pd.concat(labels, ignore_index=True).sort_values(["Date", "STN_ID"]).reset_index(drop=True)
    missing_all = pd.concat(missings, ignore_index=True) if missings else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    long_path = checkpoint_root / f"shortterm_long_{tag}.csv"
    wide_path = checkpoint_root / f"shortterm_wide_{tag}.csv"
    labels_path = checkpoint_root / f"shortterm_labels_1400_{tag}.csv"
    missing_path = checkpoint_root / f"shortterm_build_missing_{tag}.csv"
    summary_path = checkpoint_root / f"shortterm_summary_{tag}.csv"

    long_all.to_csv(long_path, index=False)
    wide_all.to_csv(wide_path, index=False)
    labels_all.to_csv(labels_path, index=False)
    missing_all.to_csv(missing_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80, flush=True)
    print(f"CHECKPOINT COMPLETE: {tag}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"long   : {long_all.shape} -> {long_path}", flush=True)
    print(f"wide   : {wide_all.shape} -> {wide_path}", flush=True)
    print(f"labels : {labels_all.shape} -> {labels_path}", flush=True)
    print(f"issues : {len(missing_all)} -> {missing_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
