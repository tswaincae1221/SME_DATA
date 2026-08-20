from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import inclusive_dates
from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.data.gk2a import api_timestamp, find_gk2a_file
from gk2a_weather.data.stations import load_station_list
from gk2a_weather.features.satellite import extract_channel_features, load_satellite_array
from gk2a_weather.utils.io import atomic_write_csv


def time_grid(day: pd.Timestamp, start_hhmm: str, end_hhmm: str, step_minutes: int):
    sh, sm = map(int, start_hhmm.split(":"))
    eh, em = map(int, end_hhmm.split(":"))
    current = day.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = day.replace(hour=eh, minute=em, second=0, microsecond=0)
    while current <= end:
        yield current.to_pydatetime()
        current += pd.Timedelta(minutes=step_minutes)


def label_path(root: Path, day: pd.Timestamp) -> Path:
    key = day.replace(hour=14, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
    return root / "asos" / "parsed" / f"asos_{key}.csv"


def load_label(root: Path, day: pd.Timestamp) -> pd.DataFrame:
    path = label_path(root, day)
    if not path.exists():
        return pd.DataFrame(columns=["STN_ID", "TA", "HM"])
    frame = pd.read_csv(path)
    return frame[["STN_ID", "TA", "HM"]].drop_duplicates("STN_ID", keep="last")


def validate_source_root(
    *,
    root: Path,
    raw_root: Path,
    start: str,
    end: str,
    start_time: str,
    end_time: str,
    step_minutes: int,
    channels: list[str],
    time_basis: str,
    minimum_source_fraction: float,
) -> dict[str, int]:
    expected_nc = 0
    found_nc = 0
    expected_labels = 0
    found_labels = 0
    missing_examples: list[str] = []

    for day in inclusive_dates(start, end):
        expected_labels += 1
        if label_path(root, day).exists():
            found_labels += 1
        for ts in time_grid(day, start_time, end_time, step_minutes):
            requested = api_timestamp(ts, time_basis)
            for channel in channels:
                expected_nc += 1
                path = find_gk2a_file(raw_root, day, channel, requested)
                if path is not None:
                    found_nc += 1
                elif len(missing_examples) < 5:
                    missing_examples.append(
                        str(raw_root / day.strftime("%Y/%m/%d") / f"{channel}_{requested}.nc")
                    )

    print(
        f"[SOURCE PREFLIGHT] GK2A={found_nc}/{expected_nc}, "
        f"ASOS={found_labels}/{expected_labels}, root={root}",
        flush=True,
    )
    minimum_nc = max(1, int(expected_nc * minimum_source_fraction))
    if found_nc < minimum_nc or found_labels == 0:
        details = "\n".join(f"  - {path}" for path in missing_examples)
        raise RuntimeError(
            "입력 폴더에 원본 데이터가 거의 없거나 ASOS 라벨이 없습니다. "
            f"GK2A={found_nc}/{expected_nc}, ASOS={found_labels}/{expected_labels}\n"
            f"root={root}\n"
            "Colab에 데이터를 보유한 Google 계정을 마운트했는지와 "
            "OUTPUT_ROOT 경로를 확인하세요.\n"
            + ("missing examples:\n" + details if details else "")
        )
    return {
        "expected_nc": expected_nc,
        "found_nc": found_nc,
        "expected_labels": expected_labels,
        "found_labels": found_labels,
    }


def print_issue_summary(missing_df: pd.DataFrame) -> None:
    if not len(missing_df):
        return
    print("[BUILD ISSUE SUMMARY]", flush=True)
    counts = missing_df.groupby("reason", dropna=False).size().sort_values(ascending=False)
    for reason, count in counts.head(10).items():
        print(f"  {count:4d} | {reason}", flush=True)
    print("[BUILD ISSUE EXAMPLES]", flush=True)
    print(missing_df.head(10).to_string(index=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build merged short-term GK-2A + 14:00 TA/HM datasets")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--station-list", default="")
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--start-time", default="12:00")
    p.add_argument("--end-time", default="14:00")
    p.add_argument("--step-minutes", type=int, default=10)
    p.add_argument("--minimum-source-fraction", type=float, default=0.5)
    args = p.parse_args()

    if not 0.0 < args.minimum_source_fraction <= 1.0:
        raise ValueError("--minimum-source-fraction은 0보다 크고 1 이하여야 합니다.")

    config = load_yaml(args.config)
    project, satellite = config["project"], config["satellite"]
    station_path = args.station_list or project["station_list"]
    stations = load_station_list(resolve_project_path(station_path)).rename(
        columns={"latitude":"LAT","longitude":"LON","altitude":"ALT"}
    )
    station_extract = stations.rename(columns={"LAT":"latitude","LON":"longitude","ALT":"altitude"})

    root = Path(args.input_root).expanduser().resolve()
    raw_root = root / "raw_gk2a"
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = [str(c).upper() for c in satellite["channels"]]
    long_parts, labels_parts, missing = [], [], []
    preflight = validate_source_root(
        root=root,
        raw_root=raw_root,
        start=args.start,
        end=args.end,
        start_time=args.start_time,
        end_time=args.end_time,
        step_minutes=args.step_minutes,
        channels=channels,
        time_basis=str(satellite["api_time_basis"]),
        minimum_source_fraction=args.minimum_source_fraction,
    )

    for day in inclusive_dates(args.start, args.end):
        label = load_label(root, day)
        if len(label):
            tmp = label.copy()
            tmp.insert(0, "Date", int(day.strftime("%Y%m%d")))
            labels_parts.append(tmp)

        for ts in time_grid(day, args.start_time, args.end_time, args.step_minutes):
            hhmm = int(ts.strftime("%H%M"))
            requested = api_timestamp(ts, str(satellite["api_time_basis"]))
            frame = stations.copy()
            frame.insert(0, "TimeKST", hhmm)
            frame.insert(0, "Date", int(day.strftime("%Y%m%d")))

            for channel in channels:
                raw_path = find_gk2a_file(raw_root, day, channel, requested)
                if raw_path is None:
                    frame[channel] = np.nan
                    missing.append({"Date":day.strftime("%Y-%m-%d"),"TimeKST":ts.strftime("%H:%M"),"channel":channel,"reason":"raw file missing"})
                    continue
                try:
                    array = load_satellite_array(raw_path, channel)
                    feat = extract_channel_features(array, station_extract, channel=channel, radii_km=(0.0,))[["STN_ID", f"{channel}_center_mean"]]
                    feat = feat.rename(columns={f"{channel}_center_mean": channel})
                    frame = frame.merge(feat, on="STN_ID", how="left", validate="1:1")
                except Exception as exc:
                    frame[channel] = np.nan
                    missing.append({"Date":day.strftime("%Y-%m-%d"),"TimeKST":ts.strftime("%H:%M"),"channel":channel,"reason":str(exc)})

            frame["TA"] = np.nan
            frame["HM"] = np.nan
            if hhmm == 1400 and len(label):
                frame = frame.drop(columns=["TA", "HM"]).merge(label, on="STN_ID", how="left", validate="1:1")
            long_parts.append(frame)

    if not long_parts:
        raise RuntimeError("no data built")

    long_df = pd.concat(long_parts, ignore_index=True).sort_values(["Date","STN_ID","TimeKST"]).reset_index(drop=True)
    labels_df = pd.concat(labels_parts, ignore_index=True) if labels_parts else pd.DataFrame(columns=["Date","STN_ID","TA","HM"])
    labels_df = labels_df.sort_values(["Date","STN_ID"]).reset_index(drop=True)

    pivot = long_df.pivot(index=["Date","STN_ID"], columns="TimeKST", values=channels)
    pivot.columns = [f"{channel}_{int(hhmm):04d}" for channel, hhmm in pivot.columns]
    wide_df = pivot.reset_index().merge(stations, on="STN_ID", how="left", validate="m:1")
    wide_df = wide_df.merge(labels_df, on=["Date","STN_ID"], how="left", validate="1:1")

    missing_df = pd.DataFrame(
        missing, columns=["Date", "TimeKST", "channel", "reason"]
    )
    print_issue_summary(missing_df)
    if len(missing_df) >= max(1, preflight["expected_nc"] // 2):
        failed_path = out_dir / "shortterm_build_failed_issues.csv"
        atomic_write_csv(missing_df, failed_path)
        raise RuntimeError(
            "절반 이상의 위성 파일이 읽기/특징 추출에 실패했습니다. "
            "기존 정상 결과를 덮어쓰지 않고 중단합니다. "
            f"진단 파일: {failed_path}"
        )

    atomic_write_csv(long_df, out_dir / "shortterm_long.csv")
    atomic_write_csv(wide_df, out_dir / "shortterm_wide.csv")
    atomic_write_csv(labels_df, out_dir / "shortterm_labels_1400.csv")
    atomic_write_csv(missing_df, out_dir / "shortterm_build_missing.csv")
    print(f"long={long_df.shape}, wide={wide_df.shape}, labels={labels_df.shape}, issues={len(missing)}")


if __name__ == "__main__":
    main()
