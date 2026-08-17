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


def load_label(root: Path, day: pd.Timestamp) -> pd.DataFrame:
    key = day.replace(hour=14, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
    path = root / "asos" / "parsed" / f"asos_{key}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["STN_ID", "TA", "HM"])
    frame = pd.read_csv(path)
    return frame[["STN_ID", "TA", "HM"]].drop_duplicates("STN_ID", keep="last")


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
    args = p.parse_args()

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

    atomic_write_csv(long_df, out_dir / "shortterm_long.csv")
    atomic_write_csv(wide_df, out_dir / "shortterm_wide.csv")
    atomic_write_csv(labels_df, out_dir / "shortterm_labels_1400.csv")
    atomic_write_csv(pd.DataFrame(missing), out_dir / "shortterm_build_missing.csv")
    print(f"long={long_df.shape}, wide={wide_df.shape}, labels={labels_df.shape}, issues={len(missing)}")


if __name__ == "__main__":
    main()
