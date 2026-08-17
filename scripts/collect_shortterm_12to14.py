from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from _common import inclusive_dates, require_api_key
from gk2a_weather.config import load_yaml, resolve_project_path
from gk2a_weather.data.asos import fetch_asos, parse_asos_response
from gk2a_weather.data.gk2a import api_timestamp, canonical_gk2a_path, download_gk2a_channel
from gk2a_weather.data.http import DownloadError
from gk2a_weather.data.stations import load_station_list
from gk2a_weather.utils.io import atomic_write_csv, atomic_write_text


def time_grid(day: pd.Timestamp, start_hhmm: str, end_hhmm: str, step_minutes: int):
    sh, sm = map(int, start_hhmm.split(":"))
    eh, em = map(int, end_hhmm.split(":"))
    current = day.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = day.replace(hour=eh, minute=em, second=0, microsecond=0)
    if step_minutes <= 0 or current > end:
        raise ValueError("invalid time grid")
    while current <= end:
        yield current.to_pydatetime()
        current += pd.Timedelta(minutes=step_minutes)


def main() -> None:
    p = argparse.ArgumentParser(description="Collect 12:00-14:00 KST GK-2A every 10 min + 14:00 ASOS labels")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--station-list", default="")
    p.add_argument("--output-root", required=True)
    p.add_argument("--start-time", default="12:00")
    p.add_argument("--end-time", default="14:00")
    p.add_argument("--step-minutes", type=int, default=10)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    config = load_yaml(args.config)
    project, satellite, asos_cfg = config["project"], config["satellite"], config["asos"]
    api_key = require_api_key()
    station_path = args.station_list or project["station_list"]
    stations = load_station_list(resolve_project_path(station_path))
    allowed_ids = set(stations["STN_ID"].astype(int))

    root = Path(args.output_root).expanduser().resolve()
    sat_root = root / "raw_gk2a"
    asos_raw_root = root / "asos" / "raw"
    asos_parsed_root = root / "asos" / "parsed"
    output_dir = root / "outputs"
    for d in (sat_root, asos_raw_root, asos_parsed_root, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    failures = []
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required") from exc

    with requests.Session() as session:
        for day in inclusive_dates(args.start, args.end):
            for timestamp_kst in time_grid(day, args.start_time, args.end_time, args.step_minutes):
                requested = api_timestamp(timestamp_kst, str(satellite["api_time_basis"]))
                for channel_value in satellite["channels"]:
                    channel = str(channel_value).upper()
                    output_path = canonical_gk2a_path(sat_root, day, channel, requested)
                    min_bytes = int(satellite["minimum_file_bytes"])
                    if output_path.exists() and output_path.stat().st_size >= min_bytes and not args.force:
                        continue
                    try:
                        download_gk2a_channel(
                            timestamp_kst,
                            channel,
                            api_key=api_key,
                            output_path=output_path,
                            area=str(satellite["area"]),
                            api_time_basis=str(satellite["api_time_basis"]),
                            timeout_seconds=float(satellite["timeout_seconds"]),
                            max_retries=int(satellite["max_retries"]),
                            retry_backoff_seconds=float(satellite["retry_backoff_seconds"]),
                            minimum_file_bytes=min_bytes,
                            session=session,
                        )
                    except (DownloadError, OSError, ValueError) as exc:
                        failures.append({"kind":"GK2A","date_kst":day.strftime("%Y-%m-%d"),"time_kst":timestamp_kst.strftime("%H:%M"),"channel":channel,"error":str(exc)})
                    wait = float(satellite["request_interval_seconds"])
                    if wait > 0:
                        time.sleep(wait)

            label_ts = day.replace(hour=14, minute=0, second=0, microsecond=0).to_pydatetime()
            key = label_ts.strftime("%Y%m%d%H%M")
            raw_path = asos_raw_root / f"asos_{key}.txt"
            parsed_path = asos_parsed_root / f"asos_{key}.csv"
            try:
                if parsed_path.exists() and not args.force:
                    frame = pd.read_csv(parsed_path)
                elif raw_path.exists() and not args.force:
                    frame = parse_asos_response(raw_path.read_text(encoding="utf-8", errors="replace"))
                else:
                    text, frame = fetch_asos(
                        label_ts,
                        api_key=api_key,
                        timeout_seconds=float(asos_cfg["timeout_seconds"]),
                        max_retries=int(asos_cfg["max_retries"]),
                        retry_backoff_seconds=float(asos_cfg["retry_backoff_seconds"]),
                        session=session,
                    )
                    atomic_write_text(raw_path, text)
                frame = frame[frame["STN_ID"].isin(allowed_ids)].copy()
                atomic_write_csv(frame, parsed_path)
            except Exception as exc:
                failures.append({"kind":"ASOS","date_kst":day.strftime("%Y-%m-%d"),"time_kst":"14:00","channel":"TA/HM","error":str(exc)})

    failure_path = output_dir / "shortterm_collection_failures.csv"
    atomic_write_csv(pd.DataFrame(failures), failure_path)
    print(f"done; failures={len(failures)} -> {failure_path}")


if __name__ == "__main__":
    main()
