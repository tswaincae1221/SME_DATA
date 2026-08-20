from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

import incremental_shortterm_pipeline as pipeline


def args_for_one_slot(data_root: Path) -> Namespace:
    return Namespace(
        years=[2019],
        start_mmdd="08-24",
        end_mmdd="08-24",
        start_time="12:00",
        end_time="12:00",
        step_minutes=10,
    )


def test_expected_inventory_is_complete_and_uses_utc(tmp_path: Path) -> None:
    args = args_for_one_slot(tmp_path)
    args.end_time = "12:10"

    inventory = pipeline.expected_inventory(args, tmp_path / "raw_gk2a")

    assert len(inventory) == 2 * len(pipeline.CHANNELS)
    assert set(inventory["TimeKST"]) == {1200, 1210}
    assert set(inventory["TimestampUTC"]) == {"201908240300", "201908240310"}
    assert inventory.groupby(["Date", "TimeKST"])["Channel"].nunique().eq(16).all()


def test_missing_values_are_filled_after_nc_and_cache_arrive(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw_root = data_root / "raw_gk2a"
    cache_root = tmp_path / "output" / "feature_cache"
    stations = pd.DataFrame(
        {
            "STN_ID": [90, 93],
            "LAT": [38.25, 37.57],
            "LON": [128.56, 126.97],
            "ALT": [17.5, 85.7],
        }
    )
    label_dir = data_root / "asos" / "parsed"
    label_dir.mkdir(parents=True)
    pd.DataFrame(
        {"STN_ID": [90, 93], "TA": [26.1, 27.2], "HM": [71.0, 65.0]}
    ).to_csv(label_dir / "asos_201908241400.csv", index=False)

    expected = pipeline.expected_inventory(args_for_one_slot(data_root), raw_root)
    missing_inventory, _ = pipeline.attach_sources(expected, raw_root, 1)
    first_paths = pipeline.build_tables(
        missing_inventory,
        data_root,
        cache_root,
        stations,
        tmp_path / "first",
        valid_cache_keys=set(),
    )
    first = pd.read_csv(first_paths["long"])
    assert first["VI004"].isna().all()

    source = raw_root / "2019" / "08" / "24" / "VI004_201908240300.nc"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"valid nc placeholder")
    pipeline.atomic_npz(
        pipeline.cache_path(cache_root, "VI004", "201908240300"),
        stn_id=np.array([90, 93]),
        values=np.array([101.5, 202.5]),
        source_size=np.array([source.stat().st_size], dtype=np.int64),
        source_mtime_ns=np.array([source.stat().st_mtime_ns], dtype=np.int64),
    )

    refreshed_inventory, _ = pipeline.attach_sources(expected, raw_root, 1)
    second_paths = pipeline.build_tables(
        refreshed_inventory,
        data_root,
        cache_root,
        stations,
        tmp_path / "second",
        valid_cache_keys={("VI004", "201908240300")},
    )
    second = pd.read_csv(second_paths["long"]).sort_values("STN_ID")

    assert second["VI004"].tolist() == [101.5, 202.5]
    assert second["available_channel_count"].tolist() == [1, 1]
    assert second[list(pipeline.CHANNELS[1:])].isna().all().all()


def test_retry_requests_only_missing_rows(tmp_path: Path, monkeypatch) -> None:
    inventory = pd.DataFrame(
        [
            {"Year": 2019, "Date": 20190824, "TimeKST": 1200, "TimestampUTC": "201908240300", "Channel": "VI004", "ExpectedPath": str(tmp_path / "have.nc"), "Status": "available"},
            {"Year": 2019, "Date": 20190824, "TimeKST": 1200, "TimestampUTC": "201908240300", "Channel": "VI005", "ExpectedPath": str(tmp_path / "missing1.nc"), "Status": "missing"},
            {"Year": 2019, "Date": 20190824, "TimeKST": 1200, "TimestampUTC": "201908240300", "Channel": "VI006", "ExpectedPath": str(tmp_path / "missing2.nc"), "Status": "missing"},
        ]
    )
    calls: list[str] = []

    def fake_download_one(**kwargs):
        calls.append(kwargs["channel"])
        return True, ""

    monkeypatch.setattr(pipeline, "download_one", fake_download_one)
    result = pipeline.retry_missing(
        inventory,
        api_key="secret",
        max_downloads=1,
        request_interval=0,
        timeout_seconds=1,
        max_retries=1,
        minimum_file_bytes=1,
        output_dir=tmp_path / "output",
    )

    assert calls == ["VI005"]
    assert result["Channel"].tolist() == ["VI005"]
    assert (tmp_path / "output" / "latest_retry_results.csv").exists()
    assert "/LE1B/" in pipeline.GK2A_URL

