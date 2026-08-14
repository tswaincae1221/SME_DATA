from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gk2a_weather.drive_preprocess import (
    KST,
    channel_feature_columns,
    group_nc_records,
    parse_nc_filename,
    record_is_in_scope,
    select_group_files,
)


def test_parse_simple_utc_filename_to_14_kst() -> None:
    record = parse_nc_filename("IR105_202307010500.nc", source_timezone="utc")
    assert record is not None
    assert record.channel == "IR105"
    assert record.timestamp_kst == datetime(2023, 7, 1, 14, 0, tzinfo=KST)


def test_parse_original_kma_filename() -> None:
    record = parse_nc_filename(
        "gk2a_ami_le1b_nr016_ko020lc_202208240500.nc",
        source_timezone="utc",
    )
    assert record is not None
    assert record.channel == "NR016"
    assert record.timestamp_utc.strftime("%Y%m%d%H%M") == "202208240500"


def test_scope_is_based_on_kst_date_month_and_hour() -> None:
    record = parse_nc_filename("VI004_202506300500.nc", source_timezone="utc")
    assert record is not None
    assert record_is_in_scope(
        record,
        start_date="2019-06-01",
        end_date="2025-08-31",
        months=[6, 7, 8],
        target_hour_kst=14,
    )
    assert not record_is_in_scope(
        record,
        start_date="2019-06-01",
        end_date="2025-08-31",
        months=[7, 8],
        target_hour_kst=14,
    )


def test_duplicate_selection_prefers_explicit_ko(tmp_path: Path) -> None:
    fd = tmp_path / "gk2a_ami_le1b_ir105_fd020ge_202307010500.nc"
    ko = tmp_path / "gk2a_ami_le1b_ir105_ko020lc_202307010500.nc"
    fd.write_bytes(b"larger-full-disk-file")
    ko.write_bytes(b"ko")
    records = [parse_nc_filename(fd), parse_nc_filename(ko)]
    valid_records = [record for record in records if record is not None]
    groups = group_nc_records(valid_records)
    selected, duplicates = select_group_files(next(iter(groups.values())))
    assert selected["IR105"].path == ko
    assert duplicates["IR105"] == [str(fd)]


def test_channel_schema_includes_center_and_patch_stats() -> None:
    columns = channel_feature_columns("IR105", (0.0, 5.0, 15.0))
    assert "IR105_center_mean" in columns
    assert "IR105_r5km_std" in columns
    assert "IR105_r15km_p90" in columns
    assert "IR105_center_minus_r15km" in columns
    assert "IR105_missing" in columns
