from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import shortterm_multiyear_phased as phased


def build_args(tmp_path: Path, *, allow_incomplete: bool) -> SimpleNamespace:
    return SimpleNamespace(
        years=[2025],
        output_root=str(tmp_path),
        allow_incomplete=allow_incomplete,
        resume_build=False,
        start_mmdd="08-24",
        end_mmdd="08-30",
        start_time="12:00",
        end_time="14:00",
        step_minutes=10,
        config="configs/data.yaml",
        station_list="",
    )


def incomplete_status() -> dict:
    return {
        "year": 2025,
        "valid_nc": 1386,
        "expected_nc": 1456,
        "valid_asos": 7,
        "expected_asos": 7,
        "complete": False,
        "missing_examples": ["/raw/missing.nc"],
    }


def test_incomplete_build_requires_explicit_opt_in(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(phased, "expected_year_status", lambda *args: incomplete_status())

    with pytest.raises(RuntimeError, match="--allow-incomplete"):
        phased.build_phase(build_args(tmp_path, allow_incomplete=False), {}, tmp_path)

    status = pd.read_csv(tmp_path / "outputs" / "build_preflight_status.csv")
    assert status.loc[0, "valid_nc"] == 1386
    assert not bool(status.loc[0, "complete"])


def test_incomplete_build_continues_and_combines(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(phased, "expected_year_status", lambda *args: incomplete_status())
    monkeypatch.setattr(phased, "run", lambda cmd, cwd: calls.append(cmd))
    monkeypatch.setattr(phased, "combine_years", lambda args: calls.append(["combine"]))

    phased.build_phase(build_args(tmp_path, allow_incomplete=True), {}, tmp_path)

    assert calls[0][2] == "scripts/build_shortterm_12to14_dataset.py"
    assert calls[-1] == ["combine"]


def test_resume_rebuilds_stale_missing_output(tmp_path) -> None:
    for name in (
        "shortterm_long.csv",
        "shortterm_wide.csv",
        "shortterm_labels_1400.csv",
    ):
        (tmp_path / name).write_text("value\n1\n", encoding="utf-8")

    (tmp_path / "shortterm_build_missing.csv").write_text("\n", encoding="utf-8")
    assert phased.yearly_build_is_reusable(tmp_path, source_complete=True)

    (tmp_path / "shortterm_build_missing.csv").write_text(
        "reason\nold raw gap\n", encoding="utf-8"
    )
    assert not phased.yearly_build_is_reusable(tmp_path, source_complete=True)
    assert phased.yearly_build_is_reusable(tmp_path, source_complete=False)
