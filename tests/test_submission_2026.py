import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_inference_module_is_inference_only_and_le1b_scoped():
    path = ROOT / "scripts/submission_2026_inference.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert not any(node.func.attr == "fit" for node in calls)
    assert "submission_predictor" in source
    satellite_source = (ROOT / "scripts/submission_predictor.py").read_text(encoding="utf-8")
    assert "/LE1B/" in satellite_source
    assert "/LE2/" not in satellite_source


def test_generated_notebook_preserves_fixed_guide_cells():
    guide_path = ROOT.parent / "upload/submission_notebook_2(1).ipynb"
    if not guide_path.is_file():
        pytest.skip("local organiser guide attachment is not present in the checkout")
    guide = json.loads(
        guide_path.read_text(encoding="utf-8")
    )
    generated = json.loads(
        (ROOT / "notebooks/submission_2026_end_to_end.ipynb").read_text(encoding="utf-8")
    )
    assert generated["cells"][2]["source"] == guide["cells"][2]["source"]
    assert generated["cells"][6]["source"] == guide["cells"][6]["source"]
    free_cell = "".join(generated["cells"][4]["source"])
    assert "PRED_DATES" in free_cell
    assert "predict_from_api" in free_cell
    assert "20260624" not in free_cell
    assert "20260630" not in free_cell
    assert ".fit(" not in free_cell


def test_bundle_recipe_has_96_station_submission_contract():
    source = (ROOT / "scripts/build_submission_2026_bundle.py").read_text(encoding="utf-8")
    assert "asos_inference_inputs\": False" in source
    assert "direct_year_feature\": False" in source
    assert "sequence_supported_mmdd" in source
    assert "june_recipe" in source


def test_june_inference_requests_only_target_time():
    source = (ROOT / "scripts/submission_2026_inference.py").read_text(encoding="utf-8")
    assert "_collect_14h_api" in source
    assert 'frame.insert(0, "TimeKST", 1400)' in source
    assert "to_api_datetime(target_dt, target_dt)" in source
