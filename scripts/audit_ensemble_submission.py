#!/usr/bin/env python3
"""Audit the filled official notebook and minimal ensemble model package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_PACKAGE_FILES = {
    "feature_spec.json",
    "ensemble_component_weights.csv",
    "lstm_model.pt",
    "lstm_normalization.npz",
    "catboost_TA.cbm",
    "catboost_HM.cbm",
    "submission_predictor.py",
    "manifest.sha256.json",
}
PROHIBITED_ENDPOINTS = {
    "kma_sfctm2.php",
    "/api/typ01/",
    "openstreetmap",
    "osmnx",
    "era5",
}
SATELLITE_CHANNELS = {
    "VI004", "VI005", "VI006", "VI008", "NR013", "NR016", "SW038",
    "WV063", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112",
    "IR123", "IR133",
}
ALLOWED_FEATURES = SATELLITE_CHANNELS | {
    "LAT", "LON", "ALT", "month", "day", "dayofyear",
    "doy_sin", "doy_cos",
}


def code_text(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--package-dir", required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    notebook = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    package = Path(args.package_dir)
    errors: list[str] = []

    if len(template.get("cells", [])) != len(notebook.get("cells", [])):
        errors.append("official notebook cell count changed")
    else:
        for index, (before, after) in enumerate(zip(template["cells"], notebook["cells"])):
            if index == 4:
                continue
            if before.get("cell_type") != after.get("cell_type") or before.get("source") != after.get("source"):
                errors.append(f"official notebook cell {index} changed outside the free cell")

    code = code_text(notebook)
    if "pred = _predictor.predict_from_api(" not in code:
        errors.append("free cell does not assign the inference result to pred")
    if "pred_dates=PRED_DATES" not in code:
        errors.append("predictor does not use dynamic PRED_DATES")
    if "to_api_datetime=to_api_datetime" not in code:
        errors.append("official KST-to-UTC converter is not passed to the predictor")
    free_code = "".join(notebook["cells"][4].get("source", []))
    if re.search(r"20\d{6}", free_code):
        errors.append("an 8-digit date is hardcoded in the free cell")
    if any(marker in code.lower() for marker in (".fit(", "optimizer", "backward(")):
        errors.append("training operation found in the submission notebook")

    missing_files = REQUIRED_PACKAGE_FILES - {path.name for path in package.iterdir() if path.is_file()}
    if missing_files:
        errors.append(f"package missing files: {sorted(missing_files)}")
    predictor_path = package / "submission_predictor.py"
    predictor = predictor_path.read_text(encoding="utf-8") if predictor_path.exists() else ""
    combined = (code + "\n" + predictor).lower()
    for marker in PROHIBITED_ENDPOINTS:
        if marker.lower() in combined:
            errors.append(f"prohibited endpoint/source marker found: {marker}")
    for line_number, line in enumerate(predictor.splitlines(), start=1):
        lowered = line.lower()
        if "apihub.kma.go.kr" in lowered and "/gk2a/le1b/" not in lowered:
            errors.append(f"predictor line {line_number}: KMA URL outside /GK2A/LE1B/")
    if "/GK2A/LE1B/" not in predictor or "/KO/data" not in predictor:
        errors.append("predictor does not contain the required LE1B/KO endpoint")

    spec_path = package / "feature_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        feature_map = spec.get("catboost_features_by_target", {})
        for target in ("TA", "HM"):
            features = set(feature_map.get(target, []))
            if not features:
                errors.append(f"missing CatBoost feature list for {target}")
            unexpected = features - ALLOWED_FEATURES
            if unexpected:
                errors.append(f"{target} contains disallowed features: {sorted(unexpected)}")
        if set(spec.get("lstm_channels", [])) != SATELLITE_CHANNELS:
            errors.append("LSTM channel list is not exactly the allowed 16 GK-2A channels")
        allowed_static = {"LAT", "LON", "ALT", "doy_sin", "doy_cos"}
        if set(spec.get("lstm_static_features", [])) != allowed_static:
            errors.append("LSTM static feature list differs from official coordinates/date derivatives")

    if errors:
        raise SystemExit("[FAIL]\n- " + "\n- ".join(errors))
    print("[PASS] official cells, dynamic dates, LE1B/KO path, features, and package contract")


if __name__ == "__main__":
    main()
