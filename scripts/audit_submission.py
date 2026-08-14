"""최종 Kaggle 추론 노트북과 연결 Dataset의 실격 위험을 정적으로 검사한다."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


PROHIBITED_MARKERS = {
    "평가기간 ASOS API": ("kma_sfctm2.php", "gk2a_weather.data.asos", "collect_labels"),
    "Kaggle Secrets": ("from kaggle_secrets", "import kaggle_secrets", "UserSecretsClient("),
    "추론 중 학습": ("train_with_group_cv", ".fit("),
    "금지 외부 자료": ("era5", "openstreetmap", "osmnx"),
}
PLACEHOLDER_MARKERS = ("REPLACE", "YOUR_", "본인의_", "여기에", "PLACEHOLDER")


def _notebook_code(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return "\n\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
    return path.read_text(encoding="utf-8")


def _literal_assignment(code: str, name: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"노트북 코드를 파싱할 수 없습니다: {exc}") from exc
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
            return None
    return None


def audit(notebook_path: Path, dataset_dir: Path, allow_placeholder: bool) -> list[str]:
    errors: list[str] = []
    code = _notebook_code(notebook_path)
    lowered = code.lower()

    for label, markers in PROHIBITED_MARKERS.items():
        found = [marker for marker in markers if marker.lower() in lowered]
        if found:
            errors.append(f"{label} 금지 문자열: {', '.join(found)}")
    for line_number, line in enumerate(code.splitlines(), start=1):
        lowered_line = line.lower()
        if "api/gk2a/" in lowered_line and "api/gk2a/le1b/" not in lowered_line:
            errors.append(f"노트북 {line_number}행에 /LE1B/ 이외 GK-2A 경로")

    for name in ("PRED_START", "PRED_END", "API_KEY"):
        value = _literal_assignment(code, name)
        if value is None:
            errors.append(f"{name}은 코드 최상위 문자열 리터럴로 직접 지정해야 합니다.")
        elif name.startswith("PRED_") and not re.fullmatch(r"\d{8}", value):
            errors.append(f"{name}은 YYYYMMDD 8자리여야 합니다: {value!r}")
        elif name == "API_KEY" and not allow_placeholder and any(
            marker.lower() in value.lower() for marker in PLACEHOLDER_MARKERS
        ):
            errors.append("API_KEY 자리표시자를 최종 제출용 실제 키로 교체하지 않았습니다.")

    if "pd.date_range(PRED_START, PRED_END" not in code.replace("freq =", "freq="):
        errors.append("예측 날짜가 PRED_START와 PRED_END에서 동적으로 생성되지 않습니다.")
    if "pred = run_kaggle_inference" not in code:
        errors.append("최종 결과 변수 pred 생성 코드를 찾지 못했습니다.")

    required_files = {
        "baseline.joblib",
        "station_list.csv",
        "data.yaml",
        "requirements-inference.txt",
        "gk2a_weather/kaggle.py",
        "gk2a_weather/data/gk2a.py",
        "gk2a_weather/models/bundle.py",
        "gk2a_weather/models/predict.py",
    }
    actual_files = {
        str(path.relative_to(dataset_dir)).replace("\\", "/")
        for path in dataset_dir.rglob("*")
        if path.is_file()
    }
    for missing in sorted(required_files - actual_files):
        errors.append(f"Dataset 필수 파일 누락: {missing}")

    forbidden_paths = (
        "gk2a_weather/data/asos.py",
        "gk2a_weather/models/train.py",
        "scripts/collect_labels.py",
        "data/raw",
        "data/interim",
    )
    for path in forbidden_paths:
        if any(item == path or item.startswith(path + "/") for item in actual_files):
            errors.append(f"추론 Dataset에 포함하면 안 되는 경로: {path}")

    for path in dataset_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".txt", ".yaml", ".yml", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for label, markers in PROHIBITED_MARKERS.items():
            found = [marker for marker in markers if marker.lower() in text]
            if found:
                errors.append(
                    f"Dataset {path.relative_to(dataset_dir)}에 {label} 문자열: {', '.join(found)}"
                )
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "api/gk2a/" in line and "api/gk2a/le1b/" not in line:
                errors.append(
                    f"Dataset {path.relative_to(dataset_dir)}:{line_number}에 "
                    "/LE1B/ 이외 GK-2A 경로"
                )
    gk2a_path = dataset_dir / "gk2a_weather/data/gk2a.py"
    if gk2a_path.is_file() and "/GK2A/LE1B/" not in gk2a_path.read_text(
        encoding="utf-8", errors="ignore"
    ):
        errors.append("Dataset GK-2A 수집 코드의 API 경로에 /LE1B/가 없습니다.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle 제출물 규정 준수 여부를 검사합니다.")
    parser.add_argument("--notebook", required=True, help="최종 .ipynb 또는 템플릿 .py")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="GitHub 템플릿 검사 때만 API 키 자리표시자를 허용합니다.",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    dataset_dir = Path(args.dataset_dir)
    if not notebook_path.exists():
        raise SystemExit(f"노트북 파일이 없습니다: {notebook_path}")
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset 폴더가 없습니다: {dataset_dir}")

    errors = audit(notebook_path, dataset_dir, args.allow_placeholder)
    if errors:
        print("[FAIL] 제출 규정 점검에서 문제가 발견됐습니다.")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("[PASS] 정적 점검을 통과했습니다.")
    print("이후 비공개 Copy & Edit → 날짜만 변경 → Run All 재현까지 직접 확인하세요.")


if __name__ == "__main__":
    main()
