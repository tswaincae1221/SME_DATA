"""규정에 맞는 최소 Kaggle 추론 Dataset 폴더와 ZIP을 만든다.

평가 기간 ASOS 조회 코드가 제출물에 섞이지 않도록 허용 목록에 있는 추론
파일만 복사한다. 학습 코드, 라벨 수집기, 원자료, API 키는 포함하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import yaml

from _common import ROOT

from gk2a_weather.data.stations import load_station_list
from gk2a_weather.models.bundle import ModelBundle


INFERENCE_SOURCE_FILES = (
    "__init__.py",
    "constants.py",
    "kaggle.py",
    "data/__init__.py",
    "data/gk2a.py",
    "data/http.py",
    "data/stations.py",
    "features/__init__.py",
    "features/satellite.py",
    "features/static.py",
    "models/__init__.py",
    "models/bundle.py",
    "models/predict.py",
    "utils/__init__.py",
    "utils/io.py",
)

INFERENCE_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "joblib",
    "requests",
    "PyYAML",
    "xarray",
    "h5netcdf",
)

PROHIBITED_TEXT = (
    "kma_sfctm2.php",
    "gk2a_weather.data.asos",
    "collect_labels",
    "kaggle_secrets",
    "UserSecretsClient",
    "era5",
    "openstreetmap",
    "osmnx",
)

PROHIBITED_FEATURE_MARKERS = (
    "ta_lag",
    "hm_lag",
    "recent_ta",
    "recent_hm",
    "station_ta_mean",
    "station_hm_mean",
    "climatology",
    "climate_normal",
    "coast_distance",
    "distance_to_coast",
    "landcover",
    "terrain",
    "population",
    "nightlight",
    "soil_",
    "dem_",
    "osm_",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_safe_output(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {Path("/").resolve(), ROOT.resolve(), Path.cwd().resolve()}
    if resolved in forbidden or len(resolved.parts) < 4:
        raise ValueError(f"출력 폴더가 너무 넓습니다: {resolved}")


def _freeze_inference_requirements() -> str:
    lines = []
    missing = []
    for distribution in INFERENCE_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
            continue
        lines.append(f"{distribution}=={version}")
    if missing:
        raise RuntimeError(
            "추론 의존성을 설치한 학습 환경에서 다시 실행하세요. 누락: "
            + ", ".join(missing)
        )
    return "\n".join(lines) + "\n"


def _scan_text_files(root: Path) -> None:
    violations = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".txt", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in PROHIBITED_TEXT:
            if marker.lower() in text.lower():
                violations.append(f"{path.relative_to(root)}: {marker}")
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if "api/gk2a/" in lowered and "api/gk2a/le1b/" not in lowered:
                violations.append(
                    f"{path.relative_to(root)}:{line_number}: /LE1B/ 이외 GK-2A 경로"
                )
    if violations:
        raise RuntimeError("제출 Dataset 금지 문자열 발견:\n- " + "\n- ".join(violations))


def build_dataset(
    *,
    model_path: Path,
    station_list_path: Path,
    config_path: Path,
    output_dir: Path,
    zip_path: Path,
) -> tuple[Path, Path]:
    for path in (model_path, station_list_path, config_path):
        if not path.exists():
            raise FileNotFoundError(path)

    bundle = joblib.load(model_path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"ModelBundle 형식이 아닙니다: {type(bundle)}")
    prohibited_features = [
        column
        for column in bundle.feature_columns
        if any(marker in column.lower() for marker in PROHIBITED_FEATURE_MARKERS)
    ]
    if prohibited_features:
        raise ValueError(
            "모델에 금지된 ASOS 통계 또는 외부 지리 특징이 "
            f"저장되어 있습니다: {prohibited_features}"
        )

    stations = load_station_list(station_list_path)
    if len(stations) != 96 or stations["STN_ID"].nunique() != 96:
        raise ValueError("공식 station_list.csv는 서로 다른 관측소 96개여야 합니다.")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    satellite_config = config.get("satellite", {})
    if satellite_config.get("api_time_basis") not in {"kst", "utc"}:
        raise ValueError("위성 API 시간 기준을 검증한 뒤 kst 또는 utc로 확정하세요.")

    _assert_safe_output(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="inference_dataset_", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / output_dir.name
        package_target = staging / "gk2a_weather"
        package_source = ROOT / "src/gk2a_weather"

        for relative in INFERENCE_SOURCE_FILES:
            source = package_source / relative
            destination = package_target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        gk2a_source = (package_target / "data/gk2a.py").read_text(encoding="utf-8")
        if "/GK2A/LE1B/" not in gk2a_source:
            raise RuntimeError("추론 GK-2A API 경로에 /LE1B/가 없습니다.")

        shutil.copy2(model_path, staging / "baseline.joblib")
        stations.rename(
            columns={"latitude": "LAT", "longitude": "LON", "altitude": "ALT"}
        ).to_csv(staging / "station_list.csv", index=False)
        (staging / "data.yaml").write_text(
            yaml.safe_dump({"satellite": satellite_config}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (staging / "requirements-inference.txt").write_text(
            _freeze_inference_requirements(), encoding="utf-8"
        )

        _scan_text_files(staging)
        manifest = {
            str(path.relative_to(staging)): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        (staging / "manifest.sha256.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(staging, output_dir)

    archive_base = zip_path.with_suffix("")
    made = Path(shutil.make_archive(str(archive_base), "zip", output_dir.parent, output_dir.name))
    if made != zip_path:
        if zip_path.exists():
            zip_path.unlink()
        made.replace(zip_path)
    return output_dir, zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="최소 Kaggle 추론 Dataset을 만듭니다.")
    parser.add_argument("--model", default="models/baseline.joblib")
    parser.add_argument("--station-list", default="data/metadata/station_list.csv")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output-dir", default="outputs/kaggle_dataset")
    parser.add_argument("--zip", dest="zip_path", default="outputs/kaggle_dataset.zip")
    args = parser.parse_args()

    output_dir, zip_path = build_dataset(
        model_path=(ROOT / args.model).resolve(),
        station_list_path=(ROOT / args.station_list).resolve(),
        config_path=(ROOT / args.config).resolve(),
        output_dir=(ROOT / args.output_dir).resolve(),
        zip_path=(ROOT / args.zip_path).resolve(),
    )
    print(f"Dataset 폴더: {output_dir}")
    print(f"업로드 ZIP: {zip_path}")
    print("ASOS 수집 코드와 API 키가 제외된 최소 추론 Dataset을 생성했습니다.")


if __name__ == "__main__":
    main()
