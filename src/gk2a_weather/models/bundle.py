"""학습 결과와 추론 메타데이터를 함께 보관하는 직렬화 형식.

이 모듈은 학습 로직과 분리되어 있어 제출용 추론 패키지에 학습 코드를
포함하지 않고도 joblib 모델을 불러올 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.pipeline import Pipeline


@dataclass
class ModelBundle:
    ta_model: Pipeline
    hm_model: Pipeline
    feature_columns: list[str]
    categorical_columns: list[str]
    satellite_columns: list[str]
    metrics: dict[str, Any]
