"""대회 공식 점수와 안전한 RMSE 계산."""

from __future__ import annotations

import numpy as np


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(prediction)
    if not mask.any():
        raise ValueError("RMSE를 계산할 유효한 정답·예측 쌍이 없습니다.")
    return float(np.sqrt(np.mean((truth[mask] - prediction[mask]) ** 2)))


def competition_score(
    y_ta: np.ndarray,
    pred_ta: np.ndarray,
    y_hm: np.ndarray,
    pred_hm: np.ndarray,
) -> dict[str, float]:
    rmse_ta = masked_rmse(y_ta, pred_ta)
    rmse_hm = masked_rmse(y_hm, pred_hm)
    return {
        "rmse_ta": rmse_ta,
        "rmse_hm": rmse_hm,
        "score": rmse_ta + 0.1 * rmse_hm,
    }

