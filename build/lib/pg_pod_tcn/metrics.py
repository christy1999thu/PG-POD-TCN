from __future__ import annotations

import numpy as np


def nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rmse = np.sqrt(np.mean((prediction - target) ** 2))
    scale = np.sqrt(np.mean(target**2))
    return float(rmse / max(scale, 1e-12))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def mape(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs((prediction - target) / np.maximum(np.abs(target), epsilon))))


def global_ssim(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Global SSIM over flattened snapshots; useful but not windowed image SSIM."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, prediction.shape[-1])
    target = np.asarray(target, dtype=np.float64).reshape(-1, target.shape[-1])
    mu_x = prediction.mean(axis=1)
    mu_y = target.mean(axis=1)
    var_x = prediction.var(axis=1)
    var_y = target.var(axis=1)
    covariance = ((prediction - mu_x[:, None]) * (target - mu_y[:, None])).mean(axis=1)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(np.mean(score))


def mass_drift_percent(
    fields: np.ndarray,
    cell_volumes: np.ndarray,
    solid_volume: np.ndarray,
) -> float:
    predicted = np.sum(fields * cell_volumes[:, None, :], axis=-1)
    relative = np.abs(predicted - solid_volume) / np.maximum(np.abs(solid_volume), 1e-12)
    return float(100.0 * np.mean(relative))

