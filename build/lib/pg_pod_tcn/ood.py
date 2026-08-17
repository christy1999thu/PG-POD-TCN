from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pg_pod_tcn.data import WindowDataset


@dataclass
class MahalanobisDetector:
    regularization: float = 1e-3
    quantile: float = 0.99
    mean_: np.ndarray | None = None
    precision_: np.ndarray | None = None
    threshold_: float | None = None

    def fit(self, features: np.ndarray) -> MahalanobisDetector:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            raise ValueError("features must be a 2-D matrix with at least two rows")
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        covariance = centered.T @ centered / max(len(matrix) - 1, 1)
        scale = float(np.trace(covariance) / max(covariance.shape[0], 1))
        covariance += np.eye(covariance.shape[0]) * self.regularization * max(scale, 1e-8)
        self.precision_ = np.linalg.pinv(covariance)
        distances = self.score(matrix)
        self.threshold_ = float(np.quantile(distances, self.quantile))
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted(allow_missing_threshold=True)
        centered = np.asarray(features, dtype=np.float64) - self.mean_
        squared = np.einsum("bi,ij,bj->b", centered, self.precision_, centered)
        return np.sqrt(np.maximum(squared, 0.0))

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.score(features) > float(self.threshold_)

    def calibrate(self, validation_features: np.ndarray) -> MahalanobisDetector:
        """Relax the train-derived threshold using known in-distribution validation cases."""
        self._check_fitted()
        validation_scores = self.score(validation_features)
        validation_threshold = float(np.quantile(validation_scores, self.quantile))
        self.threshold_ = max(float(self.threshold_), validation_threshold)
        return self

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            regularization=np.asarray(self.regularization),
            quantile=np.asarray(self.quantile),
            mean=self.mean_,
            precision=self.precision_,
            threshold=np.asarray(self.threshold_),
        )

    @classmethod
    def load(cls, path: str | Path) -> MahalanobisDetector:
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                regularization=float(payload["regularization"].item()),
                quantile=float(payload["quantile"].item()),
                mean_=payload["mean"].astype(np.float64),
                precision_=payload["precision"].astype(np.float64),
                threshold_=float(payload["threshold"].item()),
            )

    def _check_fitted(self, allow_missing_threshold: bool = False) -> None:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisDetector has not been fitted")
        if not allow_missing_threshold and self.threshold_ is None:
            raise RuntimeError("MahalanobisDetector threshold is unavailable")


def dataset_features(dataset: WindowDataset) -> np.ndarray:
    rows = []
    for case_index, start in dataset.indices:
        case = dataset.cases[case_index]
        history = case.coeff_scaled[start : start + dataset.history]
        rows.append(np.concatenate((case.params_scaled, history.mean(axis=0))))
    return np.asarray(rows, dtype=np.float64)
