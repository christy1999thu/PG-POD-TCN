from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PODReducer:
    energy: float = 0.99
    max_modes: int | None = None
    mean_: np.ndarray | None = None
    basis_: np.ndarray | None = None
    singular_values_: np.ndarray | None = None
    captured_energy_: float | None = None

    def fit(self, snapshots: np.ndarray) -> PODReducer:
        matrix = np.asarray(snapshots, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            raise ValueError("snapshots must have shape [samples, features] with at least two samples")
        if not 0 < self.energy <= 1:
            raise ValueError("energy must lie in (0, 1]")
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        power = singular_values**2
        cumulative = np.cumsum(power) / max(power.sum(), np.finfo(float).eps)
        modes = int(np.searchsorted(cumulative, self.energy) + 1)
        if self.max_modes is not None:
            modes = min(modes, int(self.max_modes))
        self.basis_ = vt[:modes].astype(np.float32)
        self.singular_values_ = singular_values[:modes].astype(np.float64)
        self.captured_energy_ = float(cumulative[modes - 1])
        return self

    @property
    def n_modes(self) -> int:
        self._check_fitted()
        return int(self.basis_.shape[0])

    def transform(self, snapshots: np.ndarray) -> np.ndarray:
        self._check_fitted()
        matrix = np.asarray(snapshots, dtype=np.float32)
        return ((matrix - self.mean_) @ self.basis_.T).astype(np.float32)

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        self._check_fitted()
        coeff = np.asarray(coefficients, dtype=np.float32)
        return (coeff @ self.basis_ + self.mean_).astype(np.float32)

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.basis_ is None:
            raise RuntimeError("PODReducer has not been fitted")
