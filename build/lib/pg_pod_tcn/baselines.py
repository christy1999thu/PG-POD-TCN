from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pg_pod_tcn.data import PreparedCase


def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    last = np.asarray(history)[..., -1, :]
    return np.repeat(last[..., None, :], horizon, axis=-2)


@dataclass
class DMDModel:
    ridge: float = 1e-5
    transition_: np.ndarray | None = None

    def fit(self, cases: Sequence[PreparedCase]) -> DMDModel:
        previous = np.concatenate([case.coeff_scaled[:-1] for case in cases], axis=0)
        following = np.concatenate([case.coeff_scaled[1:] for case in cases], axis=0)
        gram = previous.T @ previous + self.ridge * np.eye(previous.shape[1])
        self.transition_ = np.linalg.solve(gram, previous.T @ following).astype(np.float32)
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self.transition_ is None:
            raise RuntimeError("DMDModel has not been fitted")
        state = np.asarray(history, dtype=np.float32)[..., -1, :]
        predictions = []
        for _ in range(horizon):
            state = state @ self.transition_
            predictions.append(state)
        return np.stack(predictions, axis=-2)
