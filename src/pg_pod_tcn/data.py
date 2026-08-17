from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CaseData:
    """One complete CFD-DEM operating condition.

    `solid_fraction` has shape [time, *grid]. `cell_volumes` has shape [*grid]
    and must be zero outside the physical domain. Macroscopic targets are aligned
    with the field snapshots.
    """

    case_id: str
    time: np.ndarray
    solid_fraction: np.ndarray
    pressure_drop: np.ndarray
    bed_height: np.ndarray
    params: np.ndarray
    param_names: tuple[str, ...]
    cell_volumes: np.ndarray
    solid_volume: np.ndarray

    def validate(self) -> None:
        time_steps = int(self.solid_fraction.shape[0])
        if self.solid_fraction.ndim < 3:
            raise ValueError("solid_fraction must have shape [time, *grid] with at least 2-D grid")
        if self.time.shape != (time_steps,):
            raise ValueError("time must align with solid_fraction")
        for name, values in (
            ("pressure_drop", self.pressure_drop),
            ("bed_height", self.bed_height),
            ("solid_volume", self.solid_volume),
        ):
            if values.shape != (time_steps,):
                raise ValueError(f"{name} must have shape [{time_steps}]")
        if self.cell_volumes.shape != self.solid_fraction.shape[1:]:
            raise ValueError("cell_volumes must match the spatial field shape")
        if self.params.ndim != 1 or len(self.params) != len(self.param_names):
            raise ValueError("params and param_names must be one-dimensional and aligned")
        arrays = (
            self.time,
            self.solid_fraction,
            self.pressure_drop,
            self.bed_height,
            self.params,
            self.cell_volumes,
            self.solid_volume,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError(f"Non-finite values found in case {self.case_id}")
        if np.any(self.solid_fraction < -1e-6):
            raise ValueError("solid_fraction contains negative values")
        if np.any(self.cell_volumes < 0):
            raise ValueError("cell_volumes contains negative values")


def save_case(case: CaseData, path: str | Path) -> None:
    case.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        case_id=np.asarray(case.case_id),
        time=case.time.astype(np.float64),
        solid_fraction=case.solid_fraction.astype(np.float32),
        pressure_drop=case.pressure_drop.astype(np.float32),
        bed_height=case.bed_height.astype(np.float32),
        params=case.params.astype(np.float32),
        param_names=np.asarray(case.param_names, dtype="U64"),
        cell_volumes=case.cell_volumes.astype(np.float64),
        solid_volume=case.solid_volume.astype(np.float64),
    )


def load_case(path: str | Path) -> CaseData:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "case_id",
            "time",
            "solid_fraction",
            "pressure_drop",
            "bed_height",
            "params",
            "param_names",
            "cell_volumes",
            "solid_volume",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{source} is missing arrays: {sorted(missing)}")
        case = CaseData(
            case_id=str(payload["case_id"].item()),
            time=payload["time"].astype(np.float64),
            solid_fraction=payload["solid_fraction"].astype(np.float32),
            pressure_drop=payload["pressure_drop"].astype(np.float32),
            bed_height=payload["bed_height"].astype(np.float32),
            params=payload["params"].astype(np.float32),
            param_names=tuple(str(item) for item in payload["param_names"].tolist()),
            cell_volumes=payload["cell_volumes"].astype(np.float64),
            solid_volume=payload["solid_volume"].astype(np.float64),
        )
    case.validate()
    return case


def load_case_directory(root: str | Path) -> list[CaseData]:
    paths = sorted(Path(root).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz cases found in {Path(root).resolve()}")
    cases = [load_case(path) for path in paths]
    expected_names = cases[0].param_names
    expected_grid = cases[0].solid_fraction.shape[1:]
    for case in cases[1:]:
        if case.param_names != expected_names:
            raise ValueError("All cases must use identical param_names in identical order")
        if case.solid_fraction.shape[1:] != expected_grid:
            raise ValueError("All cases must use the same spatial grid")
    return cases


def split_cases(
    cases: Sequence[CaseData],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[CaseData], list[CaseData], list[CaseData]]:
    if len(cases) < 5:
        raise ValueError("At least five complete cases are required for train/val/test splitting")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1):
        raise ValueError("train_fraction and val_fraction must lie in (0, 1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than one")
    order = np.random.default_rng(seed).permutation(len(cases))
    n_train = max(1, int(round(len(cases) * train_fraction)))
    n_val = max(1, int(round(len(cases) * val_fraction)))
    if n_train + n_val >= len(cases):
        n_train = len(cases) - n_val - 1
    train = [cases[index] for index in order[:n_train]]
    val = [cases[index] for index in order[n_train : n_train + n_val]]
    test = [cases[index] for index in order[n_train + n_val :]]
    return train, val, test


@dataclass
class PreparedCase:
    case_id: str
    coeff_scaled: np.ndarray
    params_scaled: np.ndarray
    macros_scaled: np.ndarray
    fields_flat: np.ndarray
    cell_volumes_flat: np.ndarray
    solid_volume: np.ndarray


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy windows that preserve complete-case splitting."""

    def __init__(
        self,
        cases: Sequence[PreparedCase],
        history: int,
        horizon: int,
        stride: int = 1,
    ) -> None:
        if history < 1 or horizon < 1 or stride < 1:
            raise ValueError("history, horizon, and stride must be positive")
        self.cases = list(cases)
        self.history = history
        self.horizon = horizon
        self.indices: list[tuple[int, int]] = []
        for case_index, case in enumerate(self.cases):
            stop = len(case.coeff_scaled) - history - horizon + 1
            self.indices.extend((case_index, start) for start in range(0, max(0, stop), stride))
        if not self.indices:
            raise ValueError("No windows available; reduce history/horizon or add longer cases")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case_index, start = self.indices[index]
        case = self.cases[case_index]
        pivot = start + self.history
        end = pivot + self.horizon
        return {
            "history_coeff": torch.from_numpy(case.coeff_scaled[start:pivot]).float(),
            "params": torch.from_numpy(case.params_scaled).float(),
            "future_coeff": torch.from_numpy(case.coeff_scaled[pivot:end]).float(),
            "future_macro": torch.from_numpy(case.macros_scaled[pivot:end]).float(),
            "future_field": torch.from_numpy(case.fields_flat[pivot:end]).float(),
            "cell_volumes": torch.from_numpy(case.cell_volumes_flat).float(),
            "solid_volume": torch.from_numpy(case.solid_volume[pivot:end]).float(),
            "case_index": torch.tensor(case_index, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }
