from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pg_pod_tcn.data import CaseData, PreparedCase
from pg_pod_tcn.pod import PODReducer


@dataclass
class ArrayStandardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> ArrayStandardizer:
        array = np.asarray(values, dtype=np.float64)
        self.mean_ = array.mean(axis=0)
        scale = array.std(axis=0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (values * self.scale_ + self.mean_).astype(np.float32)

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ArrayStandardizer has not been fitted")


@dataclass
class PreprocessorBundle:
    pod: PODReducer
    coefficient_scaler: ArrayStandardizer
    parameter_scaler: ArrayStandardizer
    macro_scaler: ArrayStandardizer
    grid_shape: tuple[int, ...]
    param_names: tuple[str, ...]

    def prepare(self, cases: Sequence[CaseData]) -> list[PreparedCase]:
        prepared: list[PreparedCase] = []
        for case in cases:
            fields = case.solid_fraction.reshape(len(case.time), -1).astype(np.float32)
            coefficients = self.pod.transform(fields)
            macros = np.column_stack((case.pressure_drop, case.bed_height))
            prepared.append(
                PreparedCase(
                    case_id=case.case_id,
                    coeff_scaled=self.coefficient_scaler.transform(coefficients),
                    params_scaled=self.parameter_scaler.transform(case.params),
                    macros_scaled=self.macro_scaler.transform(macros),
                    fields_flat=fields,
                    cell_volumes_flat=case.cell_volumes.reshape(-1).astype(np.float32),
                    solid_volume=case.solid_volume.astype(np.float32),
                )
            )
        return prepared

    def save(self, path: str | Path) -> None:
        self.pod._check_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            pod_energy=np.asarray(self.pod.energy),
            pod_max_modes=np.asarray(-1 if self.pod.max_modes is None else self.pod.max_modes),
            pod_mean=self.pod.mean_,
            pod_basis=self.pod.basis_,
            pod_singular_values=self.pod.singular_values_,
            pod_captured_energy=np.asarray(self.pod.captured_energy_),
            coeff_mean=self.coefficient_scaler.mean_,
            coeff_scale=self.coefficient_scaler.scale_,
            param_mean=self.parameter_scaler.mean_,
            param_scale=self.parameter_scaler.scale_,
            macro_mean=self.macro_scaler.mean_,
            macro_scale=self.macro_scaler.scale_,
            grid_shape=np.asarray(self.grid_shape, dtype=np.int64),
            param_names=np.asarray(self.param_names, dtype="U64"),
        )

    @classmethod
    def load(cls, path: str | Path) -> PreprocessorBundle:
        with np.load(path, allow_pickle=False) as payload:
            max_modes_value = int(payload["pod_max_modes"].item())
            pod = PODReducer(
                energy=float(payload["pod_energy"].item()),
                max_modes=None if max_modes_value < 0 else max_modes_value,
                mean_=payload["pod_mean"].astype(np.float32),
                basis_=payload["pod_basis"].astype(np.float32),
                singular_values_=payload["pod_singular_values"].astype(np.float64),
                captured_energy_=float(payload["pod_captured_energy"].item()),
            )
            return cls(
                pod=pod,
                coefficient_scaler=ArrayStandardizer(
                    payload["coeff_mean"].astype(np.float32),
                    payload["coeff_scale"].astype(np.float32),
                ),
                parameter_scaler=ArrayStandardizer(
                    payload["param_mean"].astype(np.float32),
                    payload["param_scale"].astype(np.float32),
                ),
                macro_scaler=ArrayStandardizer(
                    payload["macro_mean"].astype(np.float32),
                    payload["macro_scale"].astype(np.float32),
                ),
                grid_shape=tuple(int(item) for item in payload["grid_shape"]),
                param_names=tuple(str(item) for item in payload["param_names"].tolist()),
            )


def fit_preprocessors(
    train_cases: Sequence[CaseData],
    pod_energy: float,
    pod_max_modes: int | None,
) -> PreprocessorBundle:
    if not train_cases:
        raise ValueError("train_cases cannot be empty")
    train_fields = np.concatenate(
        [case.solid_fraction.reshape(len(case.time), -1) for case in train_cases], axis=0
    )
    pod = PODReducer(energy=pod_energy, max_modes=pod_max_modes).fit(train_fields)
    train_coefficients = np.concatenate(
        [pod.transform(case.solid_fraction.reshape(len(case.time), -1)) for case in train_cases],
        axis=0,
    )
    train_params = np.stack([case.params for case in train_cases])
    train_macros = np.concatenate(
        [np.column_stack((case.pressure_drop, case.bed_height)) for case in train_cases],
        axis=0,
    )
    return PreprocessorBundle(
        pod=pod,
        coefficient_scaler=ArrayStandardizer().fit(train_coefficients),
        parameter_scaler=ArrayStandardizer().fit(train_params),
        macro_scaler=ArrayStandardizer().fit(train_macros),
        grid_shape=tuple(int(item) for item in train_cases[0].solid_fraction.shape[1:]),
        param_names=train_cases[0].param_names,
    )
