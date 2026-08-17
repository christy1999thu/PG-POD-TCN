from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pg_pod_tcn.data import (
    CaseData,
    PreparedCase,
    WindowDataset,
    load_case_directory,
    split_cases,
)
from pg_pod_tcn.preprocessing import PreprocessorBundle, fit_preprocessors
from pg_pod_tcn.utils import write_json


@dataclass
class DataBundle:
    train_cases: list[CaseData]
    val_cases: list[CaseData]
    test_cases: list[CaseData]
    train_prepared: list[PreparedCase]
    val_prepared: list[PreparedCase]
    test_prepared: list[PreparedCase]
    train_dataset: WindowDataset
    val_dataset: WindowDataset
    test_dataset: WindowDataset
    preprocessors: PreprocessorBundle


def build_data_bundle(
    config: dict[str, Any],
    preprocessors: PreprocessorBundle | None = None,
    save_artifacts_to: str | Path | None = None,
) -> DataBundle:
    data_config = config["data"]
    cases = load_case_directory(data_config["root"])
    train_cases, val_cases, test_cases = split_cases(
        cases,
        train_fraction=float(data_config["train_fraction"]),
        val_fraction=float(data_config["val_fraction"]),
        seed=int(data_config.get("split_seed", 42)),
    )
    if preprocessors is None:
        max_modes_value = data_config.get("pod_max_modes")
        preprocessors = fit_preprocessors(
            train_cases,
            pod_energy=float(data_config.get("pod_energy", 0.99)),
            pod_max_modes=None if max_modes_value is None else int(max_modes_value),
        )

    train_prepared = preprocessors.prepare(train_cases)
    val_prepared = preprocessors.prepare(val_cases)
    test_prepared = preprocessors.prepare(test_cases)
    dataset_options = dict(
        history=int(data_config["history"]),
        horizon=int(data_config["horizon"]),
        stride=int(data_config.get("stride", 1)),
    )
    bundle = DataBundle(
        train_cases=train_cases,
        val_cases=val_cases,
        test_cases=test_cases,
        train_prepared=train_prepared,
        val_prepared=val_prepared,
        test_prepared=test_prepared,
        train_dataset=WindowDataset(train_prepared, **dataset_options),
        val_dataset=WindowDataset(val_prepared, **dataset_options),
        test_dataset=WindowDataset(test_prepared, **dataset_options),
        preprocessors=preprocessors,
    )
    if save_artifacts_to is not None:
        artifact_root = Path(save_artifacts_to)
        preprocessors.save(artifact_root / "preprocessors.npz")
        write_json(
            artifact_root / "split.json",
            {
                "train": [case.case_id for case in train_cases],
                "validation": [case.case_id for case in val_cases],
                "test": [case.case_id for case in test_cases],
            },
        )
    return bundle

