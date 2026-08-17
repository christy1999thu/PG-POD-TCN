from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pg_pod_tcn.data import WindowDataset, load_case
from pg_pod_tcn.evaluation import _predict_dataset, load_ensemble
from pg_pod_tcn.ood import MahalanobisDetector, dataset_features
from pg_pod_tcn.preprocessing import PreprocessorBundle
from pg_pod_tcn.utils import resolve_device


def predict_case_file(
    config: dict[str, Any],
    case_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    artifact_root = Path(config["output"]["root"])
    preprocessors = PreprocessorBundle.load(artifact_root / "preprocessors.npz")
    case = load_case(case_path)
    if case.param_names != preprocessors.param_names:
        raise ValueError(
            f"Parameter order mismatch. Expected {preprocessors.param_names}, got {case.param_names}"
        )
    prepared = preprocessors.prepare([case])
    dataset = WindowDataset(
        prepared,
        history=int(config["data"]["history"]),
        horizon=int(config["data"]["horizon"]),
        stride=int(config["data"].get("stride", 1)),
    )
    device = resolve_device(str(config["training"].get("device", "auto")))
    models = load_ensemble(artifact_root, config, preprocessors, device)
    prediction = _predict_dataset(
        models,
        dataset,
        preprocessors,
        device,
        batch_size=int(config["training"].get("batch_size", 64)),
    )
    detector = MahalanobisDetector.load(artifact_root / "ood_detector.npz")
    scores = detector.score(dataset_features(dataset))
    flags = scores > detector.threshold_

    destination = Path(output_path) if output_path else artifact_root / f"prediction_{case.case_id}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        case_id=np.asarray(case.case_id),
        predicted_solid_fraction=np.asarray(prediction["field"]).reshape(
            len(dataset), int(config["data"]["horizon"]), *preprocessors.grid_shape
        ),
        predictive_std=np.asarray(prediction["field_std"]).reshape(
            len(dataset), int(config["data"]["horizon"]), *preprocessors.grid_shape
        ),
        predicted_pressure_drop=np.asarray(prediction["macro_physical"])[..., 0],
        predicted_bed_height=np.asarray(prediction["macro_physical"])[..., 1],
        ood_score=scores,
        ood_flag=flags,
        ood_threshold=np.asarray(detector.threshold_),
    )
    return destination
