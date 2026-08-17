from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pg_pod_tcn.baselines import DMDModel, persistence_forecast
from pg_pod_tcn.losses import PODDecoder
from pg_pod_tcn.metrics import global_ssim, mae, mape, mass_drift_percent, nrmse
from pg_pod_tcn.models import build_model
from pg_pod_tcn.ood import MahalanobisDetector, dataset_features
from pg_pod_tcn.pipeline import DataBundle, build_data_bundle
from pg_pod_tcn.preprocessing import PreprocessorBundle
from pg_pod_tcn.utils import resolve_device, write_json


def evaluate_experiment(config: dict[str, Any], split: str = "test") -> dict[str, Any]:
    output_root = Path(config["output"]["root"])
    preprocessors = PreprocessorBundle.load(output_root / "preprocessors.npz")
    bundle = build_data_bundle(config, preprocessors=preprocessors)
    dataset = _select_dataset(bundle, split)
    prepared = _select_prepared(bundle, split)
    device = resolve_device(str(config["training"].get("device", "auto")))
    models = load_ensemble(output_root, config, preprocessors, device)

    prediction = _predict_dataset(
        models=models,
        dataset=dataset,
        preprocessors=preprocessors,
        device=device,
        batch_size=int(config["training"].get("batch_size", 64)),
    )
    detector_path = output_root / "ood_detector.npz"
    if detector_path.exists():
        detector = MahalanobisDetector.load(detector_path)
        features = dataset_features(dataset)
        ood_scores = detector.score(features)
        ood_flags = detector.predict(features)
    else:
        ood_scores = np.full(len(dataset), np.nan)
        ood_flags = np.zeros(len(dataset), dtype=bool)

    metrics = _calculate_metrics(prediction)
    metrics["ood_fraction"] = float(np.mean(ood_flags))
    metrics["ood_threshold"] = (
        float(detector.threshold_) if detector_path.exists() else None
    )

    dmd = DMDModel().fit(bundle.train_prepared)
    persistence_coeff = persistence_forecast(
        prediction["history_coeff"], int(config["data"]["horizon"])
    )
    dmd_coeff = dmd.predict(prediction["history_coeff"], int(config["data"]["horizon"]))
    metrics["baselines"] = {
        "persistence_field_nrmse": nrmse(
            _decode_numpy(persistence_coeff, preprocessors), prediction["target_field"]
        ),
        "pod_dmd_field_nrmse": nrmse(
            _decode_numpy(dmd_coeff, preprocessors), prediction["target_field"]
        ),
    }
    metrics["split"] = split
    metrics["cases"] = [case.case_id for case in prepared]
    metrics["windows"] = len(dataset)
    write_json(output_root / f"metrics_{split}.json", metrics)
    _save_predictions(output_root / f"predictions_{split}.npz", prediction, ood_scores, ood_flags)
    _plot_example(output_root / f"example_{split}.png", prediction, preprocessors.grid_shape)
    return metrics


def load_ensemble(
    output_root: Path,
    config: dict[str, Any],
    preprocessors: PreprocessorBundle,
    device: torch.device,
) -> list[torch.nn.Module]:
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            checkpoint_names = json.load(handle).get("checkpoints", [])
        paths = [output_root / name for name in checkpoint_names]
    else:
        paths = sorted(output_root.glob("checkpoint_seed_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {output_root.resolve()}")
    models = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = build_model(
            checkpoint.get("model_config", config["model"]),
            n_modes=int(checkpoint.get("n_modes", preprocessors.pod.n_modes)),
            n_params=int(checkpoint.get("n_params", len(preprocessors.param_names))),
            horizon=int(checkpoint.get("horizon", config["data"]["horizon"])),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
    return models


def _predict_dataset(
    models: list[torch.nn.Module],
    dataset: torch.utils.data.Dataset,
    preprocessors: PreprocessorBundle,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    decoder = PODDecoder(preprocessors).to(device)
    outputs: dict[str, list[np.ndarray]] = {
        "field": [],
        "field_std": [],
        "macro": [],
        "macro_std": [],
        "target_field": [],
        "target_macro_scaled": [],
        "history_coeff": [],
        "cell_volumes": [],
        "solid_volume": [],
    }
    elapsed = 0.0
    with torch.no_grad():
        for raw_batch in loader:
            batch = {name: value.to(device) for name, value in raw_batch.items()}
            _synchronize(device)
            started = time.perf_counter()
            model_coefficients = []
            model_macros = []
            for model in models:
                coefficient_prediction, macro_prediction = model(
                    batch["history_coeff"], batch["params"]
                )
                model_coefficients.append(coefficient_prediction)
                model_macros.append(macro_prediction)
            macro_stack = torch.stack(model_macros)
            field_stack = torch.stack([decoder(item) for item in model_coefficients])
            _synchronize(device)
            elapsed += time.perf_counter() - started

            outputs["field"].append(field_stack.mean(dim=0).cpu().numpy())
            outputs["field_std"].append(field_stack.std(dim=0, unbiased=False).cpu().numpy())
            outputs["macro"].append(macro_stack.mean(dim=0).cpu().numpy())
            outputs["macro_std"].append(macro_stack.std(dim=0, unbiased=False).cpu().numpy())
            for name in (
                "future_field",
                "future_macro",
                "history_coeff",
                "cell_volumes",
                "solid_volume",
            ):
                output_name = {
                    "future_field": "target_field",
                    "future_macro": "target_macro_scaled",
                }.get(name, name)
                outputs[output_name].append(batch[name].cpu().numpy())

    merged = {name: np.concatenate(values, axis=0) for name, values in outputs.items()}
    merged["inference_ms_per_window"] = 1000.0 * elapsed / max(len(dataset), 1)
    macro_mean = np.asarray(preprocessors.macro_scaler.mean_)
    macro_scale = np.asarray(preprocessors.macro_scaler.scale_)
    merged["macro_physical"] = merged["macro"] * macro_scale + macro_mean
    merged["target_macro"] = merged["target_macro_scaled"] * macro_scale + macro_mean
    merged["macro_std_physical"] = merged["macro_std"] * macro_scale
    return merged


def _calculate_metrics(prediction: dict[str, np.ndarray | float]) -> dict[str, float]:
    field = np.asarray(prediction["field"])
    target_field = np.asarray(prediction["target_field"])
    macro = np.asarray(prediction["macro_physical"])
    target_macro = np.asarray(prediction["target_macro"])
    return {
        "field_nrmse": nrmse(field, target_field),
        "field_mae": mae(field, target_field),
        "field_global_ssim": global_ssim(field, target_field, data_range=0.64),
        "pressure_drop_mape": mape(macro[..., 0], target_macro[..., 0]),
        "bed_height_mape": mape(macro[..., 1], target_macro[..., 1]),
        "mass_drift_percent": mass_drift_percent(
            field,
            np.asarray(prediction["cell_volumes"]),
            np.asarray(prediction["solid_volume"]),
        ),
        "inference_ms_per_window": float(prediction["inference_ms_per_window"]),
    }


def _decode_numpy(coefficients_scaled: np.ndarray, preprocessors: PreprocessorBundle) -> np.ndarray:
    coefficients = preprocessors.coefficient_scaler.inverse_transform(coefficients_scaled)
    shape = coefficients.shape
    fields = preprocessors.pod.inverse_transform(coefficients.reshape(-1, shape[-1]))
    return fields.reshape(*shape[:-1], fields.shape[-1])


def _save_predictions(
    path: Path,
    prediction: dict[str, np.ndarray | float],
    ood_scores: np.ndarray,
    ood_flags: np.ndarray,
) -> None:
    limit = min(64, len(ood_scores))
    np.savez_compressed(
        path,
        field=np.asarray(prediction["field"])[:limit],
        field_std=np.asarray(prediction["field_std"])[:limit],
        target_field=np.asarray(prediction["target_field"])[:limit],
        macro=np.asarray(prediction["macro_physical"])[:limit],
        macro_std=np.asarray(prediction["macro_std_physical"])[:limit],
        target_macro=np.asarray(prediction["target_macro"])[:limit],
        ood_score=ood_scores[:limit],
        ood_flag=ood_flags[:limit],
    )


def _plot_example(path: Path, prediction: dict[str, np.ndarray | float], grid_shape: tuple[int, ...]) -> None:
    if len(grid_shape) != 2:
        return
    truth = np.asarray(prediction["target_field"])[0, -1].reshape(grid_shape)
    forecast = np.asarray(prediction["field"])[0, -1].reshape(grid_shape)
    macro = np.asarray(prediction["macro_physical"])[0]
    target_macro = np.asarray(prediction["target_macro"])[0]
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    axes[0].imshow(truth, origin="lower", vmin=0, vmax=0.64, aspect="auto")
    axes[0].set_title("CFD-DEM target")
    axes[1].imshow(forecast, origin="lower", vmin=0, vmax=0.64, aspect="auto")
    axes[1].set_title("PG-POD-TCN")
    image = axes[2].imshow(np.abs(forecast - truth), origin="lower", aspect="auto")
    axes[2].set_title("Absolute error")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    axes[3].plot(target_macro[:, 0], label="target pressure")
    axes[3].plot(macro[:, 0], "--", label="predicted pressure")
    axes[3].set_title("Pressure-drop horizon")
    axes[3].legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _select_dataset(bundle: DataBundle, split: str):
    mapping = {
        "train": bundle.train_dataset,
        "validation": bundle.val_dataset,
        "val": bundle.val_dataset,
        "test": bundle.test_dataset,
    }
    if split not in mapping:
        raise ValueError(f"Unknown split: {split}")
    return mapping[split]


def _select_prepared(bundle: DataBundle, split: str):
    mapping = {
        "train": bundle.train_prepared,
        "validation": bundle.val_prepared,
        "val": bundle.val_prepared,
        "test": bundle.test_prepared,
    }
    return mapping[split]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
