from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pg_pod_tcn.losses import PhysicsGuidedLoss
from pg_pod_tcn.models import build_model
from pg_pod_tcn.ood import MahalanobisDetector, dataset_features
from pg_pod_tcn.pipeline import DataBundle, build_data_bundle
from pg_pod_tcn.utils import resolve_device, set_seed, write_json


def train_experiment(config: dict[str, Any]) -> DataBundle:
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config.json", config)
    bundle = build_data_bundle(config, save_artifacts_to=output_root)

    detector = MahalanobisDetector(
        regularization=float(config.get("ood", {}).get("regularization", 1e-3)),
        quantile=float(config.get("ood", {}).get("quantile", 0.99)),
    ).fit(dataset_features(bundle.train_dataset))
    detector.calibrate(dataset_features(bundle.val_dataset))
    detector.save(output_root / "ood_detector.npz")

    checkpoint_paths = []
    for seed in config["training"].get("seeds", [config.get("project", {}).get("seed", 42)]):
        checkpoint_paths.append(train_one_seed(config, bundle, int(seed), output_root))

    write_json(
        output_root / "manifest.json",
        {
            "project": config.get("project", {}),
            "n_modes": bundle.preprocessors.pod.n_modes,
            "captured_pod_energy": bundle.preprocessors.pod.captured_energy_,
            "n_params": len(bundle.preprocessors.param_names),
            "grid_shape": bundle.preprocessors.grid_shape,
            "checkpoints": [str(path.name) for path in checkpoint_paths],
            "synthetic_data_warning": (
                "Metrics are demonstration-only if the configured data root contains synthetic data."
            ),
        },
    )
    return bundle


def train_one_seed(
    config: dict[str, Any],
    bundle: DataBundle,
    seed: int,
    output_root: Path,
) -> Path:
    set_seed(seed)
    training_config = config["training"]
    device = resolve_device(str(training_config.get("device", "auto")))
    batch_size = int(training_config.get("batch_size", 64))
    num_workers = int(training_config.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(
        config["model"],
        n_modes=bundle.preprocessors.pod.n_modes,
        n_params=len(bundle.preprocessors.param_names),
        horizon=int(config["data"]["horizon"]),
    ).to(device)
    criterion = PhysicsGuidedLoss(bundle.preprocessors, config["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, int(training_config.get("patience", 20)) // 3)
    )

    checkpoint_path = output_root / f"checkpoint_seed_{seed}.pt"
    best_validation = math.inf
    patience = int(training_config.get("patience", 20))
    stale_epochs = 0
    history = []
    for epoch in range(1, int(training_config.get("epochs", 300)) + 1):
        train_metrics = _run_epoch(
            model,
            criterion,
            train_loader,
            device,
            optimizer=optimizer,
            gradient_clip=float(training_config.get("gradient_clip", 1.0)),
        )
        validation_metrics = _run_epoch(model, criterion, val_loader, device)
        scheduler.step(validation_metrics["total"])
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        print(
            f"seed={seed} epoch={epoch:03d} "
            f"train={train_metrics['total']:.6f} val={validation_metrics['total']:.6f}"
        )
        if validation_metrics["total"] < best_validation - 1e-8:
            best_validation = validation_metrics["total"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config["model"],
                    "n_modes": bundle.preprocessors.pod.n_modes,
                    "n_params": len(bundle.preprocessors.param_names),
                    "horizon": int(config["data"]["horizon"]),
                    "history": int(config["data"]["history"]),
                    "seed": seed,
                    "best_validation_loss": best_validation,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    write_json(output_root / f"history_seed_{seed}.json", history)
    return checkpoint_path


def _run_epoch(
    model: torch.nn.Module,
    criterion: PhysicsGuidedLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "coefficient", "field", "mass", "bounds", "macro")}
    samples = 0
    for raw_batch in loader:
        batch = {name: value.to(device) for name, value in raw_batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            coefficient_prediction, macro_prediction = model(
                batch["history_coeff"], batch["params"]
            )
            losses = criterion(coefficient_prediction, macro_prediction, batch)
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        batch_size = int(batch["history_coeff"].shape[0])
        samples += batch_size
        for name in totals:
            totals[name] += float(losses[name].detach().cpu()) * batch_size
    return {name: value / max(samples, 1) for name, value in totals.items()}
