from __future__ import annotations

from pathlib import Path

import numpy as np

from pg_pod_tcn.data import CaseData, save_case

PARAM_NAMES = (
    "velocity_ratio",
    "particle_diameter_mm",
    "particle_density_kg_m3",
    "restitution_coefficient",
    "friction_coefficient",
)


def generate_synthetic_dataset(
    root: str | Path,
    cases: int = 10,
    timesteps: int = 100,
    height: int = 24,
    width: int = 12,
    dt: float = 0.02,
    seed: int = 17,
    overwrite: bool = False,
) -> list[Path]:
    """Generate a deterministic, mass-conserving conical-bed demonstration dataset.

    The generator is a software test fixture, not a validated physical solver and not a
    replacement for CFD-DEM or experiments.
    """
    if cases < 5:
        raise ValueError("At least five synthetic cases are required")
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_root.glob("case_*.npz"))
    if existing and not overwrite:
        return existing

    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for index in range(cases):
        fraction = (index + 0.5) / cases
        params = np.asarray(
            [
                0.90 + 0.70 * fraction + rng.normal(0, 0.015),
                0.40 + 0.80 * ((index * 7) % cases + 0.5) / cases,
                1100.0 + 1400.0 * ((index * 3) % cases + 0.5) / cases,
                0.70 + 0.25 * ((index * 9) % cases + 0.5) / cases,
                0.20 + 0.35 * ((index * 5) % cases + 0.5) / cases,
            ],
            dtype=np.float32,
        )
        case = generate_synthetic_case(
            case_id=f"case_{index:03d}",
            params=params,
            timesteps=timesteps,
            height=height,
            width=width,
            dt=dt,
            phase=float(rng.uniform(0, 2 * np.pi)),
        )
        path = output_root / f"{case.case_id}.npz"
        save_case(case, path)
        paths.append(path)
    return paths


def generate_synthetic_case(
    case_id: str,
    params: np.ndarray,
    timesteps: int,
    height: int,
    width: int,
    dt: float,
    phase: float = 0.0,
) -> CaseData:
    velocity_ratio, _, particle_density, restitution, friction = params
    z = (np.arange(height, dtype=np.float64) + 0.5) / height
    x = (np.arange(width, dtype=np.float64) + 0.5) / width - 0.5
    zz, xx = np.meshgrid(z, x, indexing="ij")
    half_width = 0.17 + 0.34 * zz
    mask = np.abs(xx) <= half_width
    cell_volumes = np.where(mask, 1.0e-6, 0.0)

    base_height = 0.35 + 0.13 * (velocity_ratio - 0.9) / 0.7
    reference = 0.54 / (1.0 + np.exp((zz - base_height) / 0.018))
    reference *= mask
    target_solid_volume = float(np.sum(reference * cell_volumes))

    fields = np.zeros((timesteps, height, width), dtype=np.float32)
    pressure_drop = np.zeros(timesteps, dtype=np.float32)
    bed_height = np.zeros(timesteps, dtype=np.float32)
    time = np.arange(timesteps, dtype=np.float64) * dt

    oscillation_frequency = 1.2 + 1.6 * (velocity_ratio - 0.9) / 0.7
    inlet_area = max(float(np.sum(mask[0])) * 1.0e-4, 1.0e-4)
    base_pressure = target_solid_volume * particle_density * 9.81 / inlet_area

    for step, current_time in enumerate(time):
        angle = 2 * np.pi * oscillation_frequency * current_time + phase
        dynamic_height = base_height + 0.018 * np.sin(angle)
        field = 0.54 / (1.0 + np.exp((zz - dynamic_height) / 0.018))

        bubble_z = 0.08 + (0.78 * (current_time * (0.22 + 0.08 * velocity_ratio) + phase / 7)) % 0.78
        bubble_x = 0.11 * np.sin(0.65 * angle + 0.4 * restitution)
        radius_z = 0.055 + 0.018 * velocity_ratio
        radius_x = 0.06 + 0.02 * (1.0 - friction)
        bubble = np.exp(
            -0.5 * (((zz - bubble_z) / radius_z) ** 2 + ((xx - bubble_x) / radius_x) ** 2)
        )
        field *= 1.0 - 0.72 * bubble
        field *= 1.0 + 0.035 * np.sin(3.0 * angle + 8.0 * xx)
        field *= mask
        field = _enforce_solid_volume(field, cell_volumes, target_solid_volume, alpha_max=0.62)
        fields[step] = field.astype(np.float32)

        row_average = np.divide(
            np.sum(field, axis=1),
            np.maximum(np.sum(mask, axis=1), 1),
        )
        occupied = np.flatnonzero(row_average > 0.08)
        bed_height[step] = float(z[occupied[-1]]) if len(occupied) else 0.0
        pressure_drop[step] = float(
            base_pressure
            * (1.0 + 0.045 * np.sin(angle - 0.7) + 0.015 * np.sin(2.4 * angle))
        )

    solid_volume = np.full(timesteps, target_solid_volume, dtype=np.float64)
    return CaseData(
        case_id=case_id,
        time=time,
        solid_fraction=fields,
        pressure_drop=pressure_drop,
        bed_height=bed_height,
        params=np.asarray(params, dtype=np.float32),
        param_names=PARAM_NAMES,
        cell_volumes=cell_volumes,
        solid_volume=solid_volume,
    )


def _enforce_solid_volume(
    field: np.ndarray,
    cell_volumes: np.ndarray,
    target: float,
    alpha_max: float,
) -> np.ndarray:
    result = np.clip(np.asarray(field, dtype=np.float64), 0.0, alpha_max)
    valid = cell_volumes > 0
    for _ in range(20):
        current = float(np.sum(result * cell_volumes))
        error = target - current
        if abs(error) <= max(target, 1e-12) * 1e-8:
            break
        if error > 0:
            adjustable = valid & (result < alpha_max - 1e-10)
        else:
            adjustable = valid & (result > 1e-10)
        available_volume = float(np.sum(cell_volumes[adjustable]))
        if available_volume <= 0:
            break
        result[adjustable] += error / available_volume
        result = np.clip(result, 0.0, alpha_max)
    result[~valid] = 0.0
    return result
