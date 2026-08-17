from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def particles_to_solid_fraction(
    positions: np.ndarray,
    grid_edges: Sequence[np.ndarray],
    particle_volumes: float | np.ndarray,
    domain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively bin Lagrangian particles onto a regular Eulerian grid.

    Parameters
    ----------
    positions:
        Particle centers with shape [time, particles, dimensions].
    grid_edges:
        One monotonically increasing edge array per spatial dimension.
    particle_volumes:
        Scalar volume for monodisperse particles, [particles], or [time, particles].
    domain_mask:
        Optional boolean grid mask. Cells outside the domain receive zero volume.

    Returns
    -------
    solid_fraction, cell_volumes
        Arrays with shapes [time, *grid] and [*grid]. Particles outside the grid are
        ignored; callers should check conservation and reject such snapshots if needed.
    """
    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.ndim != 3:
        raise ValueError("positions must have shape [time, particles, dimensions]")
    if coordinates.shape[-1] != len(grid_edges):
        raise ValueError("The number of grid edge arrays must equal the coordinate dimension")
    edges = [np.asarray(values, dtype=np.float64) for values in grid_edges]
    if any(values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) <= 0) for values in edges):
        raise ValueError("Each grid edge array must be one-dimensional and strictly increasing")

    widths = [np.diff(values) for values in edges]
    cell_volumes = widths[0]
    for values in widths[1:]:
        cell_volumes = np.multiply.outer(cell_volumes, values)
    if domain_mask is not None:
        mask = np.asarray(domain_mask, dtype=bool)
        if mask.shape != cell_volumes.shape:
            raise ValueError("domain_mask must match the grid shape")
        cell_volumes = np.where(mask, cell_volumes, 0.0)

    weights = _broadcast_particle_volumes(
        particle_volumes, coordinates.shape[0], coordinates.shape[1]
    )
    result = np.zeros((coordinates.shape[0], *cell_volumes.shape), dtype=np.float64)
    for time_index in range(coordinates.shape[0]):
        particle_volume_grid, _ = np.histogramdd(
            coordinates[time_index], bins=edges, weights=weights[time_index]
        )
        result[time_index] = np.divide(
            particle_volume_grid,
            cell_volumes,
            out=np.zeros_like(particle_volume_grid),
            where=cell_volumes > 0,
        )
    return result.astype(np.float32), cell_volumes


def _broadcast_particle_volumes(
    particle_volumes: float | np.ndarray,
    time_steps: int,
    particles: int,
) -> np.ndarray:
    values = np.asarray(particle_volumes, dtype=np.float64)
    if values.ndim == 0:
        return np.full((time_steps, particles), float(values))
    if values.shape == (particles,):
        return np.broadcast_to(values, (time_steps, particles))
    if values.shape == (time_steps, particles):
        return values
    raise ValueError("particle_volumes must be scalar, [particles], or [time, particles]")

