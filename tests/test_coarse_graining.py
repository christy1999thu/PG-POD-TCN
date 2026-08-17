import numpy as np

from pg_pod_tcn.coarse_graining import particles_to_solid_fraction


def test_particle_binning_conserves_volume():
    positions = np.asarray(
        [
            [[0.25, 0.25], [0.75, 0.75]],
            [[0.25, 0.75], [0.75, 0.25]],
        ]
    )
    fields, cell_volumes = particles_to_solid_fraction(
        positions,
        grid_edges=[np.linspace(0, 1, 3), np.linspace(0, 1, 3)],
        particle_volumes=0.01,
    )
    measured = np.sum(fields * cell_volumes, axis=(1, 2))
    np.testing.assert_allclose(measured, [0.02, 0.02])

