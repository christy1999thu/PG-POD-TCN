import numpy as np

from pg_pod_tcn.data import load_case
from pg_pod_tcn.synthetic import generate_synthetic_dataset


def test_synthetic_cases_are_valid_and_mass_conserving(tmp_path):
    paths = generate_synthetic_dataset(
        tmp_path, cases=5, timesteps=20, height=10, width=8, dt=0.02, seed=4
    )
    case = load_case(paths[0])
    measured = np.sum(case.solid_fraction * case.cell_volumes, axis=(1, 2))
    np.testing.assert_allclose(measured, case.solid_volume, rtol=2e-6, atol=1e-10)
    assert np.max(case.solid_fraction) <= 0.620001
    assert case.param_names[0] == "velocity_ratio"

