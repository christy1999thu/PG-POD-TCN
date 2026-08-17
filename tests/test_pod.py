import numpy as np

from pg_pod_tcn.pod import PODReducer


def test_pod_reconstructs_low_rank_snapshots():
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(40, 3))
    basis = rng.normal(size=(3, 20))
    snapshots = latent @ basis + 0.2
    pod = PODReducer(energy=0.999999, max_modes=3).fit(snapshots)
    reconstructed = pod.inverse_transform(pod.transform(snapshots))
    relative_error = np.linalg.norm(reconstructed - snapshots) / np.linalg.norm(snapshots)
    assert pod.n_modes == 3
    assert relative_error < 1e-5

