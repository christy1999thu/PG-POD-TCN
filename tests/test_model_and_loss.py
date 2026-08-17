import numpy as np
import torch

from pg_pod_tcn.data import WindowDataset
from pg_pod_tcn.losses import PhysicsGuidedLoss
from pg_pod_tcn.models import TCNForecaster
from pg_pod_tcn.preprocessing import fit_preprocessors
from pg_pod_tcn.synthetic import generate_synthetic_case


def _cases():
    cases = []
    for index in range(5):
        params = np.asarray([1.0 + 0.1 * index, 0.7, 1500, 0.8, 0.3], dtype=np.float32)
        cases.append(
            generate_synthetic_case(
                f"case-{index}", params, timesteps=24, height=8, width=6, dt=0.02, phase=index
            )
        )
    return cases


def test_model_output_and_physics_loss_are_finite():
    cases = _cases()
    preprocessors = fit_preprocessors(cases, pod_energy=0.99, pod_max_modes=8)
    dataset = WindowDataset(preprocessors.prepare(cases), history=5, horizon=3, stride=2)
    batch = {name: value.unsqueeze(0) for name, value in dataset[0].items()}
    model = TCNForecaster(
        n_modes=preprocessors.pod.n_modes,
        n_params=len(preprocessors.param_names),
        horizon=3,
        channels=16,
        levels=2,
    )
    coefficient_prediction, macro_prediction = model(batch["history_coeff"], batch["params"])
    criterion = PhysicsGuidedLoss(preprocessors, {})
    losses = criterion(coefficient_prediction, macro_prediction, batch)
    assert coefficient_prediction.shape == (1, 3, preprocessors.pod.n_modes)
    assert macro_prediction.shape == (1, 3, 2)
    assert torch.isfinite(losses["total"])

