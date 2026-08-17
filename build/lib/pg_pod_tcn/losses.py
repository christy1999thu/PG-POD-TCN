from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pg_pod_tcn.preprocessing import PreprocessorBundle


class PODDecoder(nn.Module):
    """Differentiable inverse coefficient scaling and POD reconstruction."""

    def __init__(self, preprocessors: PreprocessorBundle) -> None:
        super().__init__()
        self.register_buffer("basis", torch.from_numpy(preprocessors.pod.basis_).float())
        self.register_buffer("field_mean", torch.from_numpy(preprocessors.pod.mean_).float())
        self.register_buffer(
            "coefficient_mean",
            torch.from_numpy(np.asarray(preprocessors.coefficient_scaler.mean_)).float(),
        )
        self.register_buffer(
            "coefficient_scale",
            torch.from_numpy(np.asarray(preprocessors.coefficient_scaler.scale_)).float(),
        )

    def coefficients_physical(self, coefficients_scaled: torch.Tensor) -> torch.Tensor:
        return coefficients_scaled * self.coefficient_scale + self.coefficient_mean

    def forward(self, coefficients_scaled: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficients_physical(coefficients_scaled)
        return torch.einsum("bhr,rn->bhn", coefficients, self.basis) + self.field_mean


class PhysicsGuidedLoss(nn.Module):
    def __init__(
        self,
        preprocessors: PreprocessorBundle,
        loss_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.decoder = PODDecoder(preprocessors)
        self.weights = {
            "coefficient": float(loss_config.get("coefficient", 1.0)),
            "field": float(loss_config.get("field", 0.5)),
            "mass": float(loss_config.get("mass", 0.2)),
            "bounds": float(loss_config.get("bounds", 0.1)),
            "macro": float(loss_config.get("macro", 0.5)),
        }
        self.alpha_max = float(loss_config.get("alpha_max", 0.64))

    def forward(
        self,
        coefficient_prediction: torch.Tensor,
        macro_prediction: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        field_prediction = self.decoder(coefficient_prediction)
        coefficient_loss = F.huber_loss(coefficient_prediction, batch["future_coeff"])
        field_loss = F.mse_loss(field_prediction, batch["future_field"])

        volumes = batch["cell_volumes"].unsqueeze(1)
        predicted_solid_volume = torch.sum(field_prediction * volumes, dim=-1)
        volume_target = batch["solid_volume"]
        relative_mass_error = (predicted_solid_volume - volume_target) / volume_target.clamp_min(1e-12)
        mass_loss = torch.mean(relative_mass_error**2)

        bounds_loss = torch.mean(F.relu(-field_prediction) ** 2) + torch.mean(
            F.relu(field_prediction - self.alpha_max) ** 2
        )
        macro_loss = F.mse_loss(macro_prediction, batch["future_macro"])
        components = {
            "coefficient": coefficient_loss,
            "field": field_loss,
            "mass": mass_loss,
            "bounds": bounds_loss,
            "macro": macro_loss,
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return {"total": total, **components, "field_prediction": field_prediction}

