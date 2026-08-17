from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.convolution(F.pad(inputs, (self.left_padding, 0)))


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.normalization = nn.GroupNorm(1, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalization(self.network(inputs) + self.residual(inputs))


class TCNForecaster(nn.Module):
    def __init__(
        self,
        n_modes: int,
        n_params: int,
        horizon: int,
        channels: int = 128,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = n_modes + n_params
        for level in range(levels):
            blocks.append(
                TCNResidualBlock(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_channels = channels
        self.encoder = nn.Sequential(*blocks)
        self.coefficient_head = nn.Linear(channels, horizon * n_modes)
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)
        self.macro_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, horizon * 2),
        )
        self.n_modes = n_modes
        self.horizon = horizon

    def forward(
        self, history_coeff: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_coeff.ndim != 3 or params.ndim != 2:
            raise ValueError("history_coeff must be [B, K, R] and params must be [B, P]")
        expanded_params = params.unsqueeze(1).expand(-1, history_coeff.shape[1], -1)
        inputs = torch.cat((history_coeff, expanded_params), dim=-1).transpose(1, 2)
        latent = self.encoder(inputs)[:, :, -1]
        residual = self.coefficient_head(latent).reshape(-1, self.horizon, self.n_modes)
        coefficients = history_coeff[:, -1].unsqueeze(1) + residual
        macros = self.macro_head(latent).reshape(-1, self.horizon, 2)
        return coefficients, macros


class LSTMForecaster(nn.Module):
    """POD-LSTM baseline with the same input/output contract as the TCN."""

    def __init__(
        self,
        n_modes: int,
        n_params: int,
        horizon: int,
        channels: int = 128,
        levels: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_modes + n_params,
            hidden_size=channels,
            num_layers=levels,
            dropout=dropout if levels > 1 else 0.0,
            batch_first=True,
        )
        self.coefficient_head = nn.Linear(channels, horizon * n_modes)
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)
        self.macro_head = nn.Linear(channels, horizon * 2)
        self.n_modes = n_modes
        self.horizon = horizon

    def forward(
        self, history_coeff: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_params = params.unsqueeze(1).expand(-1, history_coeff.shape[1], -1)
        inputs = torch.cat((history_coeff, expanded_params), dim=-1)
        encoded, _ = self.encoder(inputs)
        latent = encoded[:, -1]
        residual = self.coefficient_head(latent).reshape(-1, self.horizon, self.n_modes)
        coefficients = history_coeff[:, -1].unsqueeze(1) + residual
        macros = self.macro_head(latent).reshape(-1, self.horizon, 2)
        return coefficients, macros


def build_model(
    model_config: dict[str, Any],
    n_modes: int,
    n_params: int,
    horizon: int,
) -> nn.Module:
    model_type = str(model_config.get("type", "tcn")).lower()
    common = dict(
        n_modes=n_modes,
        n_params=n_params,
        horizon=horizon,
        channels=int(model_config.get("channels", 128)),
        levels=int(model_config.get("levels", 4)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    if model_type == "tcn":
        return TCNForecaster(kernel_size=int(model_config.get("kernel_size", 3)), **common)
    if model_type == "lstm":
        return LSTMForecaster(**common)
    raise ValueError(f"Unsupported model type: {model_type}")
