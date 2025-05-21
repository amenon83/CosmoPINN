"""Network architecture and PINN loss utilities."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .equations import residuals


class MLP(nn.Module):
    """Simple fully connected neural network with configurable layers."""

    def __init__(self, layers: List[int], activation: str = "tanh") -> None:
        super().__init__()
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "sin": nn.SiLU,  # using SiLU as placeholder for sine-like nonlinearity
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation {activation}")
        act_cls = activations[activation]

        modules: List[nn.Module] = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(act_cls())
        modules.append(nn.Linear(layers[-2], layers[-1]))  # output layer
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, in_features)
        return self.model(x)


class CosmicPINN(nn.Module):
    """Physics-Informed Neural Network for 2-D cosmological fluid."""

    def __init__(self, layers: List[int], activation: str = "tanh") -> None:
        super().__init__()
        self.net = MLP(layers, activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def loss_pde(self, collocation_pts: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(collocation_pts)
        res = residuals(collocation_pts, outputs)
        loss = sum([(r ** 2).mean() for r in res])
        return loss

    def loss_ic(self, ic_inputs: torch.Tensor, ic_targets: torch.Tensor) -> torch.Tensor:
        preds = self.forward(ic_inputs)
        return nn.functional.mse_loss(preds, ic_targets)

    def total_loss(
        self,
        collocation_pts: torch.Tensor,
        ic_inputs: torch.Tensor,
        ic_targets: torch.Tensor,
        lambda_ic: float = 1.0,
    ) -> torch.Tensor:
        return self.loss_pde(collocation_pts) + lambda_ic * self.loss_ic(ic_inputs, ic_targets) 