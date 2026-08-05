from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from scdsmdp.specification import ModelSpec


class StateEncoder(nn.Module):
    def __init__(self, state_size: int, hidden_size: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = state_size
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(size, hidden_size), nn.ReLU()))
            size = hidden_size
        self.layers = nn.Sequential(*layers)

    def forward(self, states: Tensor) -> Tensor:
        return cast(Tensor, self.layers(states))


class QuantileEmbedding(nn.Module):
    def __init__(self, basis_size: int, hidden_size: int) -> None:
        super().__init__()
        self.basis_size = basis_size
        self.projection = nn.Sequential(nn.Linear(basis_size, hidden_size), nn.ReLU())
        frequencies: Tensor = torch.arange(1, basis_size + 1, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)

    def forward(self, quantiles: Tensor) -> Tensor:
        frequencies = self.get_buffer("frequencies")
        basis = torch.cos(quantiles.unsqueeze(-1) * frequencies)
        return cast(Tensor, self.projection(basis))


class ImplicitQuantileNetwork(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = StateEncoder(spec.state_size, spec.hidden_size, spec.hidden_layers)
        self.quantiles = QuantileEmbedding(spec.quantile_basis, spec.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(spec.hidden_size, spec.hidden_size),
            nn.ReLU(),
            nn.Linear(spec.hidden_size, spec.action_size),
        )

    def forward(self, states: Tensor, quantiles: Tensor) -> Tensor:
        state_embedding = self.encoder(states)
        quantile_embedding = self.quantiles(quantiles)
        combined = state_embedding.unsqueeze(1) * quantile_embedding
        return cast(Tensor, self.head(combined))

    def sample(self, states: Tensor, count: int | None = None) -> tuple[Tensor, Tensor]:
        samples = count or self.spec.quantile_samples
        taus = torch.rand(states.shape[0], samples, device=states.device, dtype=states.dtype)
        return self(states, taus), taus

    def cvar(self, states: Tensor, alpha: float, count: int | None = None) -> Tensor:
        samples = count or self.spec.quantile_samples
        tail_count = max(1, int(samples * alpha))
        grid = torch.linspace(
            alpha / (2.0 * tail_count),
            alpha - alpha / (2.0 * tail_count),
            tail_count,
            device=states.device,
            dtype=states.dtype,
        )
        taus = grid.unsqueeze(0).expand(states.shape[0], -1)
        return cast(Tensor, self(states, taus).mean(dim=1))
