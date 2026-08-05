from __future__ import annotations

import torch
from torch import Tensor


def quantile_huber_loss(
    predictions: Tensor,
    targets: Tensor,
    quantiles: Tensor,
    threshold: float = 1.0,
) -> Tensor:
    residual = targets.unsqueeze(1) - predictions.unsqueeze(2)
    absolute = residual.abs()
    huber = torch.where(
        absolute <= threshold,
        0.5 * residual.square(),
        threshold * (absolute - 0.5 * threshold),
    )
    indicator = (residual.detach() < 0.0).to(predictions.dtype)
    weights = (quantiles.unsqueeze(2) - indicator).abs()
    return (weights * huber / threshold).mean()


def conservative_penalty(values: Tensor, selected_actions: Tensor) -> Tensor:
    average = values.mean(dim=1)
    broad = torch.logsumexp(average, dim=-1)
    selected = average.gather(1, selected_actions.unsqueeze(1)).squeeze(1)
    return (broad - selected).mean()


def lagrangian_penalty(
    constraint_costs: Tensor,
    thresholds: Tensor,
    multipliers: Tensor,
) -> Tensor:
    violations = constraint_costs.mean(dim=0) - thresholds
    return torch.dot(multipliers.detach(), violations)
