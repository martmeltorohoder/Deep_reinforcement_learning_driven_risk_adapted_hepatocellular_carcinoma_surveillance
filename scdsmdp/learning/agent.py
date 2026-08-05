from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from scdsmdp.learning.network import ImplicitQuantileNetwork
from scdsmdp.learning.objectives import conservative_penalty, quantile_huber_loss
from scdsmdp.learning.replay import Batch
from scdsmdp.specification import ExperimentSpec


@dataclass(frozen=True)
class UpdateResult:
    loss: float
    iqn_loss: float
    cql_loss: float
    constraint_penalty: float
    gradient_norm: float


class SCDSMDPAgent:
    def __init__(self, spec: ExperimentSpec, device: torch.device) -> None:
        self.spec = spec
        self.device = device
        self.online = ImplicitQuantileNetwork(spec.model).to(device)
        self.target = ImplicitQuantileNetwork(spec.model).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=spec.training.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=spec.training.episodes,
        )
        self.multipliers = torch.full((3,), spec.constraints.initial_multiplier, device=device)
        self.thresholds = torch.tensor(
            (
                spec.constraints.missed_cancer,
                spec.constraints.interval_violation,
                spec.constraints.annual_budget,
            ),
            device=device,
        )

    @torch.no_grad()
    def act(
        self,
        observation: NDArray[np.float32],
        mask: NDArray[np.bool_],
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> int:
        generator = rng or np.random.default_rng()
        allowed = np.flatnonzero(mask)
        if generator.random() < epsilon:
            return int(generator.choice(allowed))
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        values = self.online.cvar(state, self.spec.model.cvar_level).squeeze(0)
        action_mask = torch.as_tensor(mask, device=self.device)
        values = values.masked_fill(~action_mask, -torch.inf)
        return int(values.argmax().item())

    def update(self, batch: Batch, step: int) -> UpdateResult:
        values, quantiles = self.online.sample(batch.observations)
        actions = batch.actions.view(-1, 1, 1).expand(-1, values.shape[1], 1)
        chosen = values.gather(2, actions).squeeze(2)
        with torch.no_grad():
            targets = self._targets(batch)
        iqn = quantile_huber_loss(chosen, targets, quantiles)
        cql = conservative_penalty(values, batch.actions)
        violations = batch.constraints.mean(dim=0) - self.thresholds
        constraint = torch.dot(self.multipliers.detach(), violations)
        loss = iqn + self.spec.training.cql_weight * cql + constraint
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            self.spec.training.gradient_clip,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.polyak_update()
        if step % self.spec.constraints.update_frequency == 0:
            self.update_multipliers(batch.constraints)
        return UpdateResult(
            float(loss.detach()),
            float(iqn.detach()),
            float(cql.detach()),
            float(constraint.detach()),
            float(gradient.detach()),
        )

    @torch.no_grad()
    def _targets(self, batch: Batch) -> Tensor:
        next_cvar = self.online.cvar(batch.next_observations, self.spec.model.cvar_level)
        next_cvar = next_cvar.masked_fill(~batch.next_masks, -torch.inf)
        next_actions = next_cvar.argmax(dim=1)
        next_values, _ = self.target.sample(batch.next_observations)
        action_indices = next_actions.view(-1, 1, 1).expand(-1, next_values.shape[1], 1)
        selected = next_values.gather(2, action_indices).squeeze(2)
        return batch.rewards.unsqueeze(1) + (
            self.spec.training.discount * (1.0 - batch.terminated.unsqueeze(1)) * selected
        )

    @torch.no_grad()
    def polyak_update(self) -> None:
        rate = self.spec.training.target_rate
        for target, online in zip(self.target.parameters(), self.online.parameters(), strict=True):
            target.mul_(1.0 - rate).add_(online, alpha=rate)

    @torch.no_grad()
    def update_multipliers(self, costs: Tensor) -> None:
        violation = costs.mean(dim=0) - self.thresholds
        self.multipliers.add_(violation, alpha=self.spec.constraints.dual_learning_rate)
        self.multipliers.clamp_(min=0.0)

    def state_dict(self) -> dict[str, object]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "multipliers": self.multipliers.detach().cpu(),
            "seed": self.spec.seed,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.online.load_state_dict(cast(dict[str, Any], state["online"]))
        self.target.load_state_dict(cast(dict[str, Any], state["target"]))
        self.optimizer.load_state_dict(cast(dict[str, Any], state["optimizer"]))
        self.scheduler.load_state_dict(cast(dict[str, Any], state["scheduler"]))
        multipliers = state["multipliers"]
        if not isinstance(multipliers, Tensor):
            raise TypeError("invalid multipliers")
        self.multipliers.copy_(multipliers.to(self.device))
