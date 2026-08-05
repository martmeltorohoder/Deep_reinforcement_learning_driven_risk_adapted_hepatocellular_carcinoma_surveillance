import numpy as np
import torch

from scdsmdp.learning.agent import SCDSMDPAgent
from scdsmdp.learning.network import ImplicitQuantileNetwork
from scdsmdp.learning.objectives import conservative_penalty, quantile_huber_loss
from scdsmdp.learning.replay import Batch
from scdsmdp.specification import load_spec


def test_iqn_output_shape() -> None:
    spec = load_spec("settings/test.yaml")
    network = ImplicitQuantileNetwork(spec.model)
    states = torch.randn(5, 14)
    taus = torch.rand(5, 8)
    assert network(states, taus).shape == (5, 8, 18)


def test_cvar_output_shape() -> None:
    spec = load_spec("settings/test.yaml")
    network = ImplicitQuantileNetwork(spec.model)
    assert network.cvar(torch.randn(3, 14), 0.05).shape == (3, 18)


def test_quantile_loss_is_finite() -> None:
    predictions = torch.randn(4, 8)
    targets = torch.randn(4, 8)
    quantiles = torch.rand(4, 8)
    loss = quantile_huber_loss(predictions, targets, quantiles)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_conservative_penalty_is_finite() -> None:
    values = torch.randn(4, 8, 18)
    actions = torch.tensor([0, 1, 2, 3])
    assert torch.isfinite(conservative_penalty(values, actions))


def test_agent_respects_mask() -> None:
    spec = load_spec("settings/test.yaml")
    agent = SCDSMDPAgent(spec, torch.device("cpu"))
    observation = np.zeros(14, dtype=np.float32)
    mask = np.zeros(18, dtype=np.bool_)
    mask[11] = True
    assert agent.act(observation, mask) == 11


def test_agent_update_changes_parameters() -> None:
    spec = load_spec("settings/test.yaml")
    agent = SCDSMDPAgent(spec, torch.device("cpu"))
    before = [value.detach().clone() for value in agent.online.parameters()]
    batch = Batch(
        torch.randn(4, 14),
        torch.randint(0, 18, (4,)),
        torch.randn(4),
        torch.randn(4, 14),
        torch.zeros(4),
        torch.zeros(4, 3),
        torch.ones(4, 18, dtype=torch.bool),
    )
    result = agent.update(batch, 1)
    assert np.isfinite(result.loss)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.online.parameters(), strict=True)
    )
