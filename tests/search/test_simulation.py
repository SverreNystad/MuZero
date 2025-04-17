import torch

from src.search.nodes import Node
from src.search.simulation import MuZeroSimulation
from tests.nerual_networks.test_networks import tiny_dyn_net, tiny_pred_net


def test_muzero_simulation():
    """No actual rollouts, only predictions from current latent state"""
    num_actions = 2
    depth = 5
    latent_shape = (2, 2, 2)
    dyn_net = tiny_dyn_net(latent_shape, num_actions=num_actions)
    pred_net = tiny_pred_net(latent_shape, num_actions=num_actions)

    simulation = MuZeroSimulation(dyn_net, pred_net, depth)
    node = Node(torch.randn(1, *latent_shape))
    rewards = simulation(node)
    assert isinstance(rewards, list)
    assert len(rewards) == depth + 1
    assert all(isinstance(reward, float) for reward in rewards)


def test_alphazero_simulation():
    """Rollouts with real moves to terminal and return winner"""
