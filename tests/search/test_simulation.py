from src.nerual_networks.neural_network import DynamicsNetwork, PredictionNetwork
from src.search.nodes import Node
from src.search.simulation import MuZeroSimulation
import torch


def test_muzero_simulation():
    """No actual rollouts, only predictions from current latent state"""
    num_actions = 2
    latent_dim = 10
    depth = 5
    predictor = PredictionNetwork(latent_dim=latent_dim, num_actions=num_actions)
    dynamic_network = DynamicsNetwork(latent_dim=latent_dim, num_actions=num_actions)
    simulation = MuZeroSimulation(dynamic_network, predictor, depth=depth)
    node = Node(torch.randn(latent_dim))
    rewards = simulation(node)
    assert isinstance(rewards, list)
    assert len(rewards) == depth + 1
    assert all(isinstance(reward, float) for reward in rewards)


def test_alphazero_simulation():
    """Rollouts with real moves to terminal and return winner"""
