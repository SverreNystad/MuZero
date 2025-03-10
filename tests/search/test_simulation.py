from src.neural_network import PredictionNetwork
from src.search.nodes import Node
from src.search.simulation import MuZeroSimulation
import torch


def test_muzero_simulation():
    """No actual rollouts, only predictions from current latent state"""
    predictor = PredictionNetwork(latent_dim=10, num_actions=2)
    simulation = MuZeroSimulation(predictor)
    node = Node(torch.randn(1, 10))
    value = simulation(node)
    assert isinstance(value, float)


def test_alphazero_simulation():
    """Rollouts with real moves to terminal and return winner"""
