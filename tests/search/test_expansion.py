from src.search.expansion import expand_node
from src.neural_network import DynamicsNetwork
from src.search.nodes import Node
import torch


def test_expanding_leaf_node():
    batch_size = 2
    latent_state = torch.randn(batch_size, 10)
    root = Node(latent_state)
    dynamics_network = DynamicsNetwork(10, 2)
    assert len(root.children) == 0
    possible_actions = [torch.tensor([1] * batch_size), torch.tensor([0] * batch_size)]
    expand_node(root, possible_actions, dynamics_network)
    assert len(root.children) == len(possible_actions)
